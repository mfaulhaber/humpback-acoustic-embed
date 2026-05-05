"""Masked-span transformer training for the Sequence Models track (ADR-061).

The trainer takes per-region CRNN embedding sequences and learns a
contextual encoder via masked-span reconstruction (MSE +/- cosine).
Downstream, the encoder hidden states ``Z`` are clustered into discrete
tokens by :mod:`humpback.sequence_models.tokenization`.

Reference: spec ``docs/specs/2026-05-01-masked-transformer-sequence-model-design.md``
section 4.1.

Mask weighting: when ``mask_weight_bias`` is ``True`` the per-position
loss for a masked frame is multiplied by a tier-derived weight. Empirical
defaults bias the model toward learning event-adjacent context:

- ``event_core``: 1.5
- ``near_event``: 1.2
- ``background``: 0.5

Unmasked frames contribute zero loss regardless of tier, so background
chunks that lie outside any masked span are ignored as before.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from humpback.sequence_models.contrastive_loss import (
    ContrastiveEventMetadata,
    compute_eligible_contrastive_labels,
    supervised_contrastive_loss,
)


PresetName = Literal["small", "default", "large"]


_PRESETS: dict[str, dict[str, int]] = {
    "small": {"d_model": 128, "num_layers": 2, "num_heads": 4, "ff_dim": 512},
    "default": {"d_model": 256, "num_layers": 4, "num_heads": 8, "ff_dim": 1024},
    "large": {"d_model": 384, "num_layers": 6, "num_heads": 8, "ff_dim": 1536},
}


# Tier weighting for the masked-loss bias. Documented in the module
# docstring; consumed only when ``MaskedTransformerConfig.mask_weight_bias``
# is ``True``.
TIER_LOSS_WEIGHTS: dict[str, float] = {
    "event_core": 1.5,
    "near_event": 1.2,
    "background": 0.5,
}
_DEFAULT_TIER_WEIGHT = 1.0


@dataclass(frozen=True)
class MaskedTransformerConfig:
    """Training-time configuration for the masked-span transformer."""

    preset: PresetName = "default"
    mask_fraction: float = 0.20
    span_length_min: int = 2
    span_length_max: int = 6
    dropout: float = 0.1
    mask_weight_bias: bool = True
    cosine_loss_weight: float = 0.0
    retrieval_head_enabled: bool = False
    retrieval_dim: int | None = None
    retrieval_hidden_dim: int | None = None
    retrieval_l2_normalize: bool = True
    sequence_construction_mode: Literal["region", "event_centered", "mixed"] = "region"
    event_centered_fraction: float = 0.0
    pre_event_context_sec: float | None = None
    post_event_context_sec: float | None = None
    contrastive_loss_weight: float = 0.0
    contrastive_temperature: float = 0.07
    contrastive_label_source: Literal["none", "human_corrections"] = "none"
    contrastive_min_events_per_label: int = 4
    contrastive_min_regions_per_label: int = 2
    require_cross_region_positive: bool = True
    related_label_policy_json: str | None = None
    contrastive_sampler_enabled: bool = True
    contrastive_labels_per_batch: int = 4
    contrastive_events_per_label: int = 4
    contrastive_max_unlabeled_fraction: float = 0.25
    contrastive_region_balance: bool = True
    max_epochs: int = 30
    early_stop_patience: int = 3
    val_split: float = 0.1
    seed: int = 42
    batch_size: int = 8

    def preset_dims(self) -> dict[str, int]:
        if self.preset not in _PRESETS:
            known = ", ".join(sorted(_PRESETS))
            raise ValueError(f"unknown preset {self.preset!r}; expected one of {known}")
        return dict(_PRESETS[self.preset])


@dataclass
class TrainResult:
    """Outputs of :func:`train_masked_transformer`."""

    model: "MaskedTransformer"
    loss_curve: dict[str, list[float]]
    val_metrics: dict[str, float]
    training_mask: list[bool]
    reconstruction_error_per_chunk: list[np.ndarray]
    stopped_epoch: int
    n_train_sequences: int
    n_val_sequences: int


@dataclass
class MaskedTransformerForward:
    """Named forward contract with tuple-unpack compatibility."""

    reconstructed: torch.Tensor
    hidden: torch.Tensor
    retrieval: Optional[torch.Tensor] = None

    def __iter__(self):
        yield self.reconstructed
        yield self.hidden


class MaskedTransformer(nn.Module):
    """``Linear -> TransformerEncoder(norm_first, GELU) -> Linear``.

    The encoder operates with batch_first tensors of shape
    ``(batch, T, input_dim)``; ``forward`` returns the reconstructed
    inputs and the post-encoder hidden states.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        retrieval_head_enabled: bool = False,
        retrieval_dim: Optional[int] = None,
        retrieval_hidden_dim: Optional[int] = None,
        retrieval_l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.retrieval_head_enabled = bool(retrieval_head_enabled)
        self.retrieval_dim = int(retrieval_dim) if retrieval_dim is not None else None
        self.retrieval_hidden_dim = (
            int(retrieval_hidden_dim) if retrieval_hidden_dim is not None else None
        )
        self.retrieval_l2_normalize = bool(retrieval_l2_normalize)
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)
        if self.retrieval_head_enabled:
            resolved_dim = self.retrieval_dim or 128
            resolved_hidden = self.retrieval_hidden_dim or 512
            self.retrieval_dim = resolved_dim
            self.retrieval_hidden_dim = resolved_hidden
            self.retrieval_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, resolved_hidden),
                nn.GELU(),
                nn.Linear(resolved_hidden, resolved_dim),
            )
        else:
            self.retrieval_head = None

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> MaskedTransformerForward:
        hidden = self.encoder(
            self.input_proj(x), src_key_padding_mask=src_key_padding_mask
        )
        reconstructed = self.output_proj(hidden)
        retrieval: Optional[torch.Tensor] = None
        head = self.retrieval_head
        if head is not None:
            retrieval = head(hidden)
            if self.retrieval_l2_normalize:
                assert retrieval is not None
                retrieval = F.normalize(retrieval, p=2, dim=-1, eps=1e-12)
        return MaskedTransformerForward(
            reconstructed=reconstructed,
            hidden=hidden,
            retrieval=retrieval,
        )


def apply_span_mask(
    seq: np.ndarray,
    frac: float,
    span_min: int,
    span_max: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask contiguous spans until coverage hits ``frac``.

    The ``frac`` overshoots by up to one span length (capped at
    ``frac + 0.05`` of the sequence) — picking the next span is
    independent of the post-mask coverage, so a single span can put us
    just over the target.

    Masked positions are replaced with the sequence-mean embedding;
    unmasked positions are returned unchanged.

    Returns ``(masked_seq, mask_positions)`` where ``mask_positions`` is
    a boolean array of shape ``(T,)`` with ``True`` at masked frames.
    """
    if frac < 0.0 or frac > 1.0:
        raise ValueError(f"mask fraction must be in [0, 1], got {frac}")
    if span_min <= 0 or span_max < span_min:
        raise ValueError(
            f"span bounds invalid: span_min={span_min} span_max={span_max}"
        )

    seq = np.ascontiguousarray(seq)
    T = seq.shape[0]
    if T == 0:
        return seq.copy(), np.zeros(0, dtype=bool)

    target_masked = int(round(frac * T))
    overshoot_cap = int(math.ceil((frac + 0.05) * T))
    mask = np.zeros(T, dtype=bool)
    n_masked = 0
    # Bound iterations so we never spin forever on a degenerate seq.
    max_iters = max(8 * T, 64)

    for _ in range(max_iters):
        if n_masked >= target_masked:
            break
        if n_masked >= overshoot_cap:
            break
        span_len = int(rng.integers(span_min, span_max + 1))
        span_len = min(span_len, T)
        if span_len <= 0:
            break
        start = int(rng.integers(0, T - span_len + 1))
        end = start + span_len
        n_masked += int((~mask[start:end]).sum())
        mask[start:end] = True

    sequence_mean = seq.mean(axis=0)
    masked_seq = seq.copy()
    if mask.any():
        masked_seq[mask] = sequence_mean.astype(seq.dtype, copy=False)
    return masked_seq, mask


def _device_obj(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _pad_batch(
    sequences: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [int(seq.shape[0]) for seq in sequences]
    max_len = max(lengths)
    feature_dim = sequences[0].shape[1]
    out = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)
    pad_mask = np.ones((len(sequences), max_len), dtype=bool)
    for i, seq in enumerate(sequences):
        L = lengths[i]
        out[i, :L] = seq.astype(np.float32, copy=False)
        pad_mask[i, :L] = False
    return (
        torch.from_numpy(out),
        torch.from_numpy(pad_mask),
        lengths,
    )


def _mask_weights(mask_positions: np.ndarray, tiers: Optional[list[str]]) -> np.ndarray:
    if tiers is None:
        return np.where(mask_positions, 1.0, 0.0).astype(np.float32)
    if len(tiers) != mask_positions.shape[0]:
        raise ValueError(
            f"tier list length {len(tiers)} mismatches sequence length "
            f"{mask_positions.shape[0]}"
        )
    weights = np.zeros(mask_positions.shape[0], dtype=np.float32)
    for i, tier in enumerate(tiers):
        if mask_positions[i]:
            weights[i] = TIER_LOSS_WEIGHTS.get(tier, _DEFAULT_TIER_WEIGHT)
    return weights


@dataclass
class _BatchInputs:
    inputs: torch.Tensor  # (B, T, D)
    targets: torch.Tensor  # (B, T, D)
    pad_mask: torch.Tensor  # (B, T) — True at padded positions
    mask_positions: torch.Tensor  # (B, T) — True at masked positions
    weights: torch.Tensor  # (B, T) — per-position loss weight
    lengths: list[int]


def _build_batch(
    sequences: list[np.ndarray],
    tier_lists: Optional[list[Optional[list[str]]]],
    config: MaskedTransformerConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> _BatchInputs:
    masked_seqs: list[np.ndarray] = []
    mask_lists: list[np.ndarray] = []
    weight_lists: list[np.ndarray] = []
    targets_arr = sequences  # we'll pad after building masks

    for i, seq in enumerate(sequences):
        masked_seq, mask_positions = apply_span_mask(
            seq,
            config.mask_fraction,
            config.span_length_min,
            config.span_length_max,
            rng,
        )
        masked_seqs.append(masked_seq)
        mask_lists.append(mask_positions)
        if config.mask_weight_bias:
            tier_seq = (
                tier_lists[i]
                if tier_lists is not None and i < len(tier_lists)
                else None
            )
            weight_lists.append(_mask_weights(mask_positions, tier_seq))
        else:
            weight_lists.append(mask_positions.astype(np.float32))

    inputs_tensor, pad_mask_tensor, lengths = _pad_batch(masked_seqs)
    targets_tensor, _, _ = _pad_batch(targets_arr)

    max_T = inputs_tensor.shape[1]
    mask_arr = np.zeros((len(sequences), max_T), dtype=bool)
    weights_arr = np.zeros((len(sequences), max_T), dtype=np.float32)
    for i, mp in enumerate(mask_lists):
        L = mp.shape[0]
        mask_arr[i, :L] = mp
        weights_arr[i, :L] = weight_lists[i]

    return _BatchInputs(
        inputs=inputs_tensor.to(device),
        targets=targets_tensor.to(device),
        pad_mask=pad_mask_tensor.to(device),
        mask_positions=torch.from_numpy(mask_arr).to(device),
        weights=torch.from_numpy(weights_arr).to(device),
        lengths=lengths,
    )


def _compute_loss(
    reconstructed: torch.Tensor,
    targets: torch.Tensor,
    mask_positions: torch.Tensor,
    weights: torch.Tensor,
    cosine_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scalar_loss, per_chunk_squared_error).

    Per-chunk squared error is reduced over the feature dim and
    returned without weighting so callers can persist a raw
    reconstruction-error timeline strip.
    """
    diff = reconstructed - targets
    sq_err_per_chunk = (diff * diff).mean(dim=-1)  # (B, T)

    # Weighted MSE over masked positions only.
    mask_pos = mask_positions.to(weights.dtype)
    weighted_sq = sq_err_per_chunk * weights
    denom = (weights * mask_pos).sum().clamp_min(1.0)
    mse_loss = weighted_sq.sum() / denom

    if cosine_weight > 0.0:
        # Cosine loss only on masked positions, mean over them.
        rec_flat = reconstructed[mask_positions]
        tgt_flat = targets[mask_positions]
        if rec_flat.numel() == 0:
            cos = torch.zeros((), device=reconstructed.device)
        else:
            cos = 1.0 - F.cosine_similarity(rec_flat, tgt_flat, dim=-1).mean()
        loss = mse_loss + cosine_weight * cos
    else:
        loss = mse_loss

    return loss, sq_err_per_chunk.detach()


def _compute_retrieval_consistency_loss(
    predicted: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    mask_positions: torch.Tensor,
) -> torch.Tensor:
    """Unsupervised masked-context loss that gives Phase 1 retrieval head gradients."""
    if predicted is None or target is None or not bool(mask_positions.any()):
        reference = predicted if predicted is not None else target
        if reference is None:
            return torch.tensor(0.0, device=mask_positions.device)
        return reference.sum() * 0.0
    diff = (predicted - target.detach()).pow(2).mean(dim=-1)
    mask = mask_positions.to(diff.dtype)
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


def _pool_event_retrieval_embeddings(
    model: MaskedTransformer,
    hidden: torch.Tensor,
    metadata: list[Optional[ContrastiveEventMetadata]] | None,
    lengths: list[int],
) -> tuple[torch.Tensor, list[ContrastiveEventMetadata]]:
    if not metadata or model.retrieval_head is None:
        return hidden.new_zeros((0, model.retrieval_dim or model.d_model)), []
    pooled: list[torch.Tensor] = []
    used: list[ContrastiveEventMetadata] = []
    for i, item in enumerate(metadata):
        if item is None or not item.human_types:
            continue
        length = lengths[i]
        start = max(0, min(int(item.start_index), length))
        end = max(start + 1, min(int(item.end_index), length))
        pooled.append(hidden[i, start:end].mean(dim=0))
        used.append(item)
    if not pooled:
        return hidden.new_zeros((0, model.retrieval_dim or model.d_model)), []
    pooled_hidden = torch.stack(pooled, dim=0)
    retrieval = model.retrieval_head(pooled_hidden)
    if model.retrieval_l2_normalize:
        retrieval = F.normalize(retrieval, p=2, dim=-1, eps=1e-12)
    return retrieval, used


def _train_val_split(
    n_sequences: int, val_split: float, rng: np.random.Generator
) -> tuple[list[bool], list[int], list[int]]:
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    indices = np.arange(n_sequences)
    rng.shuffle(indices)
    n_val = int(round(n_sequences * val_split))
    n_val = min(max(n_val, 0), n_sequences - 1) if n_sequences > 1 else 0
    val_idx = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())
    training_mask = [False] * n_sequences
    for i in train_idx:
        training_mask[i] = True
    return training_mask, train_idx, val_idx


def _contrastive_epoch_order(
    n_train: int,
    events: list[Optional[ContrastiveEventMetadata]] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Prefer labeled, cross-region-friendly examples early in an epoch."""
    if events is None:
        return rng.permutation(n_train)
    labeled = [
        i
        for i, event in enumerate(events)
        if event is not None and bool(event.human_types)
    ]
    labeled_set = set(labeled)
    unlabeled = [i for i in range(n_train) if i not in labeled_set]
    if not labeled:
        return rng.permutation(n_train)

    def _sort_key(i: int) -> tuple[str, str, str]:
        event = events[i]
        if event is None:
            return "", "", ""
        return event.human_types[0], event.region_id, event.event_id

    labeled.sort(key=_sort_key)
    if unlabeled:
        unlabeled = [int(i) for i in rng.permutation(unlabeled)]
    return np.asarray([*labeled, *unlabeled], dtype=int)


def _choose_label_examples(
    candidates: list[int],
    events: Sequence[Optional[ContrastiveEventMetadata]],
    *,
    count: int,
    region_balance: bool,
) -> list[int]:
    if not region_balance:
        return candidates[:count]
    by_region: dict[str, list[int]] = defaultdict(list)
    for idx in candidates:
        event = events[idx]
        if event is not None:
            by_region[event.region_id].append(idx)
    chosen: list[int] = []
    while len(chosen) < count and by_region:
        for region in sorted(list(by_region)):
            region_items = by_region[region]
            if not region_items:
                del by_region[region]
                continue
            chosen.append(region_items.pop(0))
            if len(chosen) >= count:
                break
    return chosen


def build_contrastive_epoch_batches(
    n_train: int,
    events: Sequence[Optional[ContrastiveEventMetadata]] | None,
    rng: np.random.Generator,
    *,
    batch_size: int,
    min_events_per_label: int = 4,
    min_regions_per_label: int = 2,
    labels_per_batch: int = 4,
    events_per_label: int = 4,
    max_unlabeled_fraction: float = 0.25,
    region_balance: bool = True,
) -> list[list[int]]:
    """Plan deterministic contrastive-friendly batches for one train epoch."""
    if n_train <= 0:
        return []
    resolved_batch_size = max(1, int(batch_size))
    if not events:
        order = [int(i) for i in rng.permutation(n_train)]
        return [
            order[start : start + resolved_batch_size]
            for start in range(0, len(order), resolved_batch_size)
        ]
    metadata = [event for event in events if event is not None and event.human_types]
    eligible_labels = compute_eligible_contrastive_labels(
        metadata,
        min_events_per_label=min_events_per_label,
        min_regions_per_label=min_regions_per_label,
    )
    if not eligible_labels:
        order = [int(i) for i in rng.permutation(n_train)]
        return [
            order[start : start + resolved_batch_size]
            for start in range(0, len(order), resolved_batch_size)
        ]

    label_to_indices: dict[str, list[int]] = {label: [] for label in eligible_labels}
    eligible_labeled_indices: set[int] = set()
    rare_or_unlabeled: list[int] = []
    for idx, event in enumerate(events):
        if event is None or not event.human_types:
            rare_or_unlabeled.append(idx)
            continue
        event_labels = set(event.human_types) & eligible_labels
        if not event_labels:
            rare_or_unlabeled.append(idx)
            continue
        eligible_labeled_indices.add(idx)
        for label in event_labels:
            label_to_indices[label].append(idx)

    for label, indices in label_to_indices.items():
        shuffled = [int(i) for i in rng.permutation(indices)]
        label_to_indices[label] = shuffled
    rare_or_unlabeled = [int(i) for i in rng.permutation(rare_or_unlabeled)]

    unused_eligible = set(eligible_labeled_indices)
    rare_cursor = 0
    label_order = [str(label) for label in rng.permutation(sorted(eligible_labels))]
    label_cursor = 0
    batches: list[list[int]] = []
    per_label_count = max(2, int(events_per_label))
    max_fill = int(math.floor(resolved_batch_size * float(max_unlabeled_fraction)))

    while True:
        available_labels = [
            label
            for label in label_order
            if len([idx for idx in label_to_indices[label] if idx in unused_eligible])
            >= 2
        ]
        if not available_labels:
            break
        selected_labels: list[str] = []
        attempts = 0
        while (
            len(selected_labels) < max(1, int(labels_per_batch))
            and attempts < len(label_order) * 2
        ):
            label = label_order[label_cursor % len(label_order)]
            label_cursor += 1
            attempts += 1
            if label not in available_labels or label in selected_labels:
                continue
            selected_labels.append(label)
        if not selected_labels:
            break

        batch: list[int] = []
        for label in selected_labels:
            remaining = [
                idx for idx in label_to_indices[label] if idx in unused_eligible
            ]
            capacity = resolved_batch_size - len(batch)
            if capacity < 2:
                break
            chosen = _choose_label_examples(
                remaining,
                events,
                count=min(per_label_count, capacity, len(remaining)),
                region_balance=region_balance,
            )
            for idx in chosen:
                if idx not in batch:
                    batch.append(idx)
                    unused_eligible.discard(idx)
        if len(batch) < 2:
            for idx in batch:
                unused_eligible.add(idx)
            break

        fill_limit = min(max_fill, resolved_batch_size - len(batch))
        while fill_limit > 0 and rare_cursor < len(rare_or_unlabeled):
            batch.append(rare_or_unlabeled[rare_cursor])
            rare_cursor += 1
            fill_limit -= 1
        batches.append(batch)

    remainder = [*sorted(unused_eligible), *rare_or_unlabeled[rare_cursor:]]
    if remainder:
        remainder = [int(i) for i in rng.permutation(remainder)]
        batches.extend(
            [
                remainder[start : start + resolved_batch_size]
                for start in range(0, len(remainder), resolved_batch_size)
            ]
        )
    return batches


def train_masked_transformer(
    sequences: list[np.ndarray],
    config: MaskedTransformerConfig,
    device: str | torch.device = "cpu",
    *,
    tier_lists: Optional[list[Optional[list[str]]]] = None,
    contrastive_events: Optional[Sequence[Optional[ContrastiveEventMetadata]]] = None,
) -> TrainResult:
    """Train a masked-span transformer on per-region CRNN embeddings.

    Parameters
    ----------
    sequences
        List of ``(T, D)`` float arrays, one per CRNN region. Sequences
        keep their original ordering; ``training_mask`` returned in
        :class:`TrainResult` aligns with this list 1:1.
    config
        :class:`MaskedTransformerConfig`.
    device
        Torch device string; the worker decides whether to run on CPU or
        an accelerator (validation + fallback live there).
    tier_lists
        Optional per-sequence list of tier strings used for the
        mask-weight bias. Must align in shape with each sequence's chunk
        axis when provided. Pass ``None`` to disable bias regardless of
        ``config.mask_weight_bias``.
    contrastive_events
        Optional per-sequence event metadata. When contrastive loss is
        enabled, entries with human labels contribute event-level retrieval
        embeddings; missing or unlabeled entries remain masked-modeling-only.
    """
    if not sequences:
        raise ValueError("masked-transformer training requires at least one sequence")

    feature_dim = int(sequences[0].shape[1])
    for seq in sequences:
        if seq.shape[1] != feature_dim:
            raise ValueError("all sequences must share the same feature dimension")
    if contrastive_events is not None and len(contrastive_events) != len(sequences):
        raise ValueError("contrastive_events must align with sequences")
    contrastive_enabled = (
        config.contrastive_loss_weight > 0.0
        and config.contrastive_label_source == "human_corrections"
    )
    if contrastive_enabled and not config.retrieval_head_enabled:
        raise ValueError("contrastive training requires retrieval_head_enabled=true")

    dims = config.preset_dims()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device_obj = _device_obj(device)

    model = MaskedTransformer(
        input_dim=feature_dim,
        d_model=dims["d_model"],
        num_layers=dims["num_layers"],
        num_heads=dims["num_heads"],
        ff_dim=dims["ff_dim"],
        dropout=config.dropout,
        retrieval_head_enabled=config.retrieval_head_enabled,
        retrieval_dim=config.retrieval_dim,
        retrieval_hidden_dim=config.retrieval_hidden_dim,
        retrieval_l2_normalize=config.retrieval_l2_normalize,
    ).to(device_obj)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    training_mask, train_idx, val_idx = _train_val_split(
        len(sequences), config.val_split, rng
    )
    train_seqs = [sequences[i] for i in train_idx]
    train_tiers = [tier_lists[i] for i in train_idx] if tier_lists is not None else None
    train_events = (
        [contrastive_events[i] for i in train_idx]
        if contrastive_events is not None
        else None
    )
    val_seqs = [sequences[i] for i in val_idx]
    val_tiers = [tier_lists[i] for i in val_idx] if tier_lists is not None else None
    val_events = (
        [contrastive_events[i] for i in val_idx]
        if contrastive_events is not None
        else None
    )

    if config.mask_weight_bias and tier_lists is None:
        # No tier metadata supplied — fall back to uniform weighting on
        # masked positions.
        train_tiers = None
        val_tiers = None

    n_train = len(train_seqs)
    if n_train == 0:
        raise ValueError("training split is empty after val split")

    loss_curve_train: list[float] = []
    loss_curve_val: list[float] = []
    loss_curve_train_masked: list[float] = []
    loss_curve_val_masked: list[float] = []
    loss_curve_train_contrastive: list[float] = []
    loss_curve_val_contrastive: list[float] = []
    loss_curve_train_skipped: list[float] = []
    loss_curve_val_skipped: list[float] = []
    loss_curve_train_valid_batches: list[float] = []
    loss_curve_train_valid_anchor_count: list[float] = []
    loss_curve_train_positive_pair_count: list[float] = []
    loss_curve_train_eligible_label_count: list[float] = []
    loss_curve_train_labeled_event_count: list[float] = []
    loss_curve_train_unlabeled_fill_count: list[float] = []
    train_eligible_labels: set[str] = set()
    if contrastive_enabled and train_events is not None:
        train_eligible_labels = compute_eligible_contrastive_labels(
            [
                event
                for event in train_events
                if event is not None and event.human_types
            ],
            min_events_per_label=config.contrastive_min_events_per_label,
            min_regions_per_label=config.contrastive_min_regions_per_label,
        )
    best_val_loss = math.inf
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    stopped_epoch = config.max_epochs

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        if contrastive_enabled and config.contrastive_sampler_enabled:
            epoch_batches = build_contrastive_epoch_batches(
                n_train,
                train_events,
                rng,
                batch_size=config.batch_size,
                min_events_per_label=config.contrastive_min_events_per_label,
                min_regions_per_label=config.contrastive_min_regions_per_label,
                labels_per_batch=config.contrastive_labels_per_batch,
                events_per_label=config.contrastive_events_per_label,
                max_unlabeled_fraction=config.contrastive_max_unlabeled_fraction,
                region_balance=config.contrastive_region_balance,
            )
        else:
            perm = (
                _contrastive_epoch_order(n_train, train_events, rng)
                if contrastive_enabled
                else rng.permutation(n_train)
            )
            epoch_batches = [
                [int(i) for i in perm[start : start + config.batch_size]]
                for start in range(0, n_train, max(1, config.batch_size))
            ]
        epoch_train_loss = 0.0
        epoch_train_masked_loss = 0.0
        epoch_train_contrastive_loss = 0.0
        epoch_train_skipped = 0
        epoch_train_valid_batches = 0
        epoch_train_valid_anchor_count = 0
        epoch_train_positive_pair_count = 0
        epoch_train_labeled_event_count = 0
        epoch_train_unlabeled_fill_count = 0
        n_train_batches = 0

        for batch_idx in epoch_batches:
            batch_seqs = [train_seqs[i] for i in batch_idx]
            batch_tiers = (
                [train_tiers[i] for i in batch_idx] if train_tiers is not None else None
            )
            batch_events = (
                [train_events[i] for i in batch_idx]
                if train_events is not None
                else None
            )
            batch = _build_batch(batch_seqs, batch_tiers, config, rng, device_obj)

            optim.zero_grad(set_to_none=True)
            output = model(batch.inputs, src_key_padding_mask=batch.pad_mask)
            masked_loss, _ = _compute_loss(
                output.reconstructed,
                batch.targets,
                batch.mask_positions,
                batch.weights,
                config.cosine_loss_weight,
            )
            loss = masked_loss
            contrastive_loss = output.hidden.sum() * 0.0
            if contrastive_enabled:
                event_embeddings, event_metadata = _pool_event_retrieval_embeddings(
                    model, output.hidden, batch_events, batch.lengths
                )
                contrastive_loss, masks = supervised_contrastive_loss(
                    event_embeddings,
                    event_metadata,
                    temperature=config.contrastive_temperature,
                    min_events_per_label=config.contrastive_min_events_per_label,
                    min_regions_per_label=config.contrastive_min_regions_per_label,
                    require_cross_region_positive=config.require_cross_region_positive,
                    related_label_policy_json=config.related_label_policy_json,
                    eligible_labels=train_eligible_labels,
                )
                if masks.valid_anchor_count == 0:
                    epoch_train_skipped += 1
                else:
                    epoch_train_valid_batches += 1
                epoch_train_valid_anchor_count += masks.valid_anchor_count
                epoch_train_positive_pair_count += int(masks.positive_mask.sum().item())
                epoch_train_labeled_event_count += len(event_metadata)
                epoch_train_unlabeled_fill_count += len(batch_idx) - len(event_metadata)
                loss = loss + config.contrastive_loss_weight * contrastive_loss
            if config.retrieval_head_enabled:
                with torch.no_grad():
                    target_output = model(
                        batch.targets, src_key_padding_mask=batch.pad_mask
                    )
                retrieval_loss = _compute_retrieval_consistency_loss(
                    output.retrieval,
                    target_output.retrieval,
                    batch.mask_positions,
                )
                if not contrastive_enabled:
                    loss = loss + 0.01 * retrieval_loss
            loss.backward()
            optim.step()
            epoch_train_loss += float(loss.detach().item())
            epoch_train_masked_loss += float(masked_loss.detach().item())
            epoch_train_contrastive_loss += float(contrastive_loss.detach().item())
            n_train_batches += 1

        train_loss = epoch_train_loss / max(1, n_train_batches)
        loss_curve_train.append(train_loss)
        loss_curve_train_masked.append(
            epoch_train_masked_loss / max(1, n_train_batches)
        )
        loss_curve_train_contrastive.append(
            epoch_train_contrastive_loss / max(1, n_train_batches)
        )
        loss_curve_train_skipped.append(float(epoch_train_skipped))
        loss_curve_train_valid_batches.append(float(epoch_train_valid_batches))
        loss_curve_train_valid_anchor_count.append(
            float(epoch_train_valid_anchor_count)
        )
        loss_curve_train_positive_pair_count.append(
            float(epoch_train_positive_pair_count)
        )
        loss_curve_train_eligible_label_count.append(float(len(train_eligible_labels)))
        loss_curve_train_labeled_event_count.append(
            float(epoch_train_labeled_event_count)
        )
        loss_curve_train_unlabeled_fill_count.append(
            float(epoch_train_unlabeled_fill_count)
        )

        # ---- Validation ----
        if val_seqs:
            model.eval()
            with torch.no_grad():
                val_batch = _build_batch(val_seqs, val_tiers, config, rng, device_obj)
                output = model(
                    val_batch.inputs, src_key_padding_mask=val_batch.pad_mask
                )
                val_masked_loss, _ = _compute_loss(
                    output.reconstructed,
                    val_batch.targets,
                    val_batch.mask_positions,
                    val_batch.weights,
                    config.cosine_loss_weight,
                )
                val_loss = val_masked_loss
                val_contrastive_loss = output.hidden.sum() * 0.0
                val_skipped = 0.0
                if contrastive_enabled:
                    event_embeddings, event_metadata = _pool_event_retrieval_embeddings(
                        model, output.hidden, val_events, val_batch.lengths
                    )
                    val_contrastive_loss, masks = supervised_contrastive_loss(
                        event_embeddings,
                        event_metadata,
                        temperature=config.contrastive_temperature,
                        min_events_per_label=config.contrastive_min_events_per_label,
                        min_regions_per_label=config.contrastive_min_regions_per_label,
                        require_cross_region_positive=config.require_cross_region_positive,
                        related_label_policy_json=config.related_label_policy_json,
                    )
                    val_skipped = 1.0 if masks.valid_anchor_count == 0 else 0.0
                    val_loss = (
                        val_loss + config.contrastive_loss_weight * val_contrastive_loss
                    )
                if config.retrieval_head_enabled:
                    target_output = model(
                        val_batch.targets, src_key_padding_mask=val_batch.pad_mask
                    )
                    retrieval_loss = _compute_retrieval_consistency_loss(
                        output.retrieval,
                        target_output.retrieval,
                        val_batch.mask_positions,
                    )
                    if not contrastive_enabled:
                        val_loss = val_loss + 0.01 * retrieval_loss
            val_loss_value = float(val_loss.item())
            val_masked_loss_value = float(val_masked_loss.item())
            val_contrastive_loss_value = float(val_contrastive_loss.item())
            val_skipped_value = float(val_skipped)
        else:
            val_loss_value = float("nan")
            val_masked_loss_value = float("nan")
            val_contrastive_loss_value = 0.0
            val_skipped_value = 0.0
        loss_curve_val.append(val_loss_value)
        loss_curve_val_masked.append(val_masked_loss_value)
        loss_curve_val_contrastive.append(val_contrastive_loss_value)
        loss_curve_val_skipped.append(val_skipped_value)

        # Early stopping on val plateau (NaN treated as no-improve).
        improved = (
            val_seqs
            and not math.isnan(val_loss_value)
            and val_loss_value < best_val_loss - 1e-6
        )
        if improved:
            best_val_loss = val_loss_value
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if val_seqs and epochs_no_improve >= config.early_stop_patience:
                stopped_epoch = epoch
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    reconstruction_error_per_chunk = compute_reconstruction_error(
        model,
        sequences,
        config,
        device=device_obj,
        tier_lists=tier_lists,
    )

    val_metrics: dict[str, float] = {
        "best_val_loss": float(best_val_loss)
        if math.isfinite(best_val_loss)
        else float("nan"),
        "final_train_loss": float(loss_curve_train[-1])
        if loss_curve_train
        else float("nan"),
        "final_val_loss": float(loss_curve_val[-1]) if loss_curve_val else float("nan"),
        "stopped_epoch": float(stopped_epoch),
        "final_train_masked_loss": float(loss_curve_train_masked[-1])
        if loss_curve_train_masked
        else float("nan"),
        "final_train_contrastive_loss": float(loss_curve_train_contrastive[-1])
        if loss_curve_train_contrastive
        else 0.0,
        "final_train_total_loss": float(loss_curve_train[-1])
        if loss_curve_train
        else float("nan"),
        "final_val_masked_loss": float(loss_curve_val_masked[-1])
        if loss_curve_val_masked
        else float("nan"),
        "final_val_contrastive_loss": float(loss_curve_val_contrastive[-1])
        if loss_curve_val_contrastive
        else 0.0,
        "final_val_total_loss": float(loss_curve_val[-1])
        if loss_curve_val
        else float("nan"),
        "final_train_contrastive_skipped_batches": float(loss_curve_train_skipped[-1])
        if loss_curve_train_skipped
        else 0.0,
        "final_train_contrastive_valid_batches": float(
            loss_curve_train_valid_batches[-1]
        )
        if loss_curve_train_valid_batches
        else 0.0,
        "final_train_contrastive_valid_anchor_count": float(
            loss_curve_train_valid_anchor_count[-1]
        )
        if loss_curve_train_valid_anchor_count
        else 0.0,
        "final_train_contrastive_positive_pair_count": float(
            loss_curve_train_positive_pair_count[-1]
        )
        if loss_curve_train_positive_pair_count
        else 0.0,
        "final_val_contrastive_skipped_batches": float(loss_curve_val_skipped[-1])
        if loss_curve_val_skipped
        else 0.0,
    }

    return TrainResult(
        model=model,
        loss_curve={
            "train": loss_curve_train,
            "val": loss_curve_val,
            "train_masked": loss_curve_train_masked,
            "val_masked": loss_curve_val_masked,
            "train_contrastive": loss_curve_train_contrastive,
            "val_contrastive": loss_curve_val_contrastive,
            "train_total": loss_curve_train,
            "val_total": loss_curve_val,
            "train_contrastive_skipped_batches": loss_curve_train_skipped,
            "val_contrastive_skipped_batches": loss_curve_val_skipped,
            "train_contrastive_valid_batches": loss_curve_train_valid_batches,
            "train_contrastive_valid_anchor_count": loss_curve_train_valid_anchor_count,
            "train_contrastive_positive_pair_count": loss_curve_train_positive_pair_count,
            "train_contrastive_eligible_label_count": loss_curve_train_eligible_label_count,
            "train_contrastive_labeled_event_count": loss_curve_train_labeled_event_count,
            "train_contrastive_unlabeled_fill_count": loss_curve_train_unlabeled_fill_count,
        },
        val_metrics=val_metrics,
        training_mask=training_mask,
        reconstruction_error_per_chunk=reconstruction_error_per_chunk,
        stopped_epoch=stopped_epoch,
        n_train_sequences=len(train_idx),
        n_val_sequences=len(val_idx),
    )


def compute_reconstruction_error(
    model: MaskedTransformer,
    sequences: list[np.ndarray],
    config: MaskedTransformerConfig,
    device: str | torch.device = "cpu",
    *,
    tier_lists: Optional[list[Optional[list[str]]]] = None,
) -> list[np.ndarray]:
    """Compute deterministic per-chunk reconstruction error for sequences."""
    device_obj = _device_obj(device)
    model = model.to(device_obj)
    model.eval()
    reconstruction_error_per_chunk: list[np.ndarray] = []
    val_batch_seqs: list[np.ndarray] = list(sequences)
    val_batch_tiers = list(tier_lists) if tier_lists is not None else None
    error_rng = np.random.default_rng(config.seed + 1)
    with torch.no_grad():
        bs = max(1, config.batch_size)
        for start in range(0, len(val_batch_seqs), bs):
            batch_seqs = val_batch_seqs[start : start + bs]
            batch_tiers = (
                val_batch_tiers[start : start + bs]
                if val_batch_tiers is not None
                else None
            )
            batch = _build_batch(batch_seqs, batch_tiers, config, error_rng, device_obj)
            output = model(batch.inputs, src_key_padding_mask=batch.pad_mask)
            _, per_chunk = _compute_loss(
                output.reconstructed,
                batch.targets,
                batch.mask_positions,
                batch.weights,
                config.cosine_loss_weight,
            )
            per_chunk_np = per_chunk.cpu().numpy()
            for i, length in enumerate(batch.lengths):
                reconstruction_error_per_chunk.append(
                    per_chunk_np[i, :length].astype(np.float32, copy=True)
                )
    return reconstruction_error_per_chunk


def extract_contextual_embeddings(
    model: MaskedTransformer,
    sequences: list[np.ndarray],
    device: str | torch.device = "cpu",
    *,
    batch_size: int = 8,
) -> tuple[list[np.ndarray], list[int]]:
    """Run encoder forward (no masking) and return per-sequence hidden states.

    Returns ``(Z, lengths)`` where ``Z`` is a list aligned 1:1 with
    ``sequences`` and each entry is shape ``(T_i, d_model)``.
    """
    Z, _, lengths = extract_transformer_embeddings(
        model,
        sequences,
        device=device,
        batch_size=batch_size,
    )
    return Z, lengths


def extract_transformer_embeddings(
    model: MaskedTransformer,
    sequences: list[np.ndarray],
    device: str | torch.device = "cpu",
    *,
    batch_size: int = 8,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[int]]:
    """Run encoder forward and return contextual plus optional retrieval embeddings."""
    if not sequences:
        return [], [] if model.retrieval_head_enabled else None, []

    device_obj = _device_obj(device)
    model = model.to(device_obj)
    model.eval()

    Z: list[np.ndarray] = []
    R: list[np.ndarray] | None = [] if model.retrieval_head_enabled else None
    lengths: list[int] = [int(seq.shape[0]) for seq in sequences]
    with torch.no_grad():
        for start in range(0, len(sequences), max(1, batch_size)):
            batch_seqs = sequences[start : start + batch_size]
            inputs_tensor, pad_mask, batch_lengths = _pad_batch(batch_seqs)
            inputs_tensor = inputs_tensor.to(device_obj)
            pad_mask = pad_mask.to(device_obj)
            output = model(inputs_tensor, src_key_padding_mask=pad_mask)
            hidden_np = output.hidden.cpu().numpy()
            retrieval_np = (
                output.retrieval.cpu().numpy() if output.retrieval is not None else None
            )
            for i, L in enumerate(batch_lengths):
                Z.append(hidden_np[i, :L].astype(np.float32, copy=True))
                if R is not None and retrieval_np is not None:
                    R.append(retrieval_np[i, :L].astype(np.float32, copy=True))
    return Z, R, lengths


__all__ = [
    "MaskedTransformer",
    "MaskedTransformerConfig",
    "MaskedTransformerForward",
    "TIER_LOSS_WEIGHTS",
    "TrainResult",
    "apply_span_mask",
    "compute_reconstruction_error",
    "extract_contextual_embeddings",
    "extract_transformer_embeddings",
    "train_masked_transformer",
]
