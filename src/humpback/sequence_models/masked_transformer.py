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
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
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

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            self.input_proj(x), src_key_padding_mask=src_key_padding_mask
        )
        reconstructed = self.output_proj(hidden)
        return reconstructed, hidden


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


def train_masked_transformer(
    sequences: list[np.ndarray],
    config: MaskedTransformerConfig,
    device: str | torch.device = "cpu",
    *,
    tier_lists: Optional[list[Optional[list[str]]]] = None,
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
    """
    if not sequences:
        raise ValueError("masked-transformer training requires at least one sequence")

    feature_dim = int(sequences[0].shape[1])
    for seq in sequences:
        if seq.shape[1] != feature_dim:
            raise ValueError("all sequences must share the same feature dimension")

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
    ).to(device_obj)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    training_mask, train_idx, val_idx = _train_val_split(
        len(sequences), config.val_split, rng
    )
    train_seqs = [sequences[i] for i in train_idx]
    train_tiers = [tier_lists[i] for i in train_idx] if tier_lists is not None else None
    val_seqs = [sequences[i] for i in val_idx]
    val_tiers = [tier_lists[i] for i in val_idx] if tier_lists is not None else None

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
    best_val_loss = math.inf
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    stopped_epoch = config.max_epochs

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        epoch_train_loss = 0.0
        n_train_batches = 0

        for start in range(0, n_train, max(1, config.batch_size)):
            batch_idx = perm[start : start + config.batch_size]
            batch_seqs = [train_seqs[i] for i in batch_idx]
            batch_tiers = (
                [train_tiers[i] for i in batch_idx] if train_tiers is not None else None
            )
            batch = _build_batch(batch_seqs, batch_tiers, config, rng, device_obj)

            optim.zero_grad(set_to_none=True)
            reconstructed, _ = model(batch.inputs, src_key_padding_mask=batch.pad_mask)
            loss, _ = _compute_loss(
                reconstructed,
                batch.targets,
                batch.mask_positions,
                batch.weights,
                config.cosine_loss_weight,
            )
            loss.backward()
            optim.step()
            epoch_train_loss += float(loss.detach().item())
            n_train_batches += 1

        train_loss = epoch_train_loss / max(1, n_train_batches)
        loss_curve_train.append(train_loss)

        # ---- Validation ----
        if val_seqs:
            model.eval()
            with torch.no_grad():
                val_batch = _build_batch(val_seqs, val_tiers, config, rng, device_obj)
                reconstructed, _ = model(
                    val_batch.inputs, src_key_padding_mask=val_batch.pad_mask
                )
                val_loss, _ = _compute_loss(
                    reconstructed,
                    val_batch.targets,
                    val_batch.mask_positions,
                    val_batch.weights,
                    config.cosine_loss_weight,
                )
            val_loss_value = float(val_loss.item())
        else:
            val_loss_value = float("nan")
        loss_curve_val.append(val_loss_value)

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

    # ---- Per-chunk reconstruction error on validation pass over all sequences ----
    model.eval()
    reconstruction_error_per_chunk: list[np.ndarray] = []
    val_batch_seqs: list[np.ndarray] = list(sequences)
    val_batch_tiers = list(tier_lists) if tier_lists is not None else None
    # Use a deterministic mask (same seed) so reconstruction error is
    # reproducible across runs of an extend-k-sweep.
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
            reconstructed, _ = model(batch.inputs, src_key_padding_mask=batch.pad_mask)
            _, per_chunk = _compute_loss(
                reconstructed,
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

    val_metrics: dict[str, float] = {
        "best_val_loss": float(best_val_loss)
        if math.isfinite(best_val_loss)
        else float("nan"),
        "final_train_loss": float(loss_curve_train[-1])
        if loss_curve_train
        else float("nan"),
        "final_val_loss": float(loss_curve_val[-1]) if loss_curve_val else float("nan"),
        "stopped_epoch": float(stopped_epoch),
    }

    return TrainResult(
        model=model,
        loss_curve={
            "train": loss_curve_train,
            "val": loss_curve_val,
        },
        val_metrics=val_metrics,
        training_mask=training_mask,
        reconstruction_error_per_chunk=reconstruction_error_per_chunk,
        stopped_epoch=stopped_epoch,
        n_train_sequences=len(train_idx),
        n_val_sequences=len(val_idx),
    )


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
    if not sequences:
        return [], []

    device_obj = _device_obj(device)
    model = model.to(device_obj)
    model.eval()

    Z: list[np.ndarray] = []
    lengths: list[int] = [int(seq.shape[0]) for seq in sequences]
    with torch.no_grad():
        for start in range(0, len(sequences), max(1, batch_size)):
            batch_seqs = sequences[start : start + batch_size]
            inputs_tensor, pad_mask, batch_lengths = _pad_batch(batch_seqs)
            inputs_tensor = inputs_tensor.to(device_obj)
            pad_mask = pad_mask.to(device_obj)
            _, hidden = model(inputs_tensor, src_key_padding_mask=pad_mask)
            hidden_np = hidden.cpu().numpy()
            for i, L in enumerate(batch_lengths):
                Z.append(hidden_np[i, :L].astype(np.float32, copy=True))
    return Z, lengths


__all__ = [
    "MaskedTransformer",
    "MaskedTransformerConfig",
    "TIER_LOSS_WEIGHTS",
    "TrainResult",
    "apply_span_mask",
    "extract_contextual_embeddings",
    "train_masked_transformer",
]
