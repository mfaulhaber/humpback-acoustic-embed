"""Build HMM training sub-sequences from CRNN region chunk embeddings.

Three training modes select which chunks become training input:

- ``full_region`` — every region is one training sub-sequence; if the
  total exceeds ``target_train_chunks`` the regions are uniformly
  subsampled.
- ``event_balanced`` — stratified sub-sequence extraction. Sub-sequences
  centred on event-core chunks (deterministic walk with stride),
  on near-event chunks, and on background chunks are mixed according to
  ``event_balanced_proportions``.
- ``event_only`` — same as ``event_balanced`` but background is dropped.

Decode is always over the whole region; this module only produces the
training input plus a ``was_used_for_training`` mask per region (aligned
to the source chunk order) so the decoder pipeline can flag which
chunks influenced the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from humpback.sequence_models.event_overlap_join import (
    BACKGROUND,
    EVENT_CORE,
    NEAR_EVENT,
)

DEFAULT_PROPORTIONS: dict[str, float] = {
    EVENT_CORE: 0.40,
    NEAR_EVENT: 0.35,
    BACKGROUND: 0.25,
}


@dataclass(frozen=True)
class RegionSequence:
    """One region's chunk-aligned tensors fed to the trainer-builder."""

    region_id: str
    chunks: np.ndarray  # (T_chunks, D) float32
    tiers: np.ndarray  # (T_chunks,) object/str


@dataclass(frozen=True)
class TierConfig:
    event_balanced_proportions: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PROPORTIONS)
    )


@dataclass(frozen=True)
class SamplingConfig:
    subsequence_length_chunks: int = 32
    subsequence_stride_chunks: int = 16
    target_train_chunks: int = 200_000
    min_sequence_length_frames: int = 1
    random_seed: int = 0


@dataclass
class TrainingSet:
    """Builder output: sub-sequences + length vector + per-region masks."""

    sub_sequences: list[np.ndarray]
    lengths: np.ndarray
    was_used_for_training_per_region: dict[str, np.ndarray]


def _tier_indices(tiers: np.ndarray, target_tier: str) -> np.ndarray:
    """Return chunk indices in this region whose tier == ``target_tier``."""
    return np.flatnonzero(tiers == target_tier)


def _subseq_window(centre: int, length: int) -> tuple[int, int]:
    """Return ``(start, end)`` for a sub-sequence centred on ``centre``."""
    half = length // 2
    return centre - half, centre - half + length


def _emit_centred_subsequences(
    region: RegionSequence,
    centres: np.ndarray,
    *,
    length: int,
    stride: int,
    out_subseqs: list[np.ndarray],
    out_mask: np.ndarray,
    cap_chunks: int,
    used_so_far: int,
) -> int:
    """Emit sub-sequences centred on ``centres`` with stride deduplication.

    Mutates ``out_subseqs`` and ``out_mask`` in place. Returns the
    updated ``used_so_far`` chunk count (incl. the chunks just emitted).
    """
    last_centre = -(10**9)
    for centre in centres.tolist():
        if used_so_far >= cap_chunks:
            break
        if centre - last_centre < stride:
            continue
        start, end = _subseq_window(centre, length)
        if start < 0 or end > region.chunks.shape[0]:
            continue
        out_subseqs.append(region.chunks[start:end])
        out_mask[start:end] = True
        used_so_far += length
        last_centre = centre
    return used_so_far


def _build_full_region(
    region_sequences: list[RegionSequence],
    sampling: SamplingConfig,
) -> TrainingSet:
    eligible = [
        r
        for r in region_sequences
        if r.chunks.shape[0] >= sampling.min_sequence_length_frames
    ]
    masks = {
        r.region_id: np.zeros(r.chunks.shape[0], dtype=bool) for r in region_sequences
    }
    if not eligible:
        return TrainingSet([], np.zeros(0, dtype=np.int64), masks)

    rng = np.random.default_rng(sampling.random_seed)
    order = list(range(len(eligible)))
    rng.shuffle(order)

    sub_sequences: list[np.ndarray] = []
    used = 0
    for idx in order:
        r = eligible[idx]
        if used >= sampling.target_train_chunks:
            break
        sub_sequences.append(r.chunks)
        masks[r.region_id][:] = True
        used += r.chunks.shape[0]

    lengths = np.asarray([s.shape[0] for s in sub_sequences], dtype=np.int64)
    return TrainingSet(sub_sequences, lengths, masks)


def _build_stratified(
    region_sequences: list[RegionSequence],
    tier_config: TierConfig,
    sampling: SamplingConfig,
    *,
    include_background: bool,
) -> TrainingSet:
    """Mode B/C builder: tier-balanced sub-sequences."""
    proportions = dict(tier_config.event_balanced_proportions)
    if not include_background:
        proportions.pop(BACKGROUND, None)
        total = sum(proportions.values())
        if total <= 0:
            raise ValueError("event_only requires positive non-background proportions")
        proportions = {k: v / total for k, v in proportions.items()}

    target_per_tier = {
        tier: int(round(sampling.target_train_chunks * frac))
        for tier, frac in proportions.items()
    }

    masks = {
        r.region_id: np.zeros(r.chunks.shape[0], dtype=bool) for r in region_sequences
    }
    sub_sequences: list[np.ndarray] = []

    rng = np.random.default_rng(sampling.random_seed)
    region_order = list(range(len(region_sequences)))
    rng.shuffle(region_order)

    used_per_tier: dict[str, int] = {tier: 0 for tier in proportions}

    for tier in proportions:
        cap = target_per_tier[tier]
        for idx in region_order:
            if used_per_tier[tier] >= cap:
                break
            region = region_sequences[idx]
            if region.chunks.shape[0] < sampling.subsequence_length_chunks:
                continue
            if tier == EVENT_CORE:
                centres = _tier_indices(region.tiers, EVENT_CORE)
            else:
                idxs = _tier_indices(region.tiers, tier)
                if idxs.size == 0:
                    continue
                shuffled = idxs.copy()
                rng.shuffle(shuffled)
                centres = shuffled
            used_per_tier[tier] = _emit_centred_subsequences(
                region,
                centres,
                length=sampling.subsequence_length_chunks,
                stride=sampling.subsequence_stride_chunks,
                out_subseqs=sub_sequences,
                out_mask=masks[region.region_id],
                cap_chunks=cap,
                used_so_far=used_per_tier[tier],
            )

    lengths = np.asarray([s.shape[0] for s in sub_sequences], dtype=np.int64)
    return TrainingSet(sub_sequences, lengths, masks)


def build_training_set(
    region_sequences: list[RegionSequence],
    mode: str,
    tier_config: TierConfig,
    sampling: SamplingConfig,
) -> TrainingSet:
    """Build HMM training sub-sequences for one of three modes.

    See module docstring for mode semantics. ``region_sequences`` is the
    full list of regions; the returned masks always cover every region
    (False everywhere if a region contributes nothing).
    """
    if mode == "full_region":
        return _build_full_region(region_sequences, sampling)
    if mode == "event_balanced":
        return _build_stratified(
            region_sequences, tier_config, sampling, include_background=True
        )
    if mode == "event_only":
        return _build_stratified(
            region_sequences, tier_config, sampling, include_background=False
        )
    raise ValueError(
        f"unknown training mode {mode!r}; expected one of "
        "{'full_region', 'event_balanced', 'event_only'}"
    )
