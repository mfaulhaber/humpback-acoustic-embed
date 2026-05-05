"""Pure sequence construction helpers for masked-transformer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

SequenceConstructionMode = Literal["region", "event_centered", "mixed"]


@dataclass(frozen=True)
class EffectiveEventInterval:
    """Effective event interval in the same absolute time domain as CRNN chunks."""

    region_id: str
    start_timestamp: float
    end_timestamp: float
    event_id: str | None = None
    human_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingSequenceCandidate:
    """One training sequence plus alignment metadata for tests/diagnostics."""

    source_kind: Literal["region", "event_centered"]
    region_id: str
    sequence: np.ndarray
    tiers: list[str]
    start_index: int
    end_index: int
    event_id: str | None = None
    event_start_timestamp: float | None = None
    event_end_timestamp: float | None = None
    event_start_index: int | None = None
    event_end_index: int | None = None
    human_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequenceConstructionResult:
    """Constructed masked-transformer training sequences."""

    candidates: list[TrainingSequenceCandidate]

    @property
    def sequences(self) -> list[np.ndarray]:
        return [candidate.sequence for candidate in self.candidates]

    @property
    def tier_lists(self) -> list[list[str]]:
        return [candidate.tiers for candidate in self.candidates]

    @property
    def source_kinds(self) -> list[str]:
        return [candidate.source_kind for candidate in self.candidates]


def build_masked_transformer_training_sequences(
    *,
    region_ids: list[str],
    sequences: list[np.ndarray],
    tier_lists: Optional[list[list[str]]],
    starts: list[list[float]],
    ends: list[list[float]],
    effective_events: list[EffectiveEventInterval],
    mode: SequenceConstructionMode = "region",
    event_centered_fraction: float = 0.0,
    pre_event_context_sec: Optional[float] = None,
    post_event_context_sec: Optional[float] = None,
    seed: int = 42,
) -> SequenceConstructionResult:
    """Build training sequences without touching extraction-time alignment.

    Region mode returns the original full-region arrays unchanged. Event
    windows are derived from chunks overlapping effective event intervals plus
    configured pre/post context.
    """
    _validate_inputs(region_ids, sequences, tier_lists, starts, ends)
    if mode not in {"region", "event_centered", "mixed"}:
        raise ValueError("mode must be one of region/event_centered/mixed")

    region_candidates = _region_candidates(region_ids, sequences, tier_lists)
    if mode == "region":
        return SequenceConstructionResult(region_candidates)

    pre_context = 2.0 if pre_event_context_sec is None else float(pre_event_context_sec)
    post_context = (
        2.0 if post_event_context_sec is None else float(post_event_context_sec)
    )
    if pre_context < 0.0 or post_context < 0.0:
        raise ValueError("event context seconds must be non-negative")

    event_candidates = _event_centered_candidates(
        region_ids=region_ids,
        sequences=sequences,
        tier_lists=tier_lists,
        starts=starts,
        ends=ends,
        effective_events=effective_events,
        pre_context=pre_context,
        post_context=post_context,
    )

    if mode == "event_centered":
        return SequenceConstructionResult(event_candidates)

    if not 0.0 < float(event_centered_fraction) < 1.0:
        raise ValueError("mixed mode requires 0.0 < event_centered_fraction < 1.0")
    return SequenceConstructionResult(
        _mixed_candidates(
            region_candidates=region_candidates,
            event_candidates=event_candidates,
            event_centered_fraction=float(event_centered_fraction),
            seed=seed,
        )
    )


def _validate_inputs(
    region_ids: list[str],
    sequences: list[np.ndarray],
    tier_lists: Optional[list[list[str]]],
    starts: list[list[float]],
    ends: list[list[float]],
) -> None:
    n = len(sequences)
    if not (len(region_ids) == len(starts) == len(ends) == n):
        raise ValueError("region_ids, sequences, starts, and ends must align")
    if tier_lists is not None and len(tier_lists) != n:
        raise ValueError("tier_lists must align with sequences")
    for i, seq in enumerate(sequences):
        length = int(seq.shape[0])
        if len(starts[i]) != length or len(ends[i]) != length:
            raise ValueError("start/end metadata must align with sequence length")
        if tier_lists is not None and len(tier_lists[i]) != length:
            raise ValueError("tier metadata must align with sequence length")


def _region_candidates(
    region_ids: list[str],
    sequences: list[np.ndarray],
    tier_lists: Optional[list[list[str]]],
) -> list[TrainingSequenceCandidate]:
    out: list[TrainingSequenceCandidate] = []
    for i, (region_id, sequence) in enumerate(zip(region_ids, sequences)):
        tiers = (
            list(tier_lists[i]) if tier_lists is not None else [""] * sequence.shape[0]
        )
        out.append(
            TrainingSequenceCandidate(
                source_kind="region",
                region_id=region_id,
                sequence=sequence,
                tiers=tiers,
                start_index=0,
                end_index=int(sequence.shape[0]),
                event_id=None,
                event_start_timestamp=None,
                event_end_timestamp=None,
                event_start_index=None,
                event_end_index=None,
                human_types=(),
            )
        )
    return out


def _event_centered_candidates(
    *,
    region_ids: list[str],
    sequences: list[np.ndarray],
    tier_lists: Optional[list[list[str]]],
    starts: list[list[float]],
    ends: list[list[float]],
    effective_events: list[EffectiveEventInterval],
    pre_context: float,
    post_context: float,
) -> list[TrainingSequenceCandidate]:
    region_index = {region_id: i for i, region_id in enumerate(region_ids)}
    out: list[TrainingSequenceCandidate] = []
    for event in sorted(
        effective_events,
        key=lambda e: (e.region_id, e.start_timestamp, e.end_timestamp),
    ):
        idx = region_index.get(event.region_id)
        if idx is None:
            continue
        start_seq = starts[idx]
        end_seq = ends[idx]
        event_overlap = [
            i
            for i, (chunk_start, chunk_end) in enumerate(zip(start_seq, end_seq))
            if chunk_end > event.start_timestamp and chunk_start < event.end_timestamp
        ]
        if not event_overlap:
            continue
        window_start = event.start_timestamp - pre_context
        window_end = event.end_timestamp + post_context
        window_overlap = [
            i
            for i, (chunk_start, chunk_end) in enumerate(zip(start_seq, end_seq))
            if chunk_end > window_start and chunk_start < window_end
        ]
        if not window_overlap:
            continue
        start_idx = min(window_overlap)
        end_idx = max(window_overlap) + 1
        event_start_idx = min(event_overlap) - start_idx
        event_end_idx = max(event_overlap) + 1 - start_idx
        tiers = (
            list(tier_lists[idx][start_idx:end_idx])
            if tier_lists is not None
            else [""] * (end_idx - start_idx)
        )
        out.append(
            TrainingSequenceCandidate(
                source_kind="event_centered",
                region_id=event.region_id,
                sequence=sequences[idx][start_idx:end_idx].copy(),
                tiers=tiers,
                start_index=start_idx,
                end_index=end_idx,
                event_id=event.event_id,
                event_start_timestamp=event.start_timestamp,
                event_end_timestamp=event.end_timestamp,
                event_start_index=event_start_idx,
                event_end_index=event_end_idx,
                human_types=event.human_types,
            )
        )
    return out


def _mixed_candidates(
    *,
    region_candidates: list[TrainingSequenceCandidate],
    event_candidates: list[TrainingSequenceCandidate],
    event_centered_fraction: float,
    seed: int,
) -> list[TrainingSequenceCandidate]:
    if not region_candidates:
        return _sample(event_candidates, len(event_candidates), seed)
    if not event_candidates:
        return _sample(region_candidates, len(region_candidates), seed)

    total_available = len(region_candidates) + len(event_candidates)
    desired_event = int(round(total_available * event_centered_fraction))
    desired_event = min(len(event_candidates), max(1, desired_event))
    desired_region = min(
        len(region_candidates), max(1, total_available - desired_event)
    )

    # If one side is exhausted, backfill from the other side while keeping the
    # total bounded by available candidates.
    total_desired = min(total_available, desired_event + desired_region)
    if desired_event + desired_region < total_desired:
        desired_region = min(len(region_candidates), total_desired - desired_event)

    rng = np.random.default_rng(seed)
    region_selection = _sample_with_rng(region_candidates, desired_region, rng)
    event_selection = _sample_with_rng(event_candidates, desired_event, rng)
    combined = [*region_selection, *event_selection]
    order = rng.permutation(len(combined))
    return [combined[int(i)] for i in order]


def _sample(
    candidates: list[TrainingSequenceCandidate], count: int, seed: int
) -> list[TrainingSequenceCandidate]:
    return _sample_with_rng(candidates, count, np.random.default_rng(seed))


def _sample_with_rng(
    candidates: list[TrainingSequenceCandidate],
    count: int,
    rng: np.random.Generator,
) -> list[TrainingSequenceCandidate]:
    if count >= len(candidates):
        return list(candidates)
    indices = sorted(
        int(i) for i in rng.choice(len(candidates), size=count, replace=False)
    )
    return [candidates[i] for i in indices]


__all__ = [
    "EffectiveEventInterval",
    "SequenceConstructionMode",
    "SequenceConstructionResult",
    "TrainingSequenceCandidate",
    "build_masked_transformer_training_sequences",
]
