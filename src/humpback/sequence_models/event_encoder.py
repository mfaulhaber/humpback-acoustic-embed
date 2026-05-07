"""Pure utilities for CRNN event-level vector construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

POOL_ORDER = [
    "mean_pool",
    "top_k_pool",
    "start_pool",
    "middle_pool",
    "end_pool",
]

DESCRIPTOR_ORDER = [
    "duration",
    "log_energy",
    "peak_frequency",
    "spectral_centroid",
    "bandwidth",
    "spectral_entropy",
    "frequency_slope",
    "gap_to_previous",
]


@dataclass(frozen=True)
class EventInterval:
    event_id: str
    region_id: str
    start_timestamp: float
    end_timestamp: float
    segmentation_confidence: float = 0.0
    source_sequence_key: str = "source"


@dataclass(frozen=True)
class ChunkEmbedding:
    region_id: str
    start_timestamp: float
    end_timestamp: float
    call_probability: float
    embedding: np.ndarray


@dataclass(frozen=True)
class SelectedChunk:
    chunk: ChunkEmbedding
    overlap_seconds: float


@dataclass(frozen=True)
class EventEmbeddingResult:
    event: EventInterval
    pool_vector: np.ndarray
    pools: dict[str, np.ndarray]
    chunk_count: int
    coverage_fraction: float


def interval_overlap(
    start_a: float, end_a: float, start_b: float, end_b: float
) -> float:
    """Return seconds of half-open interval overlap."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def select_event_chunks(
    event: EventInterval,
    chunks: list[ChunkEmbedding],
    *,
    min_overlap_fraction: float = 0.25,
    min_chunks_per_event: int = 1,
) -> tuple[list[SelectedChunk], float, str | None]:
    """Select positive-overlap chunks and report coverage / skip reason."""
    event_duration = event.end_timestamp - event.start_timestamp
    if event_duration <= 0:
        return [], 0.0, "non_positive_duration"

    selected = [
        SelectedChunk(chunk=chunk, overlap_seconds=overlap)
        for chunk in chunks
        if chunk.region_id == event.region_id
        for overlap in (
            interval_overlap(
                event.start_timestamp,
                event.end_timestamp,
                chunk.start_timestamp,
                chunk.end_timestamp,
            ),
        )
        if overlap > 0
    ]
    if not selected:
        return [], 0.0, "no_overlapping_chunks"

    coverage = _union_coverage(
        event.start_timestamp,
        event.end_timestamp,
        [(s.chunk.start_timestamp, s.chunk.end_timestamp) for s in selected],
    )
    coverage_fraction = min(1.0, coverage / event_duration)
    if (
        coverage_fraction < min_overlap_fraction
        and len(selected) < min_chunks_per_event
    ):
        return selected, coverage_fraction, "insufficient_chunk_coverage"
    return selected, coverage_fraction, None


def build_event_embedding(
    event: EventInterval,
    chunks: list[ChunkEmbedding],
    *,
    enabled_pools: list[str] | None = None,
    top_k_fraction: float = 0.25,
    min_overlap_fraction: float = 0.25,
    min_chunks_per_event: int = 1,
) -> EventEmbeddingResult:
    """Build a concatenated pool vector for one event."""
    pools_to_emit = enabled_pools or list(POOL_ORDER)
    selected, coverage, skip_reason = select_event_chunks(
        event,
        chunks,
        min_overlap_fraction=min_overlap_fraction,
        min_chunks_per_event=min_chunks_per_event,
    )
    if skip_reason is not None:
        raise ValueError(skip_reason)
    if not selected:
        raise ValueError("no_overlapping_chunks")

    embeddings = np.stack(
        [np.asarray(s.chunk.embedding, dtype=np.float32) for s in selected]
    )
    overlap_weights = np.asarray(
        [s.overlap_seconds for s in selected], dtype=np.float32
    )
    if np.any(overlap_weights < 0):
        raise ValueError("overlap weights must be non-negative")

    pools: dict[str, np.ndarray] = {}
    for pool_name in pools_to_emit:
        if pool_name == "mean_pool":
            pools[pool_name] = _weighted_mean(embeddings, overlap_weights)
        elif pool_name == "top_k_pool":
            pools[pool_name] = _top_k_pool(selected, embeddings, top_k_fraction)
        elif pool_name in ("start_pool", "middle_pool", "end_pool"):
            pools[pool_name] = _segment_pool(event, selected, embeddings, pool_name)
        else:
            raise ValueError(f"unsupported event encoder pool: {pool_name!r}")

    pool_vector = np.concatenate([pools[name] for name in pools_to_emit]).astype(
        np.float32
    )
    return EventEmbeddingResult(
        event=event,
        pool_vector=pool_vector,
        pools=pools,
        chunk_count=len(selected),
        coverage_fraction=float(coverage),
    )


def compute_gap_to_previous(events: list[EventInterval]) -> dict[str, float]:
    """Compute per-event gap to previous event within each source sequence."""
    gaps: dict[str, float] = {}
    previous_end_by_source: dict[str, float] = {}
    for event in sorted(
        events,
        key=lambda e: (
            e.source_sequence_key,
            e.start_timestamp,
            e.end_timestamp,
            e.event_id,
        ),
    ):
        previous_end = previous_end_by_source.get(event.source_sequence_key)
        gaps[event.event_id] = (
            0.0
            if previous_end is None
            else max(0.0, event.start_timestamp - previous_end)
        )
        previous_end_by_source[event.source_sequence_key] = event.end_timestamp
    return gaps


def compute_acoustic_descriptors(
    audio: np.ndarray,
    *,
    sample_rate: int,
    gap_to_previous: float = 0.0,
    n_fft: int = 1024,
    hop_length: int = 512,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute the v1 acoustic descriptor set for one event crop."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        x = np.ravel(x)
    duration = float(x.shape[0] / sample_rate) if sample_rate > 0 else 0.0
    if x.size == 0:
        return {
            "duration": 0.0,
            "log_energy": float(np.log(eps)),
            "peak_frequency": 0.0,
            "spectral_centroid": 0.0,
            "bandwidth": 0.0,
            "spectral_entropy": 0.0,
            "frequency_slope": 0.0,
            "gap_to_previous": float(gap_to_previous),
        }

    log_energy = float(np.log(float(np.mean(np.square(x))) + eps))
    frames = _frame_audio(x, n_fft=n_fft, hop_length=hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    spectra = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mean_spectrum = spectra.mean(axis=0)
    total_mag = float(mean_spectrum.sum())
    if total_mag <= eps:
        peak_frequency = 0.0
        centroid = 0.0
        bandwidth = 0.0
        entropy = 0.0
    else:
        probs = mean_spectrum / total_mag
        peak_frequency = float(freqs[int(np.argmax(mean_spectrum))])
        centroid = float(np.sum(freqs * probs))
        bandwidth = float(np.sqrt(np.sum(np.square(freqs - centroid) * probs)))
        entropy_raw = -float(np.sum(probs * np.log(probs + eps)))
        entropy = entropy_raw / float(np.log(len(probs))) if len(probs) > 1 else 0.0

    frame_peaks = freqs[np.argmax(spectra, axis=1)]
    if frame_peaks.shape[0] < 2:
        slope = 0.0
    else:
        times = np.arange(frame_peaks.shape[0], dtype=np.float64) * (
            hop_length / sample_rate
        )
        slope = float(np.polyfit(times, frame_peaks.astype(np.float64), 1)[0])

    return {
        "duration": duration,
        "log_energy": log_energy,
        "peak_frequency": peak_frequency,
        "spectral_centroid": centroid,
        "bandwidth": bandwidth,
        "spectral_entropy": entropy,
        "frequency_slope": slope,
        "gap_to_previous": float(gap_to_previous),
    }


def descriptor_vector(descriptors: dict[str, Any]) -> np.ndarray:
    """Return descriptor values in the stable v1 order."""
    return np.asarray(
        [float(descriptors.get(name, 0.0)) for name in DESCRIPTOR_ORDER],
        dtype=np.float32,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights length mismatch")
    total = float(weights.sum())
    if total <= 0:
        return values.mean(axis=0).astype(np.float32)
    return ((values * weights[:, None]).sum(axis=0) / total).astype(np.float32)


def _top_k_pool(
    selected: list[SelectedChunk], embeddings: np.ndarray, top_k_fraction: float
) -> np.ndarray:
    count = max(1, ceil(len(selected) * top_k_fraction))
    order = np.argsort(
        np.asarray([s.chunk.call_probability for s in selected], dtype=np.float32)
    )[::-1]
    top_idx = order[:count]
    weights = np.asarray(
        [selected[i].overlap_seconds for i in top_idx], dtype=np.float32
    )
    return _weighted_mean(embeddings[top_idx], weights)


def _segment_pool(
    event: EventInterval,
    selected: list[SelectedChunk],
    embeddings: np.ndarray,
    pool_name: str,
) -> np.ndarray:
    duration = event.end_timestamp - event.start_timestamp
    third = duration / 3.0
    if pool_name == "start_pool":
        seg_start, seg_end = event.start_timestamp, event.start_timestamp + third
    elif pool_name == "middle_pool":
        seg_start = event.start_timestamp + third
        seg_end = event.start_timestamp + 2.0 * third
    elif pool_name == "end_pool":
        seg_start = event.start_timestamp + 2.0 * third
        seg_end = event.end_timestamp
    else:
        raise ValueError(f"unsupported segment pool: {pool_name!r}")

    weights = np.asarray(
        [
            interval_overlap(
                seg_start,
                seg_end,
                s.chunk.start_timestamp,
                s.chunk.end_timestamp,
            )
            for s in selected
        ],
        dtype=np.float32,
    )
    if float(weights.sum()) > 0:
        return _weighted_mean(embeddings, weights)

    segment_center = (seg_start + seg_end) / 2.0
    centers = np.asarray(
        [(s.chunk.start_timestamp + s.chunk.end_timestamp) / 2.0 for s in selected],
        dtype=np.float64,
    )
    return embeddings[int(np.argmin(np.abs(centers - segment_center)))].astype(
        np.float32
    )


def _union_coverage(
    event_start: float, event_end: float, intervals: list[tuple[float, float]]
) -> float:
    clipped = [
        (max(event_start, start), min(event_end, end))
        for start, end in sorted(intervals)
        if min(event_end, end) > max(event_start, start)
    ]
    if not clipped:
        return 0.0
    total = 0.0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def _frame_audio(audio: np.ndarray, *, n_fft: int, hop_length: int) -> np.ndarray:
    if audio.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[: audio.size] = audio
        return padded[None, :]
    starts = range(0, audio.size - n_fft + 1, hop_length)
    frames = [audio[start : start + n_fft] for start in starts]
    if not frames:
        frames = [audio[:n_fft]]
    return np.stack(frames).astype(np.float32)
