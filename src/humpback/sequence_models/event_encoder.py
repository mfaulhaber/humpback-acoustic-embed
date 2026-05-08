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
    "ridge_log_frequency_slope",
    "gap_to_previous",
]

DESCRIPTOR_UNITS = {
    "duration": "seconds",
    "log_energy": "log power",
    "peak_frequency": "Hz",
    "spectral_centroid": "Hz",
    "bandwidth": "Hz",
    "spectral_entropy": "normalized",
    "ridge_log_frequency_slope": "octaves/s",
    "gap_to_previous": "seconds",
}


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
    ridge_min_frequency_hz: float = 100.0,
    ridge_max_frequency_hz: float = 3000.0,
    ridge_candidate_count: int = 5,
    ridge_smoothness_penalty: float = 8.0,
    ridge_peak_prominence_ratio: float = 0.0,
) -> dict[str, float]:
    """Compute acoustic descriptors for one event crop."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        x = np.ravel(x)
    duration = float(x.shape[0] / sample_rate) if sample_rate > 0 else 0.0
    if x.size == 0:
        empty = {
            "duration": 0.0,
            "log_energy": float(np.log(eps)),
            "peak_frequency": 0.0,
            "spectral_centroid": 0.0,
            "bandwidth": 0.0,
            "spectral_entropy": 0.0,
            "gap_to_previous": float(gap_to_previous),
            "ridge_log_frequency_slope": 0.0,
        }
        return empty

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

    slope = compute_ridge_log_frequency_slope(
        spectra,
        freqs,
        sample_rate=sample_rate,
        hop_length=hop_length,
        eps=eps,
        min_frequency_hz=ridge_min_frequency_hz,
        max_frequency_hz=ridge_max_frequency_hz,
        candidate_count=ridge_candidate_count,
        smoothness_penalty=ridge_smoothness_penalty,
        peak_prominence_ratio=ridge_peak_prominence_ratio,
    )

    descriptors = {
        "duration": duration,
        "log_energy": log_energy,
        "peak_frequency": peak_frequency,
        "spectral_centroid": centroid,
        "bandwidth": bandwidth,
        "spectral_entropy": entropy,
        "gap_to_previous": float(gap_to_previous),
        "ridge_log_frequency_slope": slope,
    }
    return descriptors


def compute_ridge_log_frequency_slope(
    spectra: np.ndarray,
    freqs: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    eps: float = 1e-12,
    min_frequency_hz: float = 100.0,
    max_frequency_hz: float = 3000.0,
    candidate_count: int = 5,
    smoothness_penalty: float = 8.0,
    peak_prominence_ratio: float = 0.0,
) -> float:
    """Track a smooth spectral ridge and fit log2 frequency over time."""
    if sample_rate <= 0 or hop_length <= 0 or spectra.ndim != 2 or spectra.shape[0] < 2:
        return 0.0
    freqs = np.asarray(freqs, dtype=np.float64)
    spectra = np.asarray(spectra, dtype=np.float64)
    if freqs.ndim != 1 or freqs.shape[0] != spectra.shape[1]:
        return 0.0
    if min_frequency_hz <= 0 or max_frequency_hz <= min_frequency_hz:
        return 0.0

    band_max = min(float(max_frequency_hz), float(freqs[-1]))
    band_mask = (
        np.isfinite(freqs)
        & (freqs >= float(min_frequency_hz))
        & (freqs <= band_max)
        & (freqs > 0)
    )
    band_indices = np.flatnonzero(band_mask)
    if band_indices.size == 0:
        return 0.0

    frame_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    frame_times: list[float] = []
    for frame_idx, spectrum in enumerate(spectra[:, band_indices]):
        candidate_bins = _ridge_candidate_bins(
            spectrum,
            candidate_count=max(1, int(candidate_count)),
            peak_prominence_ratio=max(0.0, float(peak_prominence_ratio)),
            eps=eps,
        )
        if candidate_bins.size == 0:
            continue
        strengths = spectrum[candidate_bins].astype(np.float64)
        if not np.any(np.isfinite(strengths)) or float(np.max(strengths)) <= eps:
            continue
        candidate_freqs = freqs[band_indices[candidate_bins]]
        frame_candidates.append((np.log2(candidate_freqs), strengths))
        frame_times.append(float(frame_idx * hop_length / sample_rate))

    if len(frame_candidates) < 2:
        return 0.0

    path = _track_log_frequency_path(
        frame_candidates,
        smoothness_penalty=max(0.0, float(smoothness_penalty)),
        eps=eps,
    )
    if path.size < 2:
        return 0.0
    times = np.asarray(frame_times, dtype=np.float64)
    return _finite_float(_theil_sen_slope(times, path))


def _ridge_candidate_bins(
    spectrum: np.ndarray,
    *,
    candidate_count: int,
    peak_prominence_ratio: float,
    eps: float,
) -> np.ndarray:
    if spectrum.size == 0:
        return np.asarray([], dtype=np.int64)
    frame_peak = float(np.nanmax(spectrum))
    if not np.isfinite(frame_peak) or frame_peak <= eps:
        return np.asarray([], dtype=np.int64)

    if spectrum.size == 1:
        local_maxima = np.asarray([0], dtype=np.int64)
    else:
        middle = (
            np.flatnonzero(
                (spectrum[1:-1] >= spectrum[:-2]) & (spectrum[1:-1] >= spectrum[2:])
            )
            + 1
        )
        endpoints = []
        if spectrum[0] >= spectrum[1]:
            endpoints.append(0)
        if spectrum[-1] >= spectrum[-2]:
            endpoints.append(spectrum.size - 1)
        local_maxima = np.asarray([*endpoints, *middle.tolist()], dtype=np.int64)

    threshold = frame_peak * peak_prominence_ratio
    local_maxima = local_maxima[spectrum[local_maxima] >= threshold]
    if local_maxima.size == 0:
        local_maxima = np.flatnonzero(spectrum >= threshold)
    if local_maxima.size == 0:
        local_maxima = np.arange(spectrum.size, dtype=np.int64)

    order = np.argsort(spectrum[local_maxima])[::-1]
    return local_maxima[order[:candidate_count]]


def _track_log_frequency_path(
    frame_candidates: list[tuple[np.ndarray, np.ndarray]],
    *,
    smoothness_penalty: float,
    eps: float,
) -> np.ndarray:
    previous_logs, previous_strengths = frame_candidates[0]
    previous_cost = _ridge_emission_cost(previous_strengths, eps=eps)
    backpointers: list[np.ndarray] = []
    candidate_logs = [previous_logs]

    for current_logs, current_strengths in frame_candidates[1:]:
        emission = _ridge_emission_cost(current_strengths, eps=eps)
        transition = smoothness_penalty * np.square(
            previous_logs[:, None] - current_logs[None, :]
        )
        total = previous_cost[:, None] + transition + emission[None, :]
        backpointer = np.argmin(total, axis=0)
        previous_cost = total[backpointer, np.arange(current_logs.shape[0])]
        previous_logs = current_logs
        backpointers.append(backpointer.astype(np.int64))
        candidate_logs.append(current_logs)

    index = int(np.argmin(previous_cost))
    path = [float(candidate_logs[-1][index])]
    for frame_idx in range(len(backpointers) - 1, -1, -1):
        index = int(backpointers[frame_idx][index])
        path.append(float(candidate_logs[frame_idx][index]))
    path.reverse()
    return np.asarray(path, dtype=np.float64)


def _ridge_emission_cost(strengths: np.ndarray, *, eps: float) -> np.ndarray:
    strengths = np.asarray(strengths, dtype=np.float64)
    frame_peak = float(np.max(strengths))
    if frame_peak <= eps:
        return np.zeros_like(strengths, dtype=np.float64)
    normalized = np.clip(strengths / frame_peak, eps, None)
    return -np.log(normalized)


def _theil_sen_slope(times: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if times.shape[0] < 2:
        return 0.0
    if times.shape[0] == 2:
        dt = float(times[1] - times[0])
        return 0.0 if dt == 0 else float((values[1] - values[0]) / dt)

    slopes = []
    for i in range(times.shape[0] - 1):
        dt = times[i + 1 :] - times[i]
        valid = dt > 0
        if np.any(valid):
            slopes.extend(((values[i + 1 :][valid] - values[i]) / dt[valid]).tolist())
    if not slopes:
        return 0.0
    return float(np.median(np.asarray(slopes, dtype=np.float64)))


def _finite_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def descriptor_units() -> dict[str, str]:
    """Return display units for the active descriptor order."""
    return {name: DESCRIPTOR_UNITS[name] for name in DESCRIPTOR_ORDER}


def descriptor_vector(descriptors: dict[str, Any]) -> np.ndarray:
    """Return descriptor values in the stable active order."""
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
