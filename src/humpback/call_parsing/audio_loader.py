"""Shared audio loader module for call parsing workers.

Centralises coordinate conversion (relative job offsets → absolute
timestamps), caching, and the two consumer protocol families:

* **Event classification** — ``build_event_audio_loader`` returns
  ``Callable[[sample], tuple[np.ndarray, float]]``.
* **Region / segmentation** — ``build_region_audio_loader`` and
  ``build_training_audio_loader`` return ``Callable[[sample], np.ndarray]``.

All ``job_start_ts + relative_offset`` conversions happen inside
``CachedAudioSource.get_audio``; callers never perform this conversion.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from humpback.config import Settings
from humpback.models.audio import AudioFile
from humpback.processing.audio_io import decode_audio, resample
from humpback.storage import resolve_audio_path

logger = logging.getLogger(__name__)


class CachedAudioSource:
    """Wraps a file-based or hydrophone audio source with optional caching.

    Coordinate conversion (relative → absolute) is performed inside
    :meth:`get_audio` so that no caller ever does this calculation.
    """

    def __init__(self) -> None:
        self._kind: str = ""
        self._audio: np.ndarray | None = None
        self._target_sr: int = 0
        self._job_start_ts: float = 0.0
        self._job_end_ts: float = 0.0
        self._hydrophone_id: str = ""
        self._settings: Settings | None = None
        self._preload_start_rel: float | None = None

    @classmethod
    def from_file(
        cls,
        audio_file: AudioFile,
        target_sr: int,
        storage_root: Path,
    ) -> CachedAudioSource:
        """Decode *audio_file* once and cache the resampled waveform."""
        src = cls()
        src._kind = "file"
        src._target_sr = target_sr
        path = resolve_audio_path(audio_file, storage_root)
        raw, sr = decode_audio(path)
        src._audio = np.asarray(resample(raw, sr, target_sr), dtype=np.float32)
        return src

    @classmethod
    def from_hydrophone(
        cls,
        hydrophone_id: str,
        job_start_ts: float,
        job_end_ts: float,
        target_sr: int,
        settings: Settings,
        preload_span: tuple[float, float] | None = None,
    ) -> CachedAudioSource:
        """Create a hydrophone source, optionally pre-loading a time span.

        Parameters
        ----------
        preload_span:
            ``(rel_start, rel_end)`` in seconds relative to *job_start_ts*.
            When provided, ``resolve_timeline_audio`` is called once at
            construction and all subsequent ``get_audio`` calls return the
            cached buffer.
        """
        from humpback.processing.timeline_audio import resolve_timeline_audio

        src = cls()
        src._kind = "hydrophone"
        src._target_sr = target_sr
        src._job_start_ts = job_start_ts
        src._job_end_ts = job_end_ts
        src._hydrophone_id = hydrophone_id
        src._settings = settings

        if preload_span is not None:
            rel_start, rel_end = preload_span
            duration = max(0.0, rel_end - rel_start)
            src._audio = resolve_timeline_audio(
                hydrophone_id=hydrophone_id,
                local_cache_path=str(settings.s3_cache_path or ""),
                job_start_timestamp=job_start_ts,
                job_end_timestamp=job_end_ts,
                start_sec=job_start_ts + rel_start,
                duration_sec=duration,
                target_sr=target_sr,
                noaa_cache_path=str(settings.noaa_cache_path)
                if settings.noaa_cache_path
                else None,
            )
            src._preload_start_rel = rel_start

        return src

    def get_audio(
        self, rel_start_sec: float, duration_sec: float
    ) -> tuple[np.ndarray, float]:
        """Return ``(audio_buffer, buffer_start_relative)``.

        *rel_start_sec* and *duration_sec* are relative to the job start.

        * **File source:** arguments are ignored; returns the full cached
          waveform with offset ``0.0``.
        * **Hydrophone pre-loaded:** returns the cached buffer and its
          relative start offset.
        * **Hydrophone per-request:** calls ``resolve_timeline_audio``
          with ``start_sec = job_start_ts + rel_start_sec``.
        """
        if self._kind == "file":
            assert self._audio is not None
            return self._audio, 0.0

        if self._audio is not None and self._preload_start_rel is not None:
            return self._audio, self._preload_start_rel

        from humpback.processing.timeline_audio import resolve_timeline_audio

        assert self._settings is not None
        audio = resolve_timeline_audio(
            hydrophone_id=self._hydrophone_id,
            local_cache_path=str(self._settings.s3_cache_path or ""),
            job_start_timestamp=self._job_start_ts,
            job_end_timestamp=self._job_end_ts,
            start_sec=self._job_start_ts + rel_start_sec,
            duration_sec=duration_sec,
            target_sr=self._target_sr,
            noaa_cache_path=str(self._settings.noaa_cache_path)
            if self._settings.noaa_cache_path
            else None,
        )
        return audio, rel_start_sec

    def get_audio_slice(self, rel_start_sec: float, duration_sec: float) -> np.ndarray:
        """Return the audio samples for the requested range.

        Like :meth:`get_audio` but slices the buffer to the exact
        requested interval, returning a plain ``np.ndarray``.
        """
        audio, buf_start = self.get_audio(rel_start_sec, duration_sec)

        if self._kind == "file" or (
            self._audio is not None and self._preload_start_rel is not None
        ):
            # Need to slice from the cached buffer.
            sr = self._target_sr
            start_idx = int(round((rel_start_sec - buf_start) * sr))
            end_idx = start_idx + int(round(duration_sec * sr))
            start_idx = max(0, min(start_idx, audio.shape[0]))
            end_idx = max(start_idx, min(end_idx, audio.shape[0]))
            return audio[start_idx:end_idx].copy()

        # Per-request mode: audio already covers exactly the requested range.
        return audio


def _compute_preload_span(
    events: Sequence[Any],
    job_duration: float,
) -> tuple[float, float]:
    """Compute a bounding span with context padding for pre-loading.

    Uses ``start_sec`` / ``end_sec`` attributes on each event.  Context
    padding strategy: ``max(10.0, event_duration)`` per event, symmetric,
    clamped to ``[0.0, job_duration]``.
    """
    padded_starts: list[float] = []
    padded_ends: list[float] = []

    for ev in events:
        start = float(ev.start_sec)
        end = float(ev.end_sec)
        duration = max(0.0, end - start)
        context_sec = max(10.0, duration)
        pad = (context_sec - duration) / 2.0
        padded_starts.append(max(0.0, start - pad))
        padded_ends.append(min(job_duration, end + pad))

    return min(padded_starts), max(padded_ends)


def build_event_audio_loader(
    *,
    target_sr: int,
    settings: Settings,
    audio_file: AudioFile | None = None,
    storage_root: Path | None = None,
    hydrophone_id: str | None = None,
    job_start_ts: float | None = None,
    job_end_ts: float | None = None,
    preload_events: Sequence[Any] | None = None,
) -> Callable[[Any], tuple[np.ndarray, float]]:
    """Build a loader for event classification consumers.

    Returns ``Callable[[event], (audio_buffer, buffer_start_relative)]``.
    """
    if audio_file is not None:
        assert storage_root is not None
        source = CachedAudioSource.from_file(audio_file, target_sr, storage_root)
    elif hydrophone_id is not None:
        assert job_start_ts is not None
        assert job_end_ts is not None
        preload_span: tuple[float, float] | None = None
        if preload_events:
            job_duration = job_end_ts - job_start_ts
            preload_span = _compute_preload_span(preload_events, job_duration)
        source = CachedAudioSource.from_hydrophone(
            hydrophone_id=hydrophone_id,
            job_start_ts=job_start_ts,
            job_end_ts=job_end_ts,
            target_sr=target_sr,
            settings=settings,
            preload_span=preload_span,
        )
    else:
        raise ValueError("Provide either audio_file or hydrophone_id")

    is_preloaded = source._audio is not None
    _job_dur = (job_end_ts or 0.0) - (job_start_ts or 0.0)

    def _load(event: Any) -> tuple[np.ndarray, float]:
        if is_preloaded or source._kind == "file":
            return source.get_audio(float(event.start_sec), 0.0)
        # Per-request: compute context padding around the event.
        start = float(event.start_sec)
        end = float(event.end_sec)
        duration = max(0.0, end - start)
        context_sec = max(10.0, duration)
        pad = (context_sec - duration) / 2.0
        ctx_start = max(0.0, start - pad)
        ctx_end = min(_job_dur, end + pad) if _job_dur > 0 else end + pad
        ctx_duration = max(0.0, ctx_end - ctx_start)
        return source.get_audio(ctx_start, ctx_duration)

    return _load


def build_region_audio_loader(
    *,
    target_sr: int,
    settings: Settings,
    audio_file: AudioFile | None = None,
    storage_root: Path | None = None,
    hydrophone_id: str | None = None,
    job_start_ts: float | None = None,
    job_end_ts: float | None = None,
    preload_span: tuple[float, float] | None = None,
) -> Callable[[Any], np.ndarray]:
    """Build a loader for segmentation inference consumers.

    Returns ``Callable[[region], np.ndarray]`` reading
    ``padded_start_sec`` / ``padded_end_sec`` from the region.
    """
    if audio_file is not None:
        assert storage_root is not None
        source = CachedAudioSource.from_file(audio_file, target_sr, storage_root)
    elif hydrophone_id is not None:
        assert job_start_ts is not None
        assert job_end_ts is not None
        source = CachedAudioSource.from_hydrophone(
            hydrophone_id=hydrophone_id,
            job_start_ts=job_start_ts,
            job_end_ts=job_end_ts,
            target_sr=target_sr,
            settings=settings,
            preload_span=preload_span,
        )
    else:
        raise ValueError("Provide either audio_file or hydrophone_id")

    def _load(region: Any) -> np.ndarray:
        start = float(region.padded_start_sec)
        end = float(region.padded_end_sec)
        return source.get_audio_slice(start, end - start)

    return _load


def build_training_audio_loader(
    *,
    target_sr: int,
    settings: Settings,
    samples: Sequence[Any] | None = None,
) -> Callable[[Any], np.ndarray]:
    """Build a loader for segmentation training consumers.

    Handles multi-source samples where each sample carries its own
    ``hydrophone_id``, ``start_timestamp``, ``end_timestamp``.  Reads
    ``crop_start_sec`` / ``crop_end_sec`` from each sample.

    If *samples* is provided, pre-loads audio per unique hydrophone
    context group.

    Returns ``Callable[[sample], np.ndarray]``.
    """
    sources: dict[tuple[str, float, float], CachedAudioSource] = {}

    if samples:
        groups: dict[tuple[str, float, float], list[Any]] = defaultdict(list)
        for s in samples:
            key = (s.hydrophone_id, float(s.start_timestamp), float(s.end_timestamp))
            groups[key].append(s)
        for key, group_samples in groups.items():
            min_start = min(float(s.crop_start_sec) for s in group_samples)
            max_end = max(float(s.crop_end_sec) for s in group_samples)
            sources[key] = CachedAudioSource.from_hydrophone(
                hydrophone_id=key[0],
                job_start_ts=key[1],
                job_end_ts=key[2],
                target_sr=target_sr,
                settings=settings,
                preload_span=(min_start, max_end),
            )

    def _load(sample: Any) -> np.ndarray:
        key = (
            sample.hydrophone_id,
            float(sample.start_timestamp),
            float(sample.end_timestamp),
        )
        if key not in sources:
            sources[key] = CachedAudioSource.from_hydrophone(
                hydrophone_id=key[0],
                job_start_ts=key[1],
                job_end_ts=key[2],
                target_sr=target_sr,
                settings=settings,
            )
        crop_start = float(sample.crop_start_sec)
        crop_end = float(sample.crop_end_sec)
        return sources[key].get_audio_slice(crop_start, crop_end - crop_start)

    return _load


def build_multi_source_event_audio_loader(
    *,
    target_sr: int,
    settings: Settings,
    samples: Sequence[Any] | None = None,
) -> Callable[[Any], tuple[np.ndarray, float]]:
    """Build a loader for event classification training with multi-source samples.

    Each sample carries its own ``hydrophone_id``, ``start_timestamp``,
    ``end_timestamp``.  Reads ``start_sec`` / ``end_sec`` from each sample,
    computes context padding, and returns ``(audio, buffer_start_relative)``.

    If *samples* is provided, pre-loads audio per unique hydrophone
    context group using ``_compute_preload_span``.

    Returns ``Callable[[sample], tuple[np.ndarray, float]]``.
    """
    sources: dict[tuple[str, float, float], CachedAudioSource] = {}

    if samples:
        groups: dict[tuple[str, float, float], list[Any]] = defaultdict(list)
        for s in samples:
            key = (s.hydrophone_id, float(s.start_timestamp), float(s.end_timestamp))
            groups[key].append(s)
        for key, group_samples in groups.items():
            job_duration = key[2] - key[1]
            preload_span = _compute_preload_span(group_samples, job_duration)
            sources[key] = CachedAudioSource.from_hydrophone(
                hydrophone_id=key[0],
                job_start_ts=key[1],
                job_end_ts=key[2],
                target_sr=target_sr,
                settings=settings,
                preload_span=preload_span,
            )

    def _load(sample: Any) -> tuple[np.ndarray, float]:
        key = (
            sample.hydrophone_id,
            float(sample.start_timestamp),
            float(sample.end_timestamp),
        )
        source = sources.get(key)
        if source is None:
            source = CachedAudioSource.from_hydrophone(
                hydrophone_id=key[0],
                job_start_ts=key[1],
                job_end_ts=key[2],
                target_sr=target_sr,
                settings=settings,
            )
            sources[key] = source

        if source._audio is not None:
            return source.get_audio(float(sample.start_sec), 0.0)

        # Per-request: compute context padding.
        start = float(sample.start_sec)
        end = float(sample.end_sec)
        duration = max(0.0, end - start)
        context_sec = max(10.0, duration)
        pad = (context_sec - duration) / 2.0
        job_dur = key[2] - key[1]
        ctx_start = max(0.0, start - pad)
        ctx_end = min(job_dur, end + pad) if job_dur > 0 else end + pad
        ctx_duration = max(0.0, ctx_end - ctx_start)
        return source.get_audio(ctx_start, ctx_duration)

    return _load
