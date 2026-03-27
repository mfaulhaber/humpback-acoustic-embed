"""Resolve audio from archive providers for arbitrary timeline-absolute positions."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HLSManifestEntry:
    """Reusable metadata for one HLS segment in a job timeline."""

    segment_path: str
    start_epoch: float
    duration_sec: float


@dataclass(frozen=True)
class TimelineAudioManifest:
    """Reusable timeline metadata for a hydrophone job."""

    provider_kind: str
    hydrophone_id: str
    local_cache_path: str
    job_start_timestamp: float
    job_end_timestamp: float
    entries: tuple[HLSManifestEntry, ...] = ()


_manifest_cache_lock = threading.Lock()
_manifest_cache: OrderedDict[str, TimelineAudioManifest] = OrderedDict()

_pcm_cache_lock = threading.Lock()
_pcm_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
_pcm_cache_bytes = 0


def clear_timeline_audio_caches() -> None:
    """Clear in-memory manifest and PCM caches (used by tests)."""
    global _pcm_cache_bytes

    with _manifest_cache_lock:
        _manifest_cache.clear()
    with _pcm_cache_lock:
        _pcm_cache.clear()
        _pcm_cache_bytes = 0


def get_timeline_audio_cache_stats() -> dict[str, int]:
    """Return basic cache stats for tests and diagnostics."""
    with _manifest_cache_lock:
        manifest_items = len(_manifest_cache)
    with _pcm_cache_lock:
        pcm_items = len(_pcm_cache)
        pcm_bytes = _pcm_cache_bytes
    return {
        "manifest_items": manifest_items,
        "pcm_items": pcm_items,
        "pcm_bytes": pcm_bytes,
    }


def resolve_timeline_audio(
    *,
    hydrophone_id: str,
    local_cache_path: str,
    job_start_timestamp: float,
    job_end_timestamp: float,
    start_sec: float,
    duration_sec: float,
    target_sr: int = 32000,
    noaa_cache_path: str | None = None,
    timeline_cache=None,
    job_id: str | None = None,
    manifest_cache_items: int = 8,
    pcm_cache_max_bytes: int = 128 * 1024 * 1024,
) -> np.ndarray:
    """Return audio samples for an arbitrary timeline-absolute range.

    Dispatches to the appropriate archive provider based on source type.
    Gaps in coverage are filled with silence.
    Returns 1-D float32 array of length int(target_sr * duration_sec).
    """
    effective_start = max(start_sec, job_start_timestamp)
    effective_end = min(start_sec + duration_sec, job_end_timestamp)
    effective_duration = max(0.0, effective_end - effective_start)

    n_samples = int(target_sr * duration_sec)

    if effective_duration <= 0:
        return np.zeros(n_samples, dtype=np.float32)

    from humpback.config import get_archive_source

    source = get_archive_source(hydrophone_id) if hydrophone_id else None
    provider_kind = source["provider_kind"] if source else "orcasound_hls"

    if provider_kind == "orcasound_hls":
        audio = _resolve_audio_from_hls_cache(
            hydrophone_id=hydrophone_id,
            local_cache_path=local_cache_path,
            start_sec=effective_start,
            duration_sec=effective_duration,
            target_sr=target_sr,
            timeline_cache=timeline_cache,
            job_id=job_id,
            job_start_timestamp=job_start_timestamp,
            job_end_timestamp=job_end_timestamp,
            manifest_cache_items=manifest_cache_items,
            pcm_cache_max_bytes=pcm_cache_max_bytes,
        )
    else:
        audio = _resolve_audio_from_provider(
            hydrophone_id=hydrophone_id,
            noaa_cache_path=noaa_cache_path,
            start_sec=effective_start,
            duration_sec=effective_duration,
            target_sr=target_sr,
        )

    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    elif len(audio) > n_samples:
        audio = audio[:n_samples]

    return audio.astype(np.float32)


def _resolve_audio_from_provider(
    *,
    hydrophone_id: str,
    noaa_cache_path: str | None,
    start_sec: float,
    duration_sec: float,
    target_sr: int,
) -> np.ndarray:
    """Resolve audio via the archive provider abstraction (NOAA, etc.)."""
    from humpback.classifier.providers import build_archive_playback_provider

    n_samples = int(target_sr * duration_sec)
    end_sec = start_sec + duration_sec

    try:
        provider = build_archive_playback_provider(
            hydrophone_id,
            cache_path=None,
            noaa_cache_path=noaa_cache_path,
        )
        timeline = provider.build_timeline(start_sec, end_sec)
        if not timeline:
            logger.debug(
                "No segments found for %s [%.0f-%.0f]",
                hydrophone_id,
                start_sec,
                end_sec,
            )
            return np.zeros(n_samples, dtype=np.float32)

        output = np.zeros(n_samples, dtype=np.float32)
        use_chunked = hasattr(provider, "iter_decoded_segment_chunks")

        for segment in timeline:
            clip_start = max(segment.start_ts, start_sec)
            clip_end = min(segment.end_ts, end_sec)
            if clip_end <= clip_start:
                continue

            seg_clip_start = clip_start - segment.start_ts
            seg_clip_end = clip_end - segment.start_ts
            dst_start = int(round((clip_start - start_sec) * target_sr))

            try:
                seg_bytes = provider.fetch_segment(segment.key)
            except Exception:
                logger.debug(
                    "Failed to fetch segment %s, filling with silence",
                    segment.key,
                )
                continue

            try:
                if use_chunked:
                    chunks: list[np.ndarray] = []
                    for chunk_audio, _ in provider.iter_decoded_segment_chunks(  # type: ignore[attr-defined]
                        segment.key,
                        seg_bytes,
                        target_sr,
                        clip_start_sec=seg_clip_start,
                        clip_end_sec=seg_clip_end,
                        chunk_seconds=max(1.0, seg_clip_end - seg_clip_start),
                    ):
                        chunks.append(chunk_audio)
                    decoded = (
                        np.concatenate(chunks)
                        if chunks
                        else np.array([], dtype=np.float32)
                    )
                else:
                    full = provider.decode_segment(seg_bytes, target_sr)
                    src_start = max(0, int(round(seg_clip_start * target_sr)))
                    src_end = min(len(full), int(round(seg_clip_end * target_sr)))
                    decoded = full[src_start:src_end]
            except Exception:
                logger.debug(
                    "Failed to decode segment %s, filling with silence",
                    segment.key,
                )
                continue

            if len(decoded) == 0:
                continue

            dst_end = min(dst_start + len(decoded), n_samples)
            actual_len = dst_end - dst_start
            if actual_len > 0:
                output[dst_start:dst_end] = decoded[:actual_len]

        return output

    except Exception:
        logger.exception(
            "Failed to resolve audio for %s [%.0f-%.0f]",
            hydrophone_id,
            start_sec,
            end_sec,
        )
        return np.zeros(n_samples, dtype=np.float32)


def _resolve_audio_from_hls_cache(
    *,
    hydrophone_id: str,
    local_cache_path: str,
    start_sec: float,
    duration_sec: float,
    target_sr: int,
    timeline_cache=None,
    job_id: str | None = None,
    job_start_timestamp: float | None = None,
    job_end_timestamp: float | None = None,
    manifest_cache_items: int = 8,
    pcm_cache_max_bytes: int = 128 * 1024 * 1024,
) -> np.ndarray:
    """Resolve audio from HLS .ts segments in the local cache."""
    n_samples = int(target_sr * duration_sec)
    end_sec = start_sec + duration_sec

    try:
        manifest = None
        if (
            job_id
            and job_start_timestamp is not None
            and job_end_timestamp is not None
            and local_cache_path
        ):
            manifest = _get_or_build_hls_manifest(
                hydrophone_id=hydrophone_id,
                local_cache_path=local_cache_path,
                job_start_timestamp=job_start_timestamp,
                job_end_timestamp=job_end_timestamp,
                timeline_cache=timeline_cache,
                job_id=job_id,
                manifest_cache_items=manifest_cache_items,
            )

        if manifest is None:
            from humpback.classifier.s3_stream import build_hls_timeline_for_range

            timeline = build_hls_timeline_for_range(
                hydrophone_id=hydrophone_id,
                local_cache_path=local_cache_path,
                start_epoch=start_sec,
                end_epoch=end_sec,
            )
            entries = tuple(
                HLSManifestEntry(
                    segment_path=seg_path,
                    start_epoch=seg_start,
                    duration_sec=seg_duration,
                )
                for seg_path, seg_start, seg_duration in timeline
            )
            manifest = TimelineAudioManifest(
                provider_kind="orcasound_hls",
                hydrophone_id=hydrophone_id,
                local_cache_path=local_cache_path,
                job_start_timestamp=start_sec,
                job_end_timestamp=end_sec,
                entries=entries,
            )

        if not manifest.entries:
            logger.debug(
                "No HLS segments found for %s [%.0f-%.0f]",
                hydrophone_id,
                start_sec,
                end_sec,
            )
            return np.zeros(n_samples, dtype=np.float32)

        return _resolve_audio_from_hls_manifest(
            manifest=manifest,
            start_sec=start_sec,
            duration_sec=duration_sec,
            target_sr=target_sr,
            pcm_cache_max_bytes=pcm_cache_max_bytes,
        )

    except Exception:
        logger.exception(
            "Failed to resolve HLS audio for %s [%.0f-%.0f]",
            hydrophone_id,
            start_sec,
            end_sec,
        )
        return np.zeros(n_samples, dtype=np.float32)


def _resolve_audio_from_hls_manifest(
    *,
    manifest: TimelineAudioManifest,
    start_sec: float,
    duration_sec: float,
    target_sr: int,
    pcm_cache_max_bytes: int,
) -> np.ndarray:
    """Resolve audio using a reusable manifest and decoded PCM cache."""
    n_samples = int(target_sr * duration_sec)
    end_sec = start_sec + duration_sec
    output = np.zeros(n_samples, dtype=np.float32)

    for entry in manifest.entries:
        seg_start = entry.start_epoch
        seg_end = seg_start + entry.duration_sec
        overlap_start = max(seg_start, start_sec)
        overlap_end = min(seg_end, end_sec)
        if overlap_end <= overlap_start:
            continue

        audio = _get_cached_hls_pcm(
            entry.segment_path,
            target_sr,
            pcm_cache_max_bytes=pcm_cache_max_bytes,
        )
        if len(audio) == 0:
            continue

        out_start_sample = int(round((overlap_start - start_sec) * target_sr))
        out_end_sample = min(
            int(round((overlap_end - start_sec) * target_sr)), n_samples
        )

        audio_start_sample = max(0, int(round((overlap_start - seg_start) * target_sr)))
        n_copy = out_end_sample - out_start_sample
        audio_end_sample = min(len(audio), audio_start_sample + n_copy)
        n_copy = audio_end_sample - audio_start_sample
        if n_copy <= 0:
            continue

        output[out_start_sample : out_start_sample + n_copy] = audio[
            audio_start_sample:audio_end_sample
        ]

    return output


def _get_or_build_hls_manifest(
    *,
    hydrophone_id: str,
    local_cache_path: str,
    job_start_timestamp: float,
    job_end_timestamp: float,
    timeline_cache,
    job_id: str,
    manifest_cache_items: int,
) -> TimelineAudioManifest:
    key = _manifest_cache_key(
        hydrophone_id=hydrophone_id,
        local_cache_path=local_cache_path,
        job_start_timestamp=job_start_timestamp,
        job_end_timestamp=job_end_timestamp,
    )

    cached = _get_cached_manifest(key)
    if cached is not None:
        return cached

    if timeline_cache is not None:
        persisted = timeline_cache.get_audio_manifest(job_id)
        manifest = _deserialize_manifest(persisted)
        if (
            manifest is not None
            and manifest.hydrophone_id == hydrophone_id
            and manifest.local_cache_path == local_cache_path
            and manifest.job_start_timestamp == float(job_start_timestamp)
            and manifest.job_end_timestamp == float(job_end_timestamp)
        ):
            _put_cached_manifest(key, manifest, manifest_cache_items)
            return manifest

    from humpback.classifier.s3_stream import build_hls_timeline_for_range

    timeline = build_hls_timeline_for_range(
        hydrophone_id=hydrophone_id,
        local_cache_path=local_cache_path,
        start_epoch=job_start_timestamp,
        end_epoch=job_end_timestamp,
    )
    manifest = TimelineAudioManifest(
        provider_kind="orcasound_hls",
        hydrophone_id=hydrophone_id,
        local_cache_path=local_cache_path,
        job_start_timestamp=float(job_start_timestamp),
        job_end_timestamp=float(job_end_timestamp),
        entries=tuple(
            HLSManifestEntry(
                segment_path=seg_path,
                start_epoch=seg_start,
                duration_sec=seg_duration,
            )
            for seg_path, seg_start, seg_duration in timeline
        ),
    )
    _put_cached_manifest(key, manifest, manifest_cache_items)
    if timeline_cache is not None:
        timeline_cache.put_audio_manifest(job_id, _serialize_manifest(manifest))
    return manifest


def _decode_hls_segment_file(segment_path: str, target_sr: int) -> np.ndarray:
    """Decode one local HLS segment to float32 PCM."""
    from humpback.classifier.s3_stream import decode_ts_bytes

    return decode_ts_bytes(Path(segment_path).read_bytes(), target_sr)


def _get_cached_hls_pcm(
    segment_path: str,
    target_sr: int,
    *,
    pcm_cache_max_bytes: int,
) -> np.ndarray:
    key = (segment_path, target_sr)

    with _pcm_cache_lock:
        cached = _pcm_cache.pop(key, None)
        if cached is not None:
            _pcm_cache[key] = cached
            return cached

    decoded = _decode_hls_segment_file(segment_path, target_sr)

    if pcm_cache_max_bytes <= 0:
        return decoded

    global _pcm_cache_bytes
    with _pcm_cache_lock:
        existing = _pcm_cache.pop(key, None)
        if existing is not None:
            _pcm_cache_bytes -= existing.nbytes
        _pcm_cache[key] = decoded
        _pcm_cache_bytes += decoded.nbytes
        while _pcm_cache and _pcm_cache_bytes > pcm_cache_max_bytes:
            _, evicted = _pcm_cache.popitem(last=False)
            _pcm_cache_bytes -= evicted.nbytes
    return decoded


def _manifest_cache_key(
    *,
    hydrophone_id: str,
    local_cache_path: str,
    job_start_timestamp: float,
    job_end_timestamp: float,
) -> str:
    return (
        f"{hydrophone_id}|{local_cache_path}|"
        f"{float(job_start_timestamp):.6f}|{float(job_end_timestamp):.6f}"
    )


def _get_cached_manifest(key: str) -> TimelineAudioManifest | None:
    with _manifest_cache_lock:
        cached = _manifest_cache.pop(key, None)
        if cached is None:
            return None
        _manifest_cache[key] = cached
        return cached


def _put_cached_manifest(
    key: str, manifest: TimelineAudioManifest, max_items: int
) -> None:
    if max_items <= 0:
        return
    with _manifest_cache_lock:
        _manifest_cache.pop(key, None)
        _manifest_cache[key] = manifest
        while len(_manifest_cache) > max_items:
            _manifest_cache.popitem(last=False)


def _serialize_manifest(manifest: TimelineAudioManifest) -> dict:
    return {
        "provider_kind": manifest.provider_kind,
        "hydrophone_id": manifest.hydrophone_id,
        "local_cache_path": manifest.local_cache_path,
        "job_start_timestamp": manifest.job_start_timestamp,
        "job_end_timestamp": manifest.job_end_timestamp,
        "entries": [
            {
                "segment_path": entry.segment_path,
                "start_epoch": entry.start_epoch,
                "duration_sec": entry.duration_sec,
            }
            for entry in manifest.entries
        ],
    }


def _deserialize_manifest(payload: dict | None) -> TimelineAudioManifest | None:
    if not isinstance(payload, dict):
        return None
    try:
        entries_payload = payload.get("entries", [])
        if not isinstance(entries_payload, list):
            return None
        entries = tuple(
            HLSManifestEntry(
                segment_path=str(item["segment_path"]),
                start_epoch=float(item["start_epoch"]),
                duration_sec=float(item["duration_sec"]),
            )
            for item in entries_payload
            if isinstance(item, dict)
        )
        return TimelineAudioManifest(
            provider_kind=str(payload["provider_kind"]),
            hydrophone_id=str(payload["hydrophone_id"]),
            local_cache_path=str(payload["local_cache_path"]),
            job_start_timestamp=float(payload["job_start_timestamp"]),
            job_end_timestamp=float(payload["job_end_timestamp"]),
            entries=entries,
        )
    except (KeyError, TypeError, ValueError):
        return None
