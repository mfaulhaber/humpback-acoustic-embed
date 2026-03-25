"""Resolve audio from archive providers for arbitrary timeline-absolute positions."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
    """Resolve audio via the archive provider abstraction (NOAA, etc.).

    Pre-allocates a silence buffer and places decoded segment audio at the
    correct timeline offset so gaps between segments remain as silence.
    """
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

        # Pre-allocate output buffer (silence); place each segment at the
        # correct offset so inter-segment gaps stay silent.
        output = np.zeros(n_samples, dtype=np.float32)
        use_chunked = hasattr(provider, "iter_decoded_segment_chunks")

        for segment in timeline:
            clip_start = max(segment.start_ts, start_sec)
            clip_end = min(segment.end_ts, end_sec)
            if clip_end <= clip_start:
                continue

            # Offset within the segment file where our clip begins
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
                    # Chunked decode with ffmpeg seeking — only decodes
                    # the clip range, avoiding full-file decode.
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
                    # Full decode fallback
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
) -> np.ndarray:
    """Resolve audio from HLS .ts segments in the local cache."""
    from humpback.classifier.s3_stream import (
        build_hls_timeline_for_range,
        decode_segments_to_audio,
    )

    n_samples = int(target_sr * duration_sec)
    end_sec = start_sec + duration_sec

    try:
        timeline = build_hls_timeline_for_range(
            hydrophone_id=hydrophone_id,
            local_cache_path=local_cache_path,
            start_epoch=start_sec,
            end_epoch=end_sec,
        )
        if not timeline:
            logger.debug(
                "No HLS segments found for %s [%.0f-%.0f]",
                hydrophone_id,
                start_sec,
                end_sec,
            )
            return np.zeros(n_samples, dtype=np.float32)

        audio = decode_segments_to_audio(
            timeline=timeline,
            start_epoch=start_sec,
            end_epoch=end_sec,
            target_sr=target_sr,
        )
        return audio

    except Exception:
        logger.exception(
            "Failed to resolve HLS audio for %s [%.0f-%.0f]",
            hydrophone_id,
            start_sec,
            end_sec,
        )
        return np.zeros(n_samples, dtype=np.float32)
