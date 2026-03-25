"""Resolve audio from HLS cache for arbitrary timeline-absolute positions."""

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
) -> np.ndarray:
    """Return audio samples for an arbitrary timeline-absolute range.

    Gaps in the HLS cache are filled with silence.
    Returns 1-D float32 array of length int(target_sr * duration_sec).
    """
    effective_start = max(start_sec, job_start_timestamp)
    effective_end = min(start_sec + duration_sec, job_end_timestamp)
    effective_duration = max(0.0, effective_end - effective_start)

    n_samples = int(target_sr * duration_sec)

    if effective_duration <= 0:
        return np.zeros(n_samples, dtype=np.float32)

    audio = _resolve_audio_from_hls_cache(
        hydrophone_id=hydrophone_id,
        local_cache_path=local_cache_path,
        start_sec=effective_start,
        duration_sec=effective_duration,
        target_sr=target_sr,
    )

    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    elif len(audio) > n_samples:
        audio = audio[:n_samples]

    return audio.astype(np.float32)


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
