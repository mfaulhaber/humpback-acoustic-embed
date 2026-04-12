"""Feature extractor for Pass 2 segmentation.

Pure functions that turn raw audio into a ``(n_mels, T)`` log-mel
spectrogram and z-score it per region. Parameters are frozen in
``SegmentationFeatureConfig`` — see the design doc for the rationale on
``fmin=20``, ``fmax=4000``, ``hop_length=512``, and friends.

This module is deliberately independent from
``humpback/processing/features.py``. The Perch feature pipeline lives in
the sensitive-components list and uses a different parameter set; Pass 2
features stay siloed in the Pass 2 subpackage.
"""

from __future__ import annotations

import librosa
import numpy as np

from humpback.schemas.call_parsing import SegmentationFeatureConfig


def extract_logmel(audio: np.ndarray, config: SegmentationFeatureConfig) -> np.ndarray:
    """Return the log-mel spectrogram of ``audio`` at shape ``(n_mels, T)``.

    ``audio`` is expected as a 1-D float array already resampled to
    ``config.sample_rate``. Frequency bins above ``config.fmax`` and
    below ``config.fmin`` are clamped by ``librosa.feature.melspectrogram``.
    """
    if audio.ndim != 1:
        raise ValueError(f"extract_logmel expects 1-D audio, got shape {audio.shape}")

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32, copy=False),
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32, copy=False)


def normalize_per_region_zscore(logmel: np.ndarray) -> np.ndarray:
    """Return ``logmel`` z-scored across all bins / frames.

    A single scalar mean and std are computed over the entire
    ``(n_mels, T)`` tile — matching the "per-region" rule documented in
    the design spec. A small epsilon guards against divide-by-zero on
    silent inputs.
    """
    if logmel.size == 0:
        return logmel.astype(np.float32, copy=False)
    mean = float(logmel.mean())
    std = float(logmel.std())
    eps = 1e-6
    normalized = (logmel - mean) / (std + eps)
    return normalized.astype(np.float32, copy=False)


def frame_index_to_audio_sec(
    frame_idx: int, config: SegmentationFeatureConfig
) -> float:
    """Return the audio time (seconds) corresponding to a frame index."""
    return float(frame_idx) * float(config.hop_length) / float(config.sample_rate)


def audio_sec_to_frame_index(time_sec: float, config: SegmentationFeatureConfig) -> int:
    """Return the largest frame index whose start ≤ ``time_sec``."""
    if time_sec < 0:
        return 0
    return int(time_sec * config.sample_rate / config.hop_length)
