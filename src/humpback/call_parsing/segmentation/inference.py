"""Per-region segmentation inference.

One pure function: fetch a region's audio, extract features, run one
CRNN forward pass, and hand the frame probabilities to
``decode_events``. This is the step the event segmentation worker calls
for each region in ``regions.parquet``; the worker owns DB and parquet
I/O, this module owns the tensor path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from humpback.call_parsing.segmentation.decoder import decode_events
from humpback.call_parsing.segmentation.features import (
    extract_logmel,
    normalize_per_region_zscore,
)
from humpback.call_parsing.types import Event
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
)

RegionAudioLoader = Callable[[Any], np.ndarray]
"""Callable that fetches audio for one Pass 1 ``Region``.

Takes the region object, returns a 1-D float array at
``feature_config.sample_rate``. The worker supplies a concrete loader
that knows how to resolve the upstream Pass 1 job's source (audio file
vs. hydrophone range) into bytes.
"""


# Maximum audio duration for a single forward pass. Regions longer than
# this are processed with overlapping windows and the per-frame
# probabilities are averaged over the overlap zone. This keeps inference
# consistent with the 30-second crops used during feedback training.
_MAX_WINDOW_SEC: float = 30.0
_WINDOW_HOP_SEC: float = 15.0


def _infer_single(
    model: nn.Module,
    audio: np.ndarray,
    feature_config: SegmentationFeatureConfig,
) -> np.ndarray:
    """Forward pass on one audio chunk, return frame probabilities (1-D)."""
    logmel = normalize_per_region_zscore(extract_logmel(audio, feature_config))
    features_t = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(features_t)
        probs_t = torch.sigmoid(logits).squeeze(0)
    return probs_t.detach().cpu().numpy().astype(np.float32)


def _infer_windowed(
    model: nn.Module,
    audio: np.ndarray,
    feature_config: SegmentationFeatureConfig,
    max_window_sec: float = _MAX_WINDOW_SEC,
    window_hop_sec: float = _WINDOW_HOP_SEC,
) -> np.ndarray:
    """Sliding-window inference with averaged overlaps."""
    sr = feature_config.sample_rate
    hop_length = feature_config.hop_length
    total_samples = len(audio)
    total_duration = total_samples / sr

    if total_duration <= max_window_sec:
        return _infer_single(model, audio, feature_config)

    # Estimate total output frames from a full-length pass so the
    # accumulator has the right size.
    total_frames = int(np.ceil(total_samples / hop_length))
    prob_sum = np.zeros(total_frames, dtype=np.float64)
    weight = np.zeros(total_frames, dtype=np.float64)

    window_samples = int(max_window_sec * sr)
    hop_samples = int(window_hop_sec * sr)

    offset = 0
    while offset < total_samples:
        end = min(offset + window_samples, total_samples)
        # If the remaining tail is too short, extend the window backwards
        if total_samples - offset < window_samples // 2 and offset > 0:
            offset = max(0, total_samples - window_samples)
            end = total_samples

        chunk = audio[offset:end]
        chunk_probs = _infer_single(model, chunk, feature_config)

        frame_offset = int(np.round(offset / hop_length))
        n_frames = len(chunk_probs)
        frame_end = min(frame_offset + n_frames, total_frames)
        usable = frame_end - frame_offset
        prob_sum[frame_offset:frame_end] += chunk_probs[:usable]
        weight[frame_offset:frame_end] += 1.0

        if end >= total_samples:
            break
        offset += hop_samples

    # Avoid division by zero for any unvisited frames.
    weight = np.maximum(weight, 1.0)
    return (prob_sum / weight).astype(np.float32)


def run_inference(
    model: nn.Module,
    region: Any,
    audio_loader: RegionAudioLoader,
    feature_config: SegmentationFeatureConfig,
    decoder_config: SegmentationDecoderConfig,
) -> list[Event]:
    """Run the CRNN on one region and return decoded ``Event`` rows.

    ``region`` must expose ``region_id`` and ``padded_start_sec``; the
    decoder uses the padded start as the absolute-time anchor so every
    ``Event``'s timestamps land on the source audio's timeline.

    Regions longer than 30 seconds are processed with overlapping
    sliding windows so the CRNN sees the same sequence lengths it was
    trained on.
    """
    audio = audio_loader(region)

    model.eval()
    probs = _infer_windowed(model, audio, feature_config)

    hop_sec = float(feature_config.hop_length) / float(feature_config.sample_rate)
    return decode_events(
        frame_probs=probs,
        region_id=str(region.region_id),
        region_start_sec=float(region.padded_start_sec),
        hop_sec=hop_sec,
        config=decoder_config,
    )
