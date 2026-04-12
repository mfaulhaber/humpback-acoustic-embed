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
    """
    audio = audio_loader(region)
    logmel = normalize_per_region_zscore(extract_logmel(audio, feature_config))
    features_t = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).float()

    model.eval()
    with torch.no_grad():
        logits = model(features_t)
        probs_t = torch.sigmoid(logits).squeeze(0)
    probs = probs_t.detach().cpu().numpy().astype(np.float32)

    hop_sec = float(feature_config.hop_length) / float(feature_config.sample_rate)
    return decode_events(
        frame_probs=probs,
        region_id=str(region.region_id),
        region_start_sec=float(region.padded_start_sec),
        hop_sec=hop_sec,
        config=decoder_config,
    )
