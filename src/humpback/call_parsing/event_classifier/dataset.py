"""Pass 3 event crop dataset and variable-length batching.

``EventCropDataset`` lazy-loads audio via a caller-supplied
``AudioLoader``, crops each event at its exact ``[start_sec, end_sec]``
bounds, extracts log-mel features using the Pass 2 feature pipeline,
and returns ``(features, label_vector)`` pairs.  The matching
``collate_fn`` pads spectrograms to the max time dimension within each
batch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from humpback.call_parsing.segmentation.features import (
    extract_logmel,
    normalize_per_region_zscore,
)
from humpback.schemas.call_parsing import SegmentationFeatureConfig

logger = logging.getLogger(__name__)

AudioLoader = Callable[[Any], tuple[np.ndarray, float]]
"""Callable returning ``(audio, audio_start_sec)`` for one training sample.

``audio`` is a 1-D float array at the feature config's sample rate.
``audio_start_sec`` is the absolute time of the first sample in the
source's coordinate system.  For file-based sources this is ``0.0``;
for hydrophone sources it is the start of the loaded context window.
The dataset subtracts this offset so that ``sample.start_sec`` maps to
the correct sample index.
"""


class EventCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch Dataset over event crops for multi-label classification.

    Each sample must expose ``.start_sec``, ``.end_sec``, and
    ``.type_index`` (int index into the vocabulary) attributes.  The
    audio loader receives the sample and returns the full source audio;
    the dataset crops the ``[start_sec, end_sec]`` slice itself.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        feature_config: SegmentationFeatureConfig,
        audio_loader: AudioLoader,
        n_types: int,
    ) -> None:
        super().__init__()
        self.samples: list[Any] = list(samples)
        self.feature_config = feature_config
        self.audio_loader = audio_loader
        self.n_types = n_types

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        audio, ctx_start = self.audio_loader(sample)
        sr = self.feature_config.sample_rate

        rel_start = sample.start_sec - ctx_start
        rel_end = sample.end_sec - ctx_start
        start_sample = max(0, int(round(rel_start * sr)))
        end_sample = min(len(audio), int(round(rel_end * sr)))
        if end_sample <= start_sample:
            end_sample = min(len(audio), start_sample + sr)

        crop = audio[start_sample:end_sample]
        min_samples = self.feature_config.n_fft
        if len(crop) < min_samples:
            logger.warning(
                "Event crop too short (%d samples, need %d): "
                "start_sec=%.2f end_sec=%.2f",
                len(crop),
                min_samples,
                sample.start_sec,
                sample.end_sec,
            )
            features = torch.zeros(1, self.feature_config.n_mels, 1)
            label = torch.zeros(self.n_types, dtype=torch.float32)
            label[sample.type_index] = 1.0
            return features, label

        logmel = extract_logmel(crop, self.feature_config)
        logmel = normalize_per_region_zscore(logmel)

        features = torch.from_numpy(logmel).unsqueeze(0).float()

        label = torch.zeros(self.n_types, dtype=torch.float32)
        label[sample.type_index] = 1.0

        return features, label


def collate_fn(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length spectrograms to the max time within the batch.

    ``features`` are zero-padded along the time axis. ``labels`` are
    stacked directly (fixed-size multi-hot vectors).
    """
    if not batch:
        raise ValueError("collate_fn expects a non-empty batch")

    max_t = max(int(feat.shape[-1]) for feat, _ in batch)
    first_feat = batch[0][0]
    n_channels = int(first_feat.shape[0])
    n_mels = int(first_feat.shape[1])
    batch_size = len(batch)

    features_out = torch.zeros(batch_size, n_channels, n_mels, max_t)
    labels_out = torch.stack([label for _, label in batch])

    for i, (feat, _) in enumerate(batch):
        t = int(feat.shape[-1])
        features_out[i, :, :, :t] = feat

    return features_out, labels_out
