"""Pass 3 event crop dataset and variable-length batching.

``EventCropDataset`` lazy-loads audio via a caller-supplied
``AudioLoader``, crops each event at its exact ``[start_sec, end_sec]``
bounds, extracts log-mel features using the Pass 2 feature pipeline,
and returns ``(features, label_vector)`` pairs.  The matching
``collate_fn`` pads spectrograms to the max time dimension within each
batch.
"""

from __future__ import annotations

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

AudioLoader = Callable[[Any], np.ndarray]
"""Callable that fetches the full audio for one training sample.

Returns a 1-D float array already resampled to the feature config's
sample rate.  The dataset crops the event window from this array.
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
        audio = self.audio_loader(sample)
        sr = self.feature_config.sample_rate

        start_sample = max(0, int(round(sample.start_sec * sr)))
        end_sample = min(len(audio), int(round(sample.end_sec * sr)))
        if end_sample <= start_sample:
            end_sample = min(len(audio), start_sample + sr)

        crop = audio[start_sample:end_sample]
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
