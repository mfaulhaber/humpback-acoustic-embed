"""Pass 2 segmentation training dataset and framewise-target builder.

Two pieces:

- ``build_framewise_target`` — pure function that turns a sample's
  event bounds into a ``(T,)`` float32 0/1 vector. Called by both the
  dataset's ``__getitem__`` (per-sample training targets) and
  ``compute_pos_weight`` (class-balance statistics before training).
  No audio, no models, no I/O.
- ``SegmentationSampleDataset`` — a ``torch.utils.data.Dataset`` that
  lazy-loads each sample's crop audio via a caller-supplied
  ``AudioLoader`` callable, extracts log-mel features, normalizes per
  region, and returns ``(features, target, mask)`` plus a matching
  ``collate_fn`` helper for variable-length batching.

Frame convention: frame ``i`` covers the half-open tile
``[i * hop_sec, (i + 1) * hop_sec)`` in crop-relative seconds, with
center at ``(i + 0.5) * hop_sec``. This matches the decoder's round-trip
math in ``decoder.decode_events``.
"""

from __future__ import annotations

import json
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

AudioLoader = Callable[[Any], np.ndarray]
"""Callable that fetches the crop audio for one training sample.

The caller (the training worker) supplies a concrete loader that knows
how to resolve ``audio_file_id`` vs. the hydrophone triple into audio
bytes. Typed as ``Callable[[Any], np.ndarray]`` so the dataset stays
decoupled from SQLAlchemy — tests pass dataclass fakes with the same
attribute shape as ``SegmentationTrainingSample``.
"""


def _parse_events(events_json: str) -> list[dict[str, float]]:
    if not events_json:
        return []
    parsed = json.loads(events_json)
    if not isinstance(parsed, list):
        raise ValueError(
            f"events_json must be a JSON list, got {type(parsed).__name__}"
        )
    return parsed


def build_framewise_target(
    events_json: str,
    crop_start_sec: float,
    crop_end_sec: float,
    feature_config: SegmentationFeatureConfig,
) -> np.ndarray:
    """Return a ``(T,)`` float32 target vector for one training sample.

    ``events_json`` is the sample's audio-relative event list. The crop
    spans ``[crop_start_sec, crop_end_sec]`` in the same audio timeline.
    ``T`` is computed from the crop duration and the feature hop using
    the ``center=True`` frame count ``1 + n_samples // hop_length`` that
    ``librosa.feature.melspectrogram`` produces.

    Each frame is assigned ``1.0`` iff its center
    ``(i + 0.5) * hop_sec`` in crop-relative seconds falls inside any
    event's crop-relative span ``[event_start, event_end)`` — start
    bounds are inclusive, end bounds are exclusive.
    """
    events = _parse_events(events_json)
    duration_sec = max(0.0, crop_end_sec - crop_start_sec)
    n_samples = int(round(duration_sec * feature_config.sample_rate))
    n_frames = 1 + n_samples // feature_config.hop_length
    target = np.zeros(n_frames, dtype=np.float32)
    if not events or n_frames == 0:
        return target

    hop_sec = float(feature_config.hop_length) / float(feature_config.sample_rate)
    centers = (np.arange(n_frames, dtype=np.float64) + 0.5) * hop_sec
    for event in events:
        event_start = float(event["start_sec"]) - crop_start_sec
        event_end = float(event["end_sec"]) - crop_start_sec
        if event_end <= event_start:
            continue
        frame_mask = (centers >= event_start) & (centers < event_end)
        target[frame_mask] = 1.0
    return target


class SegmentationSampleDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """``torch`` Dataset over ``SegmentationTrainingSample``-shaped rows.

    Lazy-loads audio inside ``__getitem__`` via the caller-supplied
    ``audio_loader``; no I/O happens in ``__init__``.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        feature_config: SegmentationFeatureConfig,
        audio_loader: AudioLoader,
    ) -> None:
        super().__init__()
        self.samples: list[Any] = list(samples)
        self.feature_config = feature_config
        self.audio_loader = audio_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        audio = self.audio_loader(sample)
        logmel = extract_logmel(audio, self.feature_config)
        logmel = normalize_per_region_zscore(logmel)
        target_np = build_framewise_target(
            sample.events_json,
            sample.crop_start_sec,
            sample.crop_end_sec,
            self.feature_config,
        )

        # librosa's ``center=True`` frame count can differ from our
        # formula by at most one frame; align target length to the
        # actual feature frame count so downstream masked loss is shape-
        # consistent.
        n_feat = int(logmel.shape[1])
        if target_np.shape[0] > n_feat:
            target_np = target_np[:n_feat]
        elif target_np.shape[0] < n_feat:
            target_np = np.pad(target_np, (0, n_feat - target_np.shape[0]))

        features = torch.from_numpy(logmel).unsqueeze(0).float()
        target = torch.from_numpy(target_np).float()
        mask = torch.ones(n_feat, dtype=torch.bool)
        return features, target, mask


def collate_fn(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a variable-length batch of ``(features, target, mask)`` tuples.

    ``features`` pad with zeros, ``target`` pads with zeros, ``mask``
    pads with ``False`` so the masked loss ignores the padded tail.
    """
    if not batch:
        raise ValueError("collate_fn expects a non-empty batch")

    max_t = max(int(features.shape[-1]) for features, _, _ in batch)
    first_features = batch[0][0]
    n_channels = int(first_features.shape[0])
    n_mels = int(first_features.shape[1])
    batch_size = len(batch)

    features_out = torch.zeros(batch_size, n_channels, n_mels, max_t)
    targets_out = torch.zeros(batch_size, max_t)
    masks_out = torch.zeros(batch_size, max_t, dtype=torch.bool)
    for i, (features, target, mask) in enumerate(batch):
        t = int(features.shape[-1])
        features_out[i, :, :, :t] = features
        targets_out[i, :t] = target
        masks_out[i, :t] = mask
    return features_out, targets_out, masks_out


def compute_pos_weight(dataset: SegmentationSampleDataset) -> float:
    """Return ``total_neg_frames / total_pos_frames`` over ``dataset``.

    Iterates ``build_framewise_target`` over every sample — no audio
    loading required, which is why this is cheap enough to run once at
    the start of training instead of per batch. On an all-negative
    dataset, returns ``1.0`` and logs a warning so the trainer still
    produces a well-defined loss.
    """
    total_pos = 0
    total_neg = 0
    for sample in dataset.samples:
        target = build_framewise_target(
            sample.events_json,
            sample.crop_start_sec,
            sample.crop_end_sec,
            dataset.feature_config,
        )
        pos = int(target.sum())
        total_pos += pos
        total_neg += int(target.shape[0]) - pos
    if total_pos == 0:
        logger.warning(
            "compute_pos_weight: dataset has zero positive frames, returning 1.0"
        )
        return 1.0
    return float(total_neg) / float(total_pos)
