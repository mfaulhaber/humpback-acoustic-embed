"""Tests for the Pass 2 segmentation dataset + framewise-target helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import torch

from humpback.call_parsing.segmentation.dataset import (
    SegmentationSampleDataset,
    build_framewise_target,
    collate_fn,
    compute_pos_weight,
)
from humpback.schemas.call_parsing import SegmentationFeatureConfig


@dataclass
class _FakeSample:
    """Duck-typed stand-in for ``SegmentationTrainingSample``."""

    events_json: str
    crop_start_sec: float
    crop_end_sec: float
    audio_file_id: Optional[str] = None
    hydrophone_id: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None


def test_build_framewise_target_single_event_frame_centers() -> None:
    cfg = SegmentationFeatureConfig()
    hop_sec = cfg.hop_length / cfg.sample_rate
    events = [{"start_sec": 0.1, "end_sec": 0.3}]
    target = build_framewise_target(
        json.dumps(events),
        crop_start_sec=0.0,
        crop_end_sec=1.0,
        feature_config=cfg,
    )
    centers = (np.arange(target.shape[0]) + 0.5) * hop_sec
    expected = ((centers >= 0.1) & (centers < 0.3)).astype(np.float32)
    np.testing.assert_array_equal(target, expected)
    assert target.sum() > 0


def test_build_framewise_target_multiple_events() -> None:
    cfg = SegmentationFeatureConfig()
    events = [
        {"start_sec": 0.1, "end_sec": 0.2},
        {"start_sec": 0.4, "end_sec": 0.6},
    ]
    target = build_framewise_target(
        json.dumps(events),
        crop_start_sec=0.0,
        crop_end_sec=1.0,
        feature_config=cfg,
    )
    hop_sec = cfg.hop_length / cfg.sample_rate
    centers = (np.arange(target.shape[0]) + 0.5) * hop_sec
    in_first = (centers >= 0.1) & (centers < 0.2)
    in_second = (centers >= 0.4) & (centers < 0.6)
    expected = (in_first | in_second).astype(np.float32)
    np.testing.assert_array_equal(target, expected)
    # Both events contributed at least one positive frame.
    assert target[in_first].sum() > 0
    assert target[in_second].sum() > 0


def test_build_framewise_target_empty_events_all_zeros() -> None:
    cfg = SegmentationFeatureConfig()
    target = build_framewise_target(
        "[]",
        crop_start_sec=0.0,
        crop_end_sec=1.0,
        feature_config=cfg,
    )
    assert target.dtype == np.float32
    assert target.shape[0] > 0
    assert target.sum() == 0.0


def test_build_framewise_target_frame_boundary_behavior() -> None:
    """Start bound inclusive, end bound exclusive on frame centers."""
    cfg = SegmentationFeatureConfig(sample_rate=1000, hop_length=100)
    # Frame 2's center is exactly 0.25 with hop_sec=0.1. An event ending
    # at 0.25 must exclude frame 2; an event starting at 0.25 must
    # include frame 2.
    end_on_center = [{"start_sec": 0.0, "end_sec": 0.25}]
    start_on_center = [{"start_sec": 0.25, "end_sec": 0.5}]

    target_end = build_framewise_target(
        json.dumps(end_on_center),
        crop_start_sec=0.0,
        crop_end_sec=1.0,
        feature_config=cfg,
    )
    target_start = build_framewise_target(
        json.dumps(start_on_center),
        crop_start_sec=0.0,
        crop_end_sec=1.0,
        feature_config=cfg,
    )
    assert target_end[2] == 0.0, "end bound on a frame center must exclude it"
    assert target_start[2] == 1.0, "start bound on a frame center must include it"


def test_build_framewise_target_uses_crop_relative_event_bounds() -> None:
    """events_json times are audio-relative; crop offsets must shift them."""
    cfg = SegmentationFeatureConfig(sample_rate=1000, hop_length=100)
    events = [{"start_sec": 100.1, "end_sec": 100.3}]
    target = build_framewise_target(
        json.dumps(events),
        crop_start_sec=100.0,
        crop_end_sec=101.0,
        feature_config=cfg,
    )
    hop_sec = 0.1
    centers = (np.arange(target.shape[0]) + 0.5) * hop_sec
    expected = ((centers >= 0.1) & (centers < 0.3)).astype(np.float32)
    np.testing.assert_array_equal(target, expected)


def test_collate_fn_pads_to_max_T_and_builds_matching_mask() -> None:
    n_mels = 4
    short_t, long_t = 6, 10
    features_a = torch.randn(1, n_mels, short_t)
    features_b = torch.randn(1, n_mels, long_t)
    target_a = torch.ones(short_t)
    target_b = torch.zeros(long_t)
    mask_a = torch.ones(short_t, dtype=torch.bool)
    mask_b = torch.ones(long_t, dtype=torch.bool)

    batch = [(features_a, target_a, mask_a), (features_b, target_b, mask_b)]
    features, targets, masks = collate_fn(batch)

    assert features.shape == (2, 1, n_mels, long_t)
    assert targets.shape == (2, long_t)
    assert masks.shape == (2, long_t)
    # First sample's real region is True, padded tail is False.
    assert masks[0, :short_t].all()
    assert not masks[0, short_t:].any()
    # Second sample had no padding — all True.
    assert masks[1].all()
    # Target padding is zero.
    assert targets[0, short_t:].sum() == 0.0
    # Feature content is preserved in the non-padded region.
    assert torch.allclose(features[0, 0, :, :short_t], features_a[0])
    assert torch.allclose(features[1, 0, :, :long_t], features_b[0])


def test_collate_fn_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        collate_fn([])


def test_compute_pos_weight_returns_neg_over_pos_ratio() -> None:
    cfg = SegmentationFeatureConfig(sample_rate=1000, hop_length=100)
    sample_a = _FakeSample(
        events_json=json.dumps([{"start_sec": 0.0, "end_sec": 0.5}]),
        crop_start_sec=0.0,
        crop_end_sec=1.0,
    )
    sample_b = _FakeSample(
        events_json="[]",
        crop_start_sec=0.0,
        crop_end_sec=1.0,
    )
    dataset = SegmentationSampleDataset(
        samples=[sample_a, sample_b],
        feature_config=cfg,
        audio_loader=lambda _s: np.zeros(0, dtype=np.float32),
    )
    target_a = build_framewise_target(
        sample_a.events_json,
        sample_a.crop_start_sec,
        sample_a.crop_end_sec,
        cfg,
    )
    target_b = build_framewise_target(
        sample_b.events_json,
        sample_b.crop_start_sec,
        sample_b.crop_end_sec,
        cfg,
    )
    pos = int(target_a.sum() + target_b.sum())
    neg = int(target_a.shape[0] + target_b.shape[0]) - pos
    expected = neg / max(pos, 1)

    weight = compute_pos_weight(dataset)
    assert weight == pytest.approx(expected)
    # Sanity-check: the dataset has real positives.
    assert pos > 0


def test_compute_pos_weight_all_negative_returns_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = SegmentationFeatureConfig()
    sample = _FakeSample(
        events_json="[]",
        crop_start_sec=0.0,
        crop_end_sec=1.0,
    )
    dataset = SegmentationSampleDataset(
        samples=[sample],
        feature_config=cfg,
        audio_loader=lambda _s: np.zeros(0, dtype=np.float32),
    )
    with caplog.at_level("WARNING"):
        weight = compute_pos_weight(dataset)
    assert weight == 1.0
    assert any("zero positive" in record.message for record in caplog.records), (
        "expected a warning about zero positive frames"
    )


def test_dataset_len_returns_sample_count() -> None:
    cfg = SegmentationFeatureConfig()
    samples = [
        _FakeSample(events_json="[]", crop_start_sec=0.0, crop_end_sec=1.0)
        for _ in range(3)
    ]
    dataset = SegmentationSampleDataset(
        samples=samples,
        feature_config=cfg,
        audio_loader=lambda _s: np.zeros(0, dtype=np.float32),
    )
    assert len(dataset) == 3


def test_dataset_getitem_returns_features_target_mask() -> None:
    cfg = SegmentationFeatureConfig()
    audio = np.random.default_rng(0).normal(size=cfg.sample_rate).astype(np.float32)
    sample = _FakeSample(
        events_json=json.dumps([{"start_sec": 0.1, "end_sec": 0.3}]),
        crop_start_sec=0.0,
        crop_end_sec=1.0,
    )
    dataset = SegmentationSampleDataset(
        samples=[sample],
        feature_config=cfg,
        audio_loader=lambda _s: audio,
    )
    features, target, mask = dataset[0]
    assert features.ndim == 3
    assert features.shape[0] == 1
    assert features.shape[1] == cfg.n_mels
    n_feat = int(features.shape[2])
    assert target.shape == (n_feat,)
    assert mask.shape == (n_feat,)
    assert mask.dtype == torch.bool
    assert mask.all()
    # At least one positive frame because an event is present.
    assert target.sum().item() > 0


def test_dataset_getitem_is_lazy_does_not_load_in_init() -> None:
    cfg = SegmentationFeatureConfig()
    call_count = {"n": 0}

    def tracking_loader(_sample: object) -> np.ndarray:
        call_count["n"] += 1
        return np.zeros(cfg.sample_rate, dtype=np.float32)

    samples = [
        _FakeSample(events_json="[]", crop_start_sec=0.0, crop_end_sec=1.0)
        for _ in range(3)
    ]
    dataset = SegmentationSampleDataset(
        samples=samples,
        feature_config=cfg,
        audio_loader=tracking_loader,
    )
    assert call_count["n"] == 0, "audio loader must not be called from __init__"
    _ = dataset[0]
    assert call_count["n"] == 1
