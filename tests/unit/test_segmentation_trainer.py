"""Unit tests for the Pass 2 segmentation trainer driver + inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from humpback.call_parsing.segmentation.inference import run_inference
from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.call_parsing.segmentation.trainer import (
    MaskedBCEWithLogitsLoss,
    match_events_by_iou,
    split_by_audio_source,
    train_model,
)
from humpback.call_parsing.types import Event
from humpback.ml.checkpointing import load_checkpoint
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
    SegmentationTrainingConfig,
)


@dataclass
class _FakeSample:
    events_json: str
    crop_start_sec: float
    crop_end_sec: float
    audio_file_id: Optional[str] = None
    hydrophone_id: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None


@dataclass
class _FakeRegion:
    region_id: str
    padded_start_sec: float
    padded_end_sec: float


def _make_event(start: float, end: float, confidence: float = 1.0) -> Event:
    return Event(
        event_id=f"ev-{start}-{end}",
        region_id="r1",
        start_sec=start,
        end_sec=end,
        center_sec=(start + end) / 2.0,
        segmentation_confidence=confidence,
    )


# ---- split_by_audio_source ----------------------------------------------


def test_split_by_audio_source_three_sources_no_leakage_and_deterministic() -> None:
    samples = [
        _FakeSample(
            events_json="[]",
            crop_start_sec=0.0,
            crop_end_sec=1.0,
            audio_file_id=f"file-{i // 2}",
        )
        for i in range(6)
    ]
    train_a, val_a = split_by_audio_source(samples, val_fraction=0.34, seed=42)
    train_b, val_b = split_by_audio_source(samples, val_fraction=0.34, seed=42)

    # Determinism.
    assert [id(s) for s in train_a] == [id(s) for s in train_b]
    assert [id(s) for s in val_a] == [id(s) for s in val_b]

    # No audio source appears in both splits.
    train_ids = {s.audio_file_id for s in train_a}
    val_ids = {s.audio_file_id for s in val_a}
    assert not train_ids & val_ids
    assert len(train_a) + len(val_a) == 6
    assert len(val_a) == 2  # 1 of 3 groups → 2 samples


def test_split_by_audio_source_single_source_temporal_fallback() -> None:
    samples = [
        _FakeSample(
            events_json="[]",
            crop_start_sec=0.0,
            crop_end_sec=1.0,
            hydrophone_id="only",
            start_timestamp=float(i * 100),
            end_timestamp=float(i * 100 + 10),
        )
        for i in range(10)
    ]
    train, val = split_by_audio_source(samples, val_fraction=0.2, seed=0)
    assert len(train) == 8
    assert len(val) == 2
    # Val should be the last two by timestamp
    assert all(
        s.start_timestamp is not None and s.start_timestamp >= 800.0 for s in val
    )
    # Train should be the first eight
    assert all(
        s.start_timestamp is not None and s.start_timestamp < 800.0 for s in train
    )


def test_split_by_audio_source_single_source_val_zero_no_split() -> None:
    samples = [
        _FakeSample(
            events_json="[]",
            crop_start_sec=0.0,
            crop_end_sec=1.0,
            audio_file_id="only",
        )
        for _ in range(4)
    ]
    train, val = split_by_audio_source(samples, val_fraction=0.0, seed=0)
    assert len(train) == 4
    assert val == []


def test_split_by_audio_source_hydrophone_key_falls_back() -> None:
    samples = [
        _FakeSample(
            events_json="[]",
            crop_start_sec=0.0,
            crop_end_sec=1.0,
            hydrophone_id=f"hydro-{i // 2}",
        )
        for i in range(4)
    ]
    train, val = split_by_audio_source(samples, val_fraction=0.5, seed=7)
    train_ids = {s.hydrophone_id for s in train}
    val_ids = {s.hydrophone_id for s in val}
    assert not train_ids & val_ids
    assert len(train) + len(val) == 4


def test_split_by_audio_source_rejects_sample_without_source() -> None:
    samples = [_FakeSample(events_json="[]", crop_start_sec=0.0, crop_end_sec=1.0)]
    with pytest.raises(ValueError):
        split_by_audio_source(samples, val_fraction=0.2, seed=0)


# ---- match_events_by_iou ------------------------------------------------


def test_match_events_by_iou_empty_predictions() -> None:
    gts = [_make_event(0.0, 1.0), _make_event(2.0, 3.0)]
    result = match_events_by_iou([], gts, iou_threshold=0.3)
    assert result.hits == []
    assert result.misses == gts
    assert result.extras == []
    assert result.onset_errors == []
    assert result.offset_errors == []


def test_match_events_by_iou_empty_ground_truth() -> None:
    preds = [_make_event(0.0, 1.0)]
    result = match_events_by_iou(preds, [], iou_threshold=0.3)
    assert result.hits == []
    assert result.misses == []
    assert result.extras == preds


def test_match_events_by_iou_perfect_match_zero_errors() -> None:
    gt = _make_event(1.0, 2.0)
    pred = _make_event(1.0, 2.0, confidence=0.9)
    result = match_events_by_iou([pred], [gt], iou_threshold=0.3)
    assert len(result.hits) == 1
    assert result.misses == []
    assert result.extras == []
    assert result.onset_errors == [0.0]
    assert result.offset_errors == [0.0]


def test_match_events_by_iou_partial_match_tracks_hits_misses_extras() -> None:
    gts = [
        _make_event(0.0, 1.0),  # will match pred[0]
        _make_event(5.0, 6.0),  # no matching pred → miss
    ]
    preds = [
        _make_event(0.1, 0.9),  # high IoU with gts[0]
        _make_event(10.0, 11.0),  # no matching gt → extra
    ]
    result = match_events_by_iou(preds, gts, iou_threshold=0.3)
    assert len(result.hits) == 1
    assert len(result.misses) == 1
    assert len(result.extras) == 1
    assert result.onset_errors[0] == pytest.approx(0.1)
    assert result.offset_errors[0] == pytest.approx(0.1)


def test_match_events_by_iou_one_gt_to_two_preds_only_one_matches() -> None:
    gt = _make_event(0.0, 1.0)
    preds = [
        _make_event(0.05, 0.95),  # IoU ≈ 0.9 with gt
        _make_event(0.0, 1.0),  # IoU = 1.0 with gt — should win
    ]
    result = match_events_by_iou(preds, [gt], iou_threshold=0.3)
    assert len(result.hits) == 1
    assert len(result.extras) == 1
    # The best-IoU prediction (exact match) wins.
    assert result.hits[0][0] is preds[1]


# ---- MaskedBCEWithLogitsLoss -------------------------------------------


def test_masked_bce_ignores_padded_frames() -> None:
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=1.0)
    outputs = torch.zeros(1, 4)
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask_full = torch.ones(1, 4, dtype=torch.bool)
    mask_half = torch.tensor([[True, True, False, False]])
    full_loss = loss_fn(outputs, target, mask_full)
    half_loss = loss_fn(outputs, target, mask_half)
    # Both averages evaluate to BCE(0, y)=log(2) for each unmasked frame.
    assert full_loss == pytest.approx(half_loss, rel=1e-6)


# ---- train_model smoke --------------------------------------------------


def _sine_audio(freq: float, duration_sec: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(duration_sec * sample_rate), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _noise_audio(duration_sec: float, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.05, size=int(duration_sec * sample_rate)).astype(
        np.float32
    )


def test_train_model_smoke_converges(tmp_path: Path) -> None:
    os.environ["HUMPBACK_FORCE_CPU"] = "1"
    torch.manual_seed(0)

    cfg = SegmentationFeatureConfig()
    sample_rate = cfg.sample_rate
    duration = 1.0

    # Eight synthetic samples across four audio sources. Positive samples
    # contain a short 440 Hz tone burst at the labeled event window; the
    # event bounds cover the whole crop so the framewise target is all
    # ones, which gives a clean signal for a tiny model to latch onto.
    samples: list[_FakeSample] = []
    audio_by_id: dict[str, np.ndarray] = {}
    for i in range(4):
        pos_id = f"pos-{i}"
        neg_id = f"neg-{i}"
        tone = _sine_audio(440.0, duration, sample_rate)
        noise = _noise_audio(duration, sample_rate, seed=100 + i)
        audio_by_id[pos_id] = tone.astype(np.float32)
        audio_by_id[neg_id] = noise.astype(np.float32)
        samples.append(
            _FakeSample(
                events_json=json.dumps([{"start_sec": 0.0, "end_sec": duration}]),
                crop_start_sec=0.0,
                crop_end_sec=duration,
                audio_file_id=pos_id,
            )
        )
        samples.append(
            _FakeSample(
                events_json="[]",
                crop_start_sec=0.0,
                crop_end_sec=duration,
                audio_file_id=neg_id,
            )
        )

    def audio_loader(sample: _FakeSample) -> np.ndarray:
        assert sample.audio_file_id is not None
        return audio_by_id[sample.audio_file_id]

    # Small model + short run. Use one conv block and one GRU layer so
    # the smoke test stays under a second on CPU.
    config = SegmentationTrainingConfig(
        epochs=3,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        early_stopping_patience=100,
        grad_clip=1.0,
        seed=0,
        val_fraction=0.25,
        n_mels=cfg.n_mels,
        conv_channels=[8],
        gru_hidden=8,
        gru_layers=1,
    )

    checkpoint_path = tmp_path / "seg_model" / "checkpoint.pt"
    result = train_model(
        samples=samples,
        feature_config=cfg,
        decoder_config=SegmentationDecoderConfig(),
        audio_loader=audio_loader,
        config=config,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert len(result.train_losses) == 3
    assert result.train_losses[-1] < result.train_losses[0]
    # Final eval metrics are populated fields.
    assert 0.0 <= result.framewise_precision <= 1.0
    assert 0.0 <= result.framewise_recall <= 1.0
    assert 0.0 <= result.framewise_f1 <= 1.0
    assert 0.0 <= result.event_f1 <= 1.0
    assert result.pos_weight > 0
    assert result.n_train_samples > 0
    assert result.n_val_samples > 0
    assert checkpoint_path.exists()
    # Checkpoint loads back into a fresh model.
    restored = SegmentationCRNN(
        n_mels=config.n_mels,
        conv_channels=config.conv_channels,
        gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers,
    )
    cfg_meta = load_checkpoint(checkpoint_path, restored)
    assert cfg_meta["model_type"] == "SegmentationCRNN"


def test_train_model_finetune_from_pretrained(tmp_path: Path) -> None:
    """train_model with pretrained_checkpoint initializes from existing weights."""
    os.environ["HUMPBACK_FORCE_CPU"] = "1"
    torch.manual_seed(0)

    cfg = SegmentationFeatureConfig()
    sample_rate = cfg.sample_rate
    duration = 1.0

    samples: list[_FakeSample] = []
    audio_by_id: dict[str, np.ndarray] = {}
    for i in range(4):
        pos_id = f"pos-{i}"
        neg_id = f"neg-{i}"
        audio_by_id[pos_id] = _sine_audio(440.0, duration, sample_rate)
        audio_by_id[neg_id] = _noise_audio(duration, sample_rate, seed=100 + i)
        samples.append(
            _FakeSample(
                events_json=json.dumps([{"start_sec": 0.0, "end_sec": duration}]),
                crop_start_sec=0.0,
                crop_end_sec=duration,
                audio_file_id=pos_id,
            )
        )
        samples.append(
            _FakeSample(
                events_json="[]",
                crop_start_sec=0.0,
                crop_end_sec=duration,
                audio_file_id=neg_id,
            )
        )

    def audio_loader(sample: _FakeSample) -> np.ndarray:
        assert sample.audio_file_id is not None
        return audio_by_id[sample.audio_file_id]

    small_config = SegmentationTrainingConfig(
        epochs=3,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        early_stopping_patience=100,
        grad_clip=1.0,
        seed=0,
        val_fraction=0.25,
        n_mels=cfg.n_mels,
        conv_channels=[8],
        gru_hidden=8,
        gru_layers=1,
    )

    # Train an initial model.
    pretrained_path = tmp_path / "pretrained" / "checkpoint.pt"
    train_model(
        samples=samples,
        feature_config=cfg,
        decoder_config=SegmentationDecoderConfig(),
        audio_loader=audio_loader,
        config=small_config,
        checkpoint_path=pretrained_path,
        device=torch.device("cpu"),
    )
    assert pretrained_path.exists()

    # Fine-tune from the pretrained checkpoint with a fresh seed.
    finetuned_path = tmp_path / "finetuned" / "checkpoint.pt"
    torch.manual_seed(99)
    ft_result = train_model(
        samples=samples,
        feature_config=cfg,
        decoder_config=SegmentationDecoderConfig(),
        audio_loader=audio_loader,
        config=small_config,
        checkpoint_path=finetuned_path,
        device=torch.device("cpu"),
        pretrained_checkpoint=pretrained_path,
    )

    assert finetuned_path.exists()
    # Fine-tuned model should start with a lower initial loss than
    # training from scratch, since it begins from learned weights.
    torch.manual_seed(99)
    scratch_path = tmp_path / "scratch" / "checkpoint.pt"
    scratch_result = train_model(
        samples=samples,
        feature_config=cfg,
        decoder_config=SegmentationDecoderConfig(),
        audio_loader=audio_loader,
        config=small_config,
        checkpoint_path=scratch_path,
        device=torch.device("cpu"),
    )
    assert ft_result.train_losses[0] < scratch_result.train_losses[0]

    # Conv layers should be frozen during fine-tuning: the saved
    # checkpoint's conv weights must match the pretrained ones exactly.
    pretrained_model = SegmentationCRNN(
        n_mels=small_config.n_mels,
        conv_channels=small_config.conv_channels,
        gru_hidden=small_config.gru_hidden,
        gru_layers=small_config.gru_layers,
    )
    load_checkpoint(pretrained_path, pretrained_model)
    finetuned_model = SegmentationCRNN(
        n_mels=small_config.n_mels,
        conv_channels=small_config.conv_channels,
        gru_hidden=small_config.gru_hidden,
        gru_layers=small_config.gru_layers,
    )
    load_checkpoint(finetuned_path, finetuned_model)
    for name, param in finetuned_model.conv.named_parameters():
        pretrained_param = dict(pretrained_model.conv.named_parameters())[name]
        assert torch.equal(param, pretrained_param), f"conv param {name} changed"


# ---- run_inference ------------------------------------------------------


def test_run_inference_deterministic_under_fixed_seed() -> None:
    os.environ["HUMPBACK_FORCE_CPU"] = "1"
    torch.manual_seed(7)
    model = SegmentationCRNN(
        n_mels=64,
        conv_channels=[8],
        gru_hidden=8,
        gru_layers=1,
    )
    model.eval()

    feature_cfg = SegmentationFeatureConfig()
    decoder_cfg = SegmentationDecoderConfig(
        high_threshold=0.5, low_threshold=0.3, min_event_sec=0.0, merge_gap_sec=0.0
    )
    region = _FakeRegion(region_id="reg-1", padded_start_sec=10.0, padded_end_sec=12.0)
    audio = _sine_audio(440.0, 2.0, feature_cfg.sample_rate)

    def loader(_r: _FakeRegion) -> np.ndarray:
        return audio

    cpu = torch.device("cpu")
    events_a = run_inference(model, region, loader, feature_cfg, decoder_cfg, cpu)
    events_b = run_inference(model, region, loader, feature_cfg, decoder_cfg, cpu)
    assert len(events_a) == len(events_b)
    for a, b in zip(events_a, events_b):
        assert a.start_sec == pytest.approx(b.start_sec)
        assert a.end_sec == pytest.approx(b.end_sec)
        assert a.segmentation_confidence == pytest.approx(b.segmentation_confidence)
        assert a.region_id == b.region_id == "reg-1"
        # Events are anchored to ``padded_start_sec=10.0``. librosa's
        # ``center=True`` produces one extra frame beyond the raw
        # ``audio_len / hop`` count, so allow a single-hop slop at the
        # upper bound.
        hop_sec = feature_cfg.hop_length / feature_cfg.sample_rate
        assert a.start_sec >= 10.0
        assert a.end_sec <= 12.0 + 2 * hop_sec


def test_run_inference_windowed_on_long_region() -> None:
    """Regions longer than 30s use windowed inference."""
    os.environ["HUMPBACK_FORCE_CPU"] = "1"
    torch.manual_seed(7)
    model = SegmentationCRNN(
        n_mels=64,
        conv_channels=[8],
        gru_hidden=8,
        gru_layers=1,
    )
    model.eval()

    feature_cfg = SegmentationFeatureConfig()
    decoder_cfg = SegmentationDecoderConfig(
        high_threshold=0.5, low_threshold=0.3, min_event_sec=0.0, merge_gap_sec=0.0
    )
    # 60s region triggers windowed inference (>30s)
    region = _FakeRegion(
        region_id="reg-long", padded_start_sec=0.0, padded_end_sec=60.0
    )
    audio = _sine_audio(440.0, 60.0, feature_cfg.sample_rate)

    def loader(_r: _FakeRegion) -> np.ndarray:
        return audio

    cpu = torch.device("cpu")
    events = run_inference(model, region, loader, feature_cfg, decoder_cfg, cpu)
    # Windowed inference should produce events (exact count depends on
    # model weights) and all events should fall within region bounds.
    hop_sec = feature_cfg.hop_length / feature_cfg.sample_rate
    for e in events:
        assert e.start_sec >= 0.0
        assert e.end_sec <= 60.0 + 2 * hop_sec
        assert e.region_id == "reg-long"

    # Deterministic: second run should match.
    events2 = run_inference(model, region, loader, feature_cfg, decoder_cfg, cpu)
    assert len(events) == len(events2)
    for a, b in zip(events, events2):
        assert a.start_sec == pytest.approx(b.start_sec)
        assert a.end_sec == pytest.approx(b.end_sec)
