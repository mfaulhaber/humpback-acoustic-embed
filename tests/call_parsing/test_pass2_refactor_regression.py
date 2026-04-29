"""Pass 2 inference refactor regression test.

Asserts that ``_infer_windowed`` (now consuming
``iter_inference_windows``) produces byte-identical frame-probability
output to a frozen replica of the pre-refactor windowing loop. If a
future change alters the windowing math, sliding-window stitching, or
edge-handling, this test will fail clearly.

Also runs ``run_inference`` end-to-end against a deterministic stub
``nn.Module`` that mimics the SegmentationCRNN frame-head output. The
event-decoding output (number of events, their start/end seconds) is
compared against the same frozen replica's decoded events.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from humpback.call_parsing.segmentation import inference as inference_mod
from humpback.call_parsing.segmentation.decoder import decode_events
from humpback.call_parsing.segmentation.features import (
    extract_logmel,
    normalize_per_region_zscore,
)
from humpback.call_parsing.segmentation.inference import (
    _infer_single,
    _infer_windowed,
    run_inference,
)
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
)


class _DeterministicCRNN(nn.Module):
    """Per-frame logit head: ``logit = mean(features over freq+channel) - bias``.

    Pure-tensor and stateless so two forward passes over the same input
    produce bit-identical output. Output shape ``(batch, frames)``
    matches what ``_infer_single`` expects after squeeze.
    """

    def __init__(self, bias: float = 0.0) -> None:
        super().__init__()
        self.bias_value: float = bias

    def forward(self, features_t: torch.Tensor) -> torch.Tensor:
        # features_t: (B, 1, n_mels, frames)
        # Mean across channels and mels → (B, frames)
        return features_t.mean(dim=(1, 2)) - self.bias_value


def _frozen_infer_windowed(
    model: nn.Module,
    audio: np.ndarray,
    feature_config: SegmentationFeatureConfig,
    device: torch.device,
    max_window_sec: float = 30.0,
    window_hop_sec: float = 15.0,
) -> np.ndarray:
    """Replica of the pre-refactor ``_infer_windowed`` body.

    Kept verbatim to anchor the regression assertion.
    """
    sr = feature_config.sample_rate
    hop_length = feature_config.hop_length
    total_samples = len(audio)
    total_duration = total_samples / sr

    if total_duration <= max_window_sec:
        return _frozen_infer_single(model, audio, feature_config, device)

    total_frames = 1 + total_samples // hop_length
    prob_sum = np.zeros(total_frames, dtype=np.float64)
    weight = np.zeros(total_frames, dtype=np.float64)

    window_samples = int(max_window_sec * sr)
    hop_samples = int(window_hop_sec * sr)

    offset = 0
    while offset < total_samples:
        end = min(offset + window_samples, total_samples)
        if total_samples - offset < window_samples // 2 and offset > 0:
            offset = max(0, total_samples - window_samples)
            end = total_samples

        chunk = audio[offset:end]
        chunk_probs = _frozen_infer_single(model, chunk, feature_config, device)

        frame_offset = int(np.round(offset / hop_length))
        n_frames = len(chunk_probs)
        frame_end = min(frame_offset + n_frames, total_frames)
        usable = frame_end - frame_offset
        prob_sum[frame_offset:frame_end] += chunk_probs[:usable]
        weight[frame_offset:frame_end] += 1.0

        if end >= total_samples:
            break
        offset += hop_samples

    weight = np.maximum(weight, 1.0)
    return (prob_sum / weight).astype(np.float32)


def _frozen_infer_single(
    model: nn.Module,
    audio: np.ndarray,
    feature_config: SegmentationFeatureConfig,
    device: torch.device,
) -> np.ndarray:
    logmel = normalize_per_region_zscore(extract_logmel(audio, feature_config))
    features_t = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(features_t)
        probs_t = torch.sigmoid(logits).squeeze(0)
    return probs_t.detach().cpu().numpy().astype(np.float32)


def _make_synthetic_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Deterministic non-trivial waveform so windowing produces varied probs."""
    n = int(duration_sec * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    return (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.cos(2 * np.pi * 110 * t)
    ).astype(np.float32)


def test_infer_single_matches_frozen_reference():
    """Single-pass output is unchanged by the refactor."""
    feature_config = SegmentationFeatureConfig()
    model = _DeterministicCRNN(bias=0.5).eval()
    device = torch.device("cpu")
    audio = _make_synthetic_audio(10.0, feature_config.sample_rate)

    new_probs = _infer_single(model, audio, feature_config, device)
    ref_probs = _frozen_infer_single(model, audio, feature_config, device)

    np.testing.assert_array_equal(new_probs, ref_probs)


def test_infer_windowed_short_region_matches_frozen_reference():
    """Region shorter than one window: short-circuit path is unchanged."""
    feature_config = SegmentationFeatureConfig()
    model = _DeterministicCRNN(bias=0.25).eval()
    device = torch.device("cpu")
    audio = _make_synthetic_audio(20.0, feature_config.sample_rate)

    new_probs = _infer_windowed(model, audio, feature_config, device)
    ref_probs = _frozen_infer_windowed(model, audio, feature_config, device)

    np.testing.assert_array_equal(new_probs, ref_probs)


def test_infer_windowed_long_region_matches_frozen_reference():
    """Multi-window region with averaging across overlaps is byte-identical."""
    feature_config = SegmentationFeatureConfig()
    model = _DeterministicCRNN(bias=-0.1).eval()
    device = torch.device("cpu")
    audio = _make_synthetic_audio(80.0, feature_config.sample_rate)

    new_probs = _infer_windowed(model, audio, feature_config, device)
    ref_probs = _frozen_infer_windowed(model, audio, feature_config, device)

    assert new_probs.shape == ref_probs.shape
    np.testing.assert_array_equal(new_probs, ref_probs)


def test_infer_windowed_pullback_path_matches_frozen_reference():
    """Trigger the tail pull-back branch with a non-default window/hop pair."""
    feature_config = SegmentationFeatureConfig()
    model = _DeterministicCRNN(bias=0.0).eval()
    device = torch.device("cpu")
    audio = _make_synthetic_audio(20.0, feature_config.sample_rate)

    # window=10 s, hop=8 s triggers the pull-back path (see test_window_iter).
    new_probs = _infer_windowed(
        model,
        audio,
        feature_config,
        device,
        max_window_sec=10.0,
        window_hop_sec=8.0,
    )
    ref_probs = _frozen_infer_windowed(
        model, audio, feature_config, device, max_window_sec=10.0, window_hop_sec=8.0
    )

    np.testing.assert_array_equal(new_probs, ref_probs)


def test_run_inference_end_to_end_matches_frozen_reference():
    """``run_inference``'s decoded events are unchanged by the refactor."""

    class _Region:
        region_id = "region-0"
        padded_start_sec = 100.0

    feature_config = SegmentationFeatureConfig()
    decoder_config = SegmentationDecoderConfig()
    model = _DeterministicCRNN(bias=0.0).eval()
    device = torch.device("cpu")
    audio = _make_synthetic_audio(80.0, feature_config.sample_rate)

    new_events = run_inference(
        model,
        _Region(),
        audio_loader=lambda _r: audio,
        feature_config=feature_config,
        decoder_config=decoder_config,
        device=device,
    )

    ref_probs = _frozen_infer_windowed(model, audio, feature_config, device)
    hop_sec = float(feature_config.hop_length) / float(feature_config.sample_rate)
    ref_events = decode_events(
        frame_probs=ref_probs,
        region_id="region-0",
        region_start_sec=100.0,
        hop_sec=hop_sec,
        config=decoder_config,
    )

    assert len(new_events) == len(ref_events)
    for a, b in zip(new_events, ref_events):
        assert a.region_id == b.region_id
        assert a.start_sec == b.start_sec
        assert a.end_sec == b.end_sec
        assert a.center_sec == b.center_sec
        assert a.segmentation_confidence == b.segmentation_confidence


def test_inference_module_exposes_constants_for_back_compat():
    """The pre-refactor module-level constants are still importable from
    ``inference`` so any external consumer keeps working.
    """
    assert inference_mod._MAX_WINDOW_SEC == 30.0
    assert inference_mod._WINDOW_HOP_SEC == 15.0
