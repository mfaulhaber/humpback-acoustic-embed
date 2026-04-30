"""Tests for the CRNN region chunk-embedding extractor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.ml.checkpointing import save_checkpoint
from humpback.schemas.call_parsing import SegmentationFeatureConfig
from humpback.sequence_models.chunk_projection import IdentityProjection
from humpback.sequence_models.crnn_features import (
    EXPECTED_BIGRU_WIDTH,
    ChunkEmbeddingResult,
    _iter_chunk_slices,
    _stitch_centre_half,
    compute_checkpoint_sha256,
    extract_chunk_embeddings,
    load_crnn_for_extraction,
)


def _make_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    n = int(duration_sec * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    return (
        0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    ).astype(np.float32)


def _save_test_checkpoint(
    tmp_path,
    *,
    n_mels: int = 64,
    gru_hidden: int = 64,
    gru_layers: int = 2,
    sample_rate: int = 16000,
    hop_length: int = 512,
):
    feature_config = {
        "sample_rate": sample_rate,
        "n_fft": 1024,
        "hop_length": hop_length,
        "n_mels": n_mels,
    }
    model = SegmentationCRNN(
        n_mels=n_mels, gru_hidden=gru_hidden, gru_layers=gru_layers
    )
    path = tmp_path / "stub_crnn.pt"
    save_checkpoint(
        path,
        model,
        optimizer=None,
        config={
            "n_mels": n_mels,
            "conv_channels": [32, 64, 96, 128],
            "gru_hidden": gru_hidden,
            "gru_layers": gru_layers,
            "feature_config": feature_config,
        },
    )
    return path


# ---------- helper-function tests ----------


def test_iter_chunk_slices_no_truncation():
    out = list(_iter_chunk_slices(n_frames=20, frames_per_chunk=8, frame_hop=8))
    assert out == [(0, 8), (8, 16)]  # 16..20 truncated


def test_iter_chunk_slices_overlapping_hop():
    out = list(_iter_chunk_slices(n_frames=20, frames_per_chunk=8, frame_hop=4))
    assert out == [(0, 8), (4, 12), (8, 16), (12, 20)]


def test_iter_chunk_slices_too_short():
    out = list(_iter_chunk_slices(n_frames=5, frames_per_chunk=8, frame_hop=8))
    assert out == []


def test_stitch_centre_half_two_windows_overlap_split_at_midpoint():
    """Window A covers frames [0,10), B covers [6,16). Boundary midpoint
    of overlap is (6+10)//2 = 8, so A owns [0,8) and B owns [8,16)."""
    a = np.arange(10, dtype=np.float32).reshape(10, 1)
    b = np.arange(100, 110, dtype=np.float32).reshape(10, 1)
    out = _stitch_centre_half([(0, a), (6, b)], total_frames=16)
    expected = np.concatenate([a[:8], b[2:10]], axis=0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_stitch_centre_half_single_window():
    a = np.arange(8, dtype=np.float32).reshape(8, 1)
    out = _stitch_centre_half([(0, a)], total_frames=8)
    np.testing.assert_array_equal(out, a)


# ---------- extractor tests ----------


@pytest.fixture
def stub_checkpoint(tmp_path):
    return _save_test_checkpoint(tmp_path)


def test_load_crnn_for_extraction_validates_and_returns_sha(stub_checkpoint):
    loaded = load_crnn_for_extraction(stub_checkpoint, torch.device("cpu"))
    assert loaded.checkpoint_sha256 == compute_checkpoint_sha256(stub_checkpoint)
    assert 2 * loaded.model.gru_hidden == EXPECTED_BIGRU_WIDTH
    assert loaded.feature_config.sample_rate == 16000
    assert loaded.feature_config.hop_length == 512


def test_load_crnn_rejects_wrong_bigru_width(tmp_path):
    bad = _save_test_checkpoint(tmp_path, gru_hidden=32)
    with pytest.raises(ValueError, match="BiGRU output width"):
        load_crnn_for_extraction(bad, torch.device("cpu"))


def test_load_crnn_rejects_wrong_frame_rate(tmp_path):
    bad = _save_test_checkpoint(tmp_path, hop_length=160)  # 100 fps, far from 32
    with pytest.raises(ValueError, match="frame rate"):
        load_crnn_for_extraction(bad, torch.device("cpu"))


def test_extract_chunk_embeddings_shape_and_timestamps(stub_checkpoint):
    """30 s region, chunk=250 ms, hop=250 ms (no overlap) → 8-frame chunks
    at 32 ms frame-period (31.25 fps) → ~8 frames per 250 ms.
    """
    loaded = load_crnn_for_extraction(stub_checkpoint, torch.device("cpu"))
    audio = _make_audio(20.0, loaded.feature_config.sample_rate)
    projection = IdentityProjection(input_dim=8 * EXPECTED_BIGRU_WIDTH)

    result = extract_chunk_embeddings(
        model=loaded.model,
        audio=audio,
        feature_config=loaded.feature_config,
        chunk_size_seconds=0.250,
        chunk_hop_seconds=0.250,
        projection=projection,
        device=torch.device("cpu"),
    )

    assert isinstance(result, ChunkEmbeddingResult)
    assert result.embeddings.shape[1] == 8 * EXPECTED_BIGRU_WIDTH
    # ~31.25 fps × 20s ≈ 625 frames ÷ 8 frames/chunk ≈ 78 chunks.
    assert 70 <= result.embeddings.shape[0] <= 85
    # Per-chunk start timestamps strictly increase.
    assert np.all(np.diff(result.chunk_starts) > 0)
    # Each chunk spans exactly ``frames_per_chunk * frame_period``. With
    # 31.25 fps and 250 ms request, frames_per_chunk rounds to 8 → span
    # is 8 / 31.25 = 0.256 s (~6 ms more than the requested 250 ms).
    spans = result.chunk_ends - result.chunk_starts
    assert np.allclose(spans, spans[0])
    assert abs(spans[0] - 0.250) < 0.010
    # Probabilities are mean of 8 sigmoid frames → in (0, 1).
    assert np.all(result.call_probabilities > 0.0)
    assert np.all(result.call_probabilities < 1.0)


def test_extract_chunk_embeddings_doubles_count_at_125ms_hop(stub_checkpoint):
    loaded = load_crnn_for_extraction(stub_checkpoint, torch.device("cpu"))
    audio = _make_audio(10.0, loaded.feature_config.sample_rate)
    projection = IdentityProjection(input_dim=8 * EXPECTED_BIGRU_WIDTH)

    no_overlap = extract_chunk_embeddings(
        model=loaded.model,
        audio=audio,
        feature_config=loaded.feature_config,
        chunk_size_seconds=0.250,
        chunk_hop_seconds=0.250,
        projection=projection,
        device=torch.device("cpu"),
    )
    half_overlap = extract_chunk_embeddings(
        model=loaded.model,
        audio=audio,
        feature_config=loaded.feature_config,
        chunk_size_seconds=0.250,
        chunk_hop_seconds=0.125,
        projection=projection,
        device=torch.device("cpu"),
    )
    # Half-hop should produce roughly 2× chunks.
    ratio = half_overlap.embeddings.shape[0] / no_overlap.embeddings.shape[0]
    assert 1.7 < ratio < 2.2


def test_extract_chunk_embeddings_runs_windowed_for_long_audio(stub_checkpoint):
    """40 s region → triggers windowed inference (>30 s threshold)."""
    loaded = load_crnn_for_extraction(stub_checkpoint, torch.device("cpu"))
    audio = _make_audio(40.0, loaded.feature_config.sample_rate)
    projection = IdentityProjection(input_dim=8 * EXPECTED_BIGRU_WIDTH)

    result = extract_chunk_embeddings(
        model=loaded.model,
        audio=audio,
        feature_config=loaded.feature_config,
        chunk_size_seconds=0.250,
        chunk_hop_seconds=0.250,
        projection=projection,
        device=torch.device("cpu"),
    )
    # Should still produce sensible chunks.
    assert result.embeddings.shape[0] > 100
    assert result.embeddings.shape[1] == 8 * EXPECTED_BIGRU_WIDTH


def test_extract_chunk_embeddings_validates_at_runtime():
    feature_config = SegmentationFeatureConfig()
    bad_model = SegmentationCRNN(gru_hidden=32)  # wrong BiGRU width
    bad_model.eval()
    projection = IdentityProjection(input_dim=8 * EXPECTED_BIGRU_WIDTH)
    with pytest.raises(ValueError, match="BiGRU output width"):
        extract_chunk_embeddings(
            model=bad_model,
            audio=_make_audio(5.0),
            feature_config=feature_config,
            chunk_size_seconds=0.250,
            chunk_hop_seconds=0.250,
            projection=projection,
            device=torch.device("cpu"),
        )


def test_extract_chunk_embeddings_empty_audio_returns_empty(stub_checkpoint):
    loaded = load_crnn_for_extraction(stub_checkpoint, torch.device("cpu"))
    projection = IdentityProjection(input_dim=8 * EXPECTED_BIGRU_WIDTH)
    result = extract_chunk_embeddings(
        model=loaded.model,
        audio=np.zeros(0, dtype=np.float32),
        feature_config=loaded.feature_config,
        chunk_size_seconds=0.250,
        chunk_hop_seconds=0.250,
        projection=projection,
        device=torch.device("cpu"),
    )
    assert result.embeddings.shape == (0, 8 * EXPECTED_BIGRU_WIDTH)
    assert result.call_probabilities.shape == (0,)


def test_compute_checkpoint_sha256_is_stable(stub_checkpoint):
    a = compute_checkpoint_sha256(stub_checkpoint)
    b = compute_checkpoint_sha256(stub_checkpoint)
    assert a == b
    assert len(a) == 64  # hex digest
