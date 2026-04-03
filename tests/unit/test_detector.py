"""Tests for embed_audio_folder and detection pipeline with FakeTFLiteModel."""

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from humpback.classifier.detector import (
    match_embedding_records_to_row_store,
    read_detection_embedding,
    run_detection,
    write_detection_embeddings,
)
from humpback.classifier.trainer import embed_audio_folder, train_binary_classifier
from humpback.processing.inference import FakeTFLiteModel


def _write_wav(path: Path, duration: float = 2.0, sample_rate: int = 16000):
    """Write a simple sine wave WAV file."""
    import math

    n_samples = int(sample_rate * duration)
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(n_samples)
    ]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def test_embed_audio_folder(tmp_path):
    """embed_audio_folder produces correct shape with FakeTFLiteModel."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create 2 WAV files, each ~6 seconds (≥ 5s window)
    _write_wav(audio_dir / "a.wav", duration=6.0)
    _write_wav(audio_dir / "b.wav", duration=6.0)

    model = FakeTFLiteModel(vector_dim=128)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        input_format="spectrogram",
    )

    # Each 6s file → 2 windows (second is overlapped)
    assert result.ndim == 2
    assert result.shape[0] == 4  # 2 files, 2 windows each
    assert result.shape[1] == 128


def test_embed_audio_folder_no_files(tmp_path):
    """Empty folder raises ValueError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    model = FakeTFLiteModel(vector_dim=128)
    with pytest.raises(ValueError, match="No audio files"):
        embed_audio_folder(
            folder=empty_dir,
            model=model,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )


def test_embed_audio_folder_recursive(tmp_path):
    """Finds audio files in subdirectories."""
    audio_dir = tmp_path / "audio"
    sub_dir = audio_dir / "subdir"
    sub_dir.mkdir(parents=True)

    _write_wav(audio_dir / "a.wav", duration=6.0)
    _write_wav(sub_dir / "b.wav", duration=6.0)

    model = FakeTFLiteModel(vector_dim=64)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )

    assert result.shape[0] == 4  # 2 files × 2 windows each
    assert result.shape[1] == 64


def test_embed_audio_folder_longer_file(tmp_path):
    """Longer file produces multiple windows."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # 12 seconds → 3 windows at 5s each (last one overlapped)
    _write_wav(audio_dir / "long.wav", duration=12.0)

    model = FakeTFLiteModel(vector_dim=32)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )

    assert result.shape[0] == 3
    assert result.shape[1] == 32


def test_embed_audio_folder_short_files_skipped(tmp_path):
    """Files shorter than one window are skipped; raises if all are short."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # 2s files with 5s window → all skipped
    _write_wav(audio_dir / "a.wav", duration=2.0)
    _write_wav(audio_dir / "b.wav", duration=1.0)

    model = FakeTFLiteModel(vector_dim=64)
    with pytest.raises(ValueError, match="No embeddings produced"):
        embed_audio_folder(
            folder=audio_dir,
            model=model,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )


def test_embed_audio_folder_mixed_short_and_long(tmp_path):
    """Short files are skipped but long files are still processed."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    _write_wav(audio_dir / "short.wav", duration=2.0)  # skipped
    _write_wav(audio_dir / "long.wav", duration=10.0)  # 2 windows

    model = FakeTFLiteModel(vector_dim=64)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )

    assert result.shape[0] == 2
    assert result.shape[1] == 64


def test_confidence_stats_in_summary(tmp_path):
    """run_detection summary includes confidence_stats with expected keys."""
    # Create detection audio
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=12.0)

    # Train a simple classifier on fake embeddings
    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    detections, summary, _, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
    )

    assert "confidence_stats" in summary
    stats = summary["confidence_stats"]
    expected_keys = {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p10",
        "p25",
        "p75",
        "p90",
        "pct_above_threshold",
    }
    assert expected_keys == set(stats.keys())
    assert 0.0 <= stats["pct_above_threshold"] <= 1.0
    assert stats["min"] <= stats["mean"] <= stats["max"]

    # New summary keys
    assert "hop_seconds" in summary
    assert "high_threshold" in summary
    assert "low_threshold" in summary


def test_run_detection_with_hop(tmp_path):
    """With hop < window, more windows are produced per file."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=12.0)

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    _, summary_no_hop, _, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=5.0,
    )

    _, summary_hop, _, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=1.0,
    )

    assert summary_hop["n_windows"] > summary_no_hop["n_windows"]


def test_run_detection_on_file_complete_callback(tmp_path):
    """on_file_complete callback is invoked once per audio file with correct progress."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=6.0)
    _write_wav(audio_dir / "b.wav", duration=6.0)
    _write_wav(audio_dir / "c.wav", duration=1.0)  # too short, will be skipped

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    calls = []

    def on_file_complete(file_detections, files_done, files_total):
        calls.append(
            {
                "detections": list(file_detections),
                "files_done": files_done,
                "files_total": files_total,
            }
        )

    run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        on_file_complete=on_file_complete,
    )

    # Should be called 3 times (a.wav, b.wav, c.wav including skipped)
    assert len(calls) == 3
    # files_total is always 3
    assert all(c["files_total"] == 3 for c in calls)
    # files_done increments
    assert [c["files_done"] for c in calls] == [1, 2, 3]
    # The skipped file (c.wav) should have empty detections
    assert calls[2]["detections"] == []


def test_run_detection_hysteresis(tmp_path):
    """Hysteresis thresholds produce different event boundaries than single threshold."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=30.0)

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    dets_single, _, _, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=5.0,
        high_threshold=0.5,
        low_threshold=0.5,
    )

    dets_hysteresis, _, _, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=5.0,
        high_threshold=0.7,
        low_threshold=0.3,
    )

    # With higher start threshold, we should get fewer or equal events
    assert len(dets_hysteresis) <= len(dets_single)

    # All detections should have n_windows
    for det in dets_single + dets_hysteresis:
        assert "n_windows" in det
        assert det["n_windows"] >= 1


# ---------------------------------------------------------------------------
# Detection embedding storage (write / read round-trip)
# ---------------------------------------------------------------------------


class TestDetectionEmbeddingStorage:
    def test_write_read_roundtrip(self, tmp_path):
        """write_detection_embeddings + read_detection_embedding round-trip."""
        records = [
            {
                "row_id": "row-1",
                "embedding": [1.0, 2.0, 3.0, 4.0],
                "confidence": 0.85,
            },
            {
                "row_id": "row-2",
                "embedding": [5.0, 6.0, 7.0, 8.0],
                "confidence": 0.70,
            },
            {
                "row_id": "row-3",
                "embedding": [9.0, 10.0, 11.0, 12.0],
                "confidence": 0.60,
            },
        ]
        emb_path = tmp_path / "detection_embeddings.parquet"
        write_detection_embeddings(records, emb_path)

        # Read back each record by row_id
        vec1 = read_detection_embedding(emb_path, "row-1")
        assert vec1 is not None
        assert vec1 == pytest.approx([1.0, 2.0, 3.0, 4.0])

        vec2 = read_detection_embedding(emb_path, "row-2")
        assert vec2 is not None
        assert vec2 == pytest.approx([5.0, 6.0, 7.0, 8.0])

        vec3 = read_detection_embedding(emb_path, "row-3")
        assert vec3 is not None
        assert vec3 == pytest.approx([9.0, 10.0, 11.0, 12.0])

    def test_read_returns_none_for_non_matching_row_id(self, tmp_path):
        """read_detection_embedding returns None when row_id doesn't match."""
        records = [
            {
                "row_id": "row-1",
                "embedding": [1.0, 2.0, 3.0],
            },
        ]
        emb_path = tmp_path / "detection_embeddings.parquet"
        write_detection_embeddings(records, emb_path)

        assert read_detection_embedding(emb_path, "nonexistent") is None

    def test_read_returns_none_for_nonexistent_file(self, tmp_path):
        """read_detection_embedding returns None when parquet file doesn't exist."""
        missing = tmp_path / "does_not_exist.parquet"
        result = read_detection_embedding(missing, "some-row")
        assert result is None

    def test_write_empty_records_is_noop(self, tmp_path):
        """write_detection_embeddings with empty list creates no file."""
        emb_path = tmp_path / "detection_embeddings.parquet"
        write_detection_embeddings([], emb_path)
        assert not emb_path.exists()

    def test_write_with_confidence(self, tmp_path):
        """write_detection_embeddings stores confidence column in parquet."""
        import pyarrow.parquet as pq

        records = [
            {
                "row_id": "r1",
                "embedding": [1.0, 2.0],
                "confidence": 0.95,
            },
            {
                "row_id": "r2",
                "embedding": [3.0, 4.0],
                "confidence": 0.42,
            },
        ]
        emb_path = tmp_path / "detection_embeddings.parquet"
        write_detection_embeddings(records, emb_path)

        table = pq.read_table(str(emb_path))
        assert "confidence" in table.schema.names
        confs = table.column("confidence").to_pylist()
        assert confs[0] == pytest.approx(0.95, abs=1e-4)
        assert confs[1] == pytest.approx(0.42, abs=1e-4)

    def test_write_without_confidence_key(self, tmp_path):
        """Records without confidence key write None for the column."""
        import pyarrow.parquet as pq

        records = [
            {
                "row_id": "r1",
                "embedding": [1.0, 2.0],
            },
        ]
        emb_path = tmp_path / "detection_embeddings.parquet"
        write_detection_embeddings(records, emb_path)

        table = pq.read_table(str(emb_path))
        assert "confidence" in table.schema.names
        confs = table.column("confidence").to_pylist()
        assert confs[0] is None


def test_run_detection_emit_embeddings(tmp_path):
    """emit_embeddings=True in run_detection() produces embedding records."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=12.0)

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    detections, summary, _, embedding_records = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        emit_embeddings=True,
    )

    # embedding_records should be a list (not None)
    assert embedding_records is not None

    # Each detection event should have a corresponding embedding record
    # (one embedding per detection event)
    if len(detections) > 0:
        assert len(embedding_records) > 0
        for rec in embedding_records:
            assert "filename" in rec
            assert "start_sec" in rec
            assert "end_sec" in rec
            assert "embedding" in rec
            assert len(rec["embedding"]) == 64
    else:
        # If no detections, no embedding records expected
        assert len(embedding_records) == 0


def test_run_detection_no_emit_embeddings(tmp_path):
    """emit_embeddings=False (default) returns None for embedding records."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=12.0)

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    _, _, _, embedding_records = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        emit_embeddings=False,
    )

    assert embedding_records is None


def test_emit_embeddings_roundtrip_via_parquet(tmp_path):
    """Embeddings from run_detection can be written and read back via parquet."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=12.0)

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    detections, _, _, embedding_records = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        emit_embeddings=True,
    )

    if not embedding_records:
        pytest.skip("No detections produced (classifier did not fire)")

    # Assign row_ids (as the worker would do before writing)
    for i, rec in enumerate(embedding_records):
        rec["row_id"] = f"test-row-{i}"

    emb_path = tmp_path / "detection_embeddings.parquet"
    write_detection_embeddings(embedding_records, emb_path)

    # Read back each embedding by row_id
    for rec in embedding_records:
        vec = read_detection_embedding(emb_path, rec["row_id"])
        assert vec is not None
        assert vec == pytest.approx(rec["embedding"], abs=1e-5)


def test_match_embedding_records_to_row_store():
    """match_embedding_records_to_row_store assigns row_id from row store rows."""
    # Simulate embedding records produced during detection (filename-relative offsets).
    # Filename encodes a UTC base epoch of 2024-01-01T00:00:00Z = 1704067200.0
    emb_records = [
        {
            "filename": "20240101T000000Z.wav",
            "start_sec": 10.0,
            "end_sec": 25.0,
            "embedding": [1.0, 2.0, 3.0],
            "confidence": 0.9,
        },
        {
            "filename": "20240101T000000Z.wav",
            "start_sec": 50.0,
            "end_sec": 65.0,
            "embedding": [4.0, 5.0, 6.0],
            "confidence": 0.8,
        },
    ]

    base_epoch = 1704067200.0
    row_store_rows = [
        {
            "row_id": "rid-aaa",
            "start_utc": str(base_epoch + 10.0),
            "end_utc": str(base_epoch + 25.0),
        },
        {
            "row_id": "rid-bbb",
            "start_utc": str(base_epoch + 50.0),
            "end_utc": str(base_epoch + 65.0),
        },
    ]

    matched = match_embedding_records_to_row_store(emb_records, row_store_rows)
    assert len(matched) == 2
    assert matched[0]["row_id"] == "rid-aaa"
    assert matched[0]["embedding"] == [1.0, 2.0, 3.0]
    assert matched[0]["confidence"] == 0.9
    assert matched[1]["row_id"] == "rid-bbb"
    assert matched[1]["embedding"] == [4.0, 5.0, 6.0]


def test_match_embedding_records_unmatched_dropped():
    """Embedding records that don't match any row store row are dropped."""
    emb_records = [
        {
            "filename": "20240101T000000Z.wav",
            "start_sec": 10.0,
            "end_sec": 25.0,
            "embedding": [1.0],
        },
    ]
    # Row store with no matching UTC window
    row_store_rows = [
        {
            "row_id": "rid-xxx",
            "start_utc": str(9999999.0),
            "end_utc": str(9999999.0 + 15.0),
        },
    ]

    matched = match_embedding_records_to_row_store(emb_records, row_store_rows)
    assert len(matched) == 0


def test_match_embedding_records_empty_inputs():
    """Empty records or empty row store returns empty list."""
    assert (
        match_embedding_records_to_row_store(
            [], [{"row_id": "a", "start_utc": "1", "end_utc": "2"}]
        )
        == []
    )
    assert (
        match_embedding_records_to_row_store(
            [{"filename": "f", "start_sec": 0, "end_sec": 1, "embedding": []}], []
        )
        == []
    )
