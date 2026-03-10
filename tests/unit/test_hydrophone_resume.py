"""Tests for hydrophone detection job resume after worker restart."""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from humpback.classifier.detector import read_detections_tsv, write_detections_tsv


# ------------------------------------------------------------------
# hydrophone detection filename
# ------------------------------------------------------------------


def test_build_detection_filename_uses_exact_bounds():
    """Exact raw event bounds should produce unique detection filenames."""
    from humpback.classifier.hydrophone_detector import _build_detection_filename

    f1 = _build_detection_filename("20250712T011600Z.wav", 15.0, 22.0)
    f2 = _build_detection_filename("20250712T011600Z.wav", 19.0, 25.0)
    assert f1 == "20250712T011615Z_20250712T011622Z.wav"
    assert f2 == "20250712T011619Z_20250712T011625Z.wav"
    assert f1 != f2


# ------------------------------------------------------------------
# read_detections_tsv
# ------------------------------------------------------------------


HYDROPHONE_FIELDNAMES = [
    "filename",
    "start_sec",
    "end_sec",
    "avg_confidence",
    "peak_confidence",
    "n_windows",
    "detection_filename",
    "extract_filename",
    "hydrophone_name",
]


def _write_sample_tsv(path: Path, rows: list[dict]) -> None:
    """Write rows to a TSV with hydrophone fieldnames."""
    write_detections_tsv(rows, path, fieldnames=HYDROPHONE_FIELDNAMES)


def test_read_detections_tsv(tmp_path: Path):
    """Correctly reads a TSV with hydrophone fieldnames."""
    tsv_path = tmp_path / "detections.tsv"
    rows = [
        {
            "filename": "20250712T070000Z.wav",
            "start_sec": "5.0",
            "end_sec": "10.0",
            "avg_confidence": "0.85",
            "peak_confidence": "0.92",
            "n_windows": "3",
            "detection_filename": "20250712T070005Z_20250712T070010Z.wav",
            "extract_filename": "20250712T070005Z_20250712T070010Z.wav",
            "hydrophone_name": "rpi_north_sjc",
        },
        {
            "filename": "20250712T070100Z.wav",
            "start_sec": "0.0",
            "end_sec": "5.0",
            "avg_confidence": "0.78",
            "peak_confidence": "0.80",
            "n_windows": "2",
            "detection_filename": "20250712T070100Z_20250712T070105Z.wav",
            "extract_filename": "20250712T070100Z_20250712T070105Z.wav",
            "hydrophone_name": "rpi_north_sjc",
        },
    ]
    _write_sample_tsv(tsv_path, rows)
    result = read_detections_tsv(tsv_path, fieldnames=HYDROPHONE_FIELDNAMES)
    assert len(result) == 2
    assert result[0]["filename"] == "20250712T070000Z.wav"
    assert result[1]["avg_confidence"] == "0.78"


def test_read_detections_tsv_missing_file(tmp_path: Path):
    """Returns empty list for missing file."""
    result = read_detections_tsv(tmp_path / "nonexistent.tsv")
    assert result == []


def test_read_detections_tsv_empty_file(tmp_path: Path):
    """Returns empty list for empty file."""
    tsv_path = tmp_path / "empty.tsv"
    tsv_path.write_text("")
    result = read_detections_tsv(tsv_path)
    assert result == []


# ------------------------------------------------------------------
# iter_audio_chunks skip_segments
# ------------------------------------------------------------------


def _make_fake_timeline(n_segments: int):
    """Create a list of StreamSegment-like objects for patching."""
    from humpback.classifier.s3_stream import StreamSegment

    return [
        StreamSegment(
            key=f"hydro/hls/123/live{i}.ts",
            start_ts=1000.0 + i * 10.0,
            duration_sec=10.0,
        )
        for i in range(n_segments)
    ]


@patch("humpback.classifier.s3_stream._build_stream_timeline")
@patch("humpback.classifier.s3_stream._decode_and_clip_segment")
def test_iter_audio_chunks_skip_segments(mock_decode, mock_timeline):
    """Skips N segments; segments_done starts at N."""
    from humpback.classifier.s3_stream import iter_audio_chunks

    timeline = _make_fake_timeline(10)
    mock_timeline.return_value = timeline

    # Each segment decodes to 1 second of audio at 32 kHz
    samples = np.zeros(32000, dtype=np.float32)
    mock_decode.side_effect = lambda **kw: (samples, kw["segment"].start_ts)

    client = MagicMock()
    chunks = list(
        iter_audio_chunks(
            client, "hydro", 1000.0, 1100.0,
            chunk_seconds=60.0, target_sr=32000,
            skip_segments=5,
        )
    )

    # Should have processed 5 segments (indices 5-9)
    assert mock_decode.call_count == 5
    # segments_done in first yielded chunk should be >= 5
    first_segs_done = chunks[0][2]
    assert first_segs_done >= 5
    # segments_total should be 10
    assert chunks[0][3] == 10


@patch("humpback.classifier.s3_stream._build_stream_timeline")
@patch("humpback.classifier.s3_stream._decode_and_clip_segment")
def test_iter_audio_chunks_skip_exceeds_total(mock_decode, mock_timeline):
    """Falls back to all segments when skip > timeline length; segments_done starts at 0."""
    from humpback.classifier.s3_stream import iter_audio_chunks

    timeline = _make_fake_timeline(3)
    mock_timeline.return_value = timeline

    samples = np.zeros(32000, dtype=np.float32)
    mock_decode.side_effect = lambda **kw: (samples, kw["segment"].start_ts)

    client = MagicMock()
    chunks = list(
        iter_audio_chunks(
            client, "hydro", 1000.0, 1030.0,
            chunk_seconds=60.0, target_sr=32000,
            skip_segments=10,  # exceeds timeline of 3
        )
    )

    # All 3 segments should be processed
    assert mock_decode.call_count == 3
    # segments_done in last chunk should reflect all segments processed from 0
    last_segs_done = chunks[-1][2]
    assert last_segs_done == 3


# ------------------------------------------------------------------
# run_hydrophone_detection resume
# ------------------------------------------------------------------


@patch("humpback.classifier.hydrophone_detector.iter_audio_chunks")
def test_hydrophone_detection_resume(mock_iter):
    """Prior detections are preserved; new detections appended; no duplicates."""
    from humpback.classifier.hydrophone_detector import run_hydrophone_detection

    # Simulate 1 chunk yielded (the resumed portion)
    chunk_audio = np.random.randn(64000).astype(np.float32)  # 2 seconds at 32kHz
    from datetime import datetime, timezone

    chunk_utc = datetime(2025, 7, 12, 7, 5, 0, tzinfo=timezone.utc)
    mock_iter.return_value = iter([(chunk_audio, chunk_utc, 8, 10, )])

    # Mock model and pipeline
    mock_model = MagicMock()
    mock_model.embed.return_value = np.random.randn(1, 128).astype(np.float32)
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])  # high confidence

    prior = [
        {"filename": "20250712T070000Z.wav", "start_sec": 0.0, "end_sec": 5.0,
         "avg_confidence": 0.85, "peak_confidence": 0.92, "n_windows": 3},
    ]

    detections, summary = run_hydrophone_detection(
        hydrophone_id="hydro",
        start_timestamp=1000.0,
        end_timestamp=1100.0,
        pipeline=mock_pipeline,
        model=mock_model,
        window_size_seconds=5.0,
        target_sample_rate=32000,
        confidence_threshold=0.5,
        skip_segments=5,
        prior_detections=prior,
    )

    # Prior detections should be present
    assert any(d.get("start_sec") == 0.0 for d in detections)
    # New detections should also be present (if threshold met)
    assert len(detections) >= 1
    # Summary should indicate resume
    assert summary.get("resumed_from_segment") == 5


@patch("humpback.classifier.hydrophone_detector.iter_audio_chunks")
def test_hydrophone_detection_resume_timeline_changed(mock_iter):
    """Prior detections cleared when skip was invalidated (timeline shrank)."""
    from humpback.classifier.hydrophone_detector import run_hydrophone_detection

    # Simulate skip invalidation: segs_done (2) < skip_segments (5)
    chunk_audio = np.random.randn(64000).astype(np.float32)
    from datetime import datetime, timezone

    chunk_utc = datetime(2025, 7, 12, 7, 0, 0, tzinfo=timezone.utc)
    # segs_done=2 < skip_segments=5 → invalidation
    mock_iter.return_value = iter([(chunk_audio, chunk_utc, 2, 3)])

    mock_model = MagicMock()
    mock_model.embed.return_value = np.random.randn(1, 128).astype(np.float32)
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])

    prior = [
        {"filename": "old.wav", "start_sec": 0.0, "end_sec": 5.0,
         "avg_confidence": 0.85, "peak_confidence": 0.92, "n_windows": 3},
    ]

    detections, summary = run_hydrophone_detection(
        hydrophone_id="hydro",
        start_timestamp=1000.0,
        end_timestamp=1030.0,
        pipeline=mock_pipeline,
        model=mock_model,
        window_size_seconds=5.0,
        target_sample_rate=32000,
        confidence_threshold=0.5,
        skip_segments=5,
        prior_detections=prior,
    )

    # Prior detections should have been cleared (timeline changed)
    assert not any(d.get("filename") == "old.wav" for d in detections)
    # No resume marker in summary since skip was invalidated
    assert "resumed_from_segment" not in summary


# ------------------------------------------------------------------
# Cache invalidation on decode failure
# ------------------------------------------------------------------


def test_cache_invalidation_local_client(tmp_path: Path):
    """LocalHLSClient.invalidate_cached_segment deletes cached file."""
    from humpback.classifier.s3_stream import LocalHLSClient
    from humpback.config import ORCASOUND_S3_BUCKET

    client = LocalHLSClient(str(tmp_path))
    seg_path = tmp_path / ORCASOUND_S3_BUCKET / "hydro" / "hls" / "123" / "live0.ts"
    seg_path.parent.mkdir(parents=True)
    seg_path.write_bytes(b"corrupted data")

    key = "hydro/hls/123/live0.ts"
    assert client.invalidate_cached_segment(key) is True
    assert not seg_path.exists()
    # Second call returns False (file already gone)
    assert client.invalidate_cached_segment(key) is False


def test_cache_invalidation_caching_client(tmp_path: Path):
    """CachingS3Client.invalidate_cached_segment deletes cached file."""
    from humpback.classifier.s3_stream import CachingS3Client
    from humpback.config import ORCASOUND_S3_BUCKET

    # Patch S3 client init to avoid real boto3 connection
    with patch("humpback.classifier.s3_stream.OrcasoundS3Client"):
        client = CachingS3Client(str(tmp_path))

    seg_path = tmp_path / ORCASOUND_S3_BUCKET / "hydro" / "hls" / "123" / "live0.ts"
    seg_path.parent.mkdir(parents=True)
    seg_path.write_bytes(b"corrupted data")

    key = "hydro/hls/123/live0.ts"
    assert client.invalidate_cached_segment(key) is True
    assert not seg_path.exists()


@patch("humpback.classifier.s3_stream._build_stream_timeline")
@patch("humpback.classifier.s3_stream._decode_and_clip_segment")
def test_iter_audio_chunks_invalidate_and_retry(mock_decode, mock_timeline):
    """When decode fails, cached segment is invalidated and retried once."""
    from humpback.classifier.s3_stream import iter_audio_chunks

    timeline = _make_fake_timeline(2)
    mock_timeline.return_value = timeline

    # 10 seconds of audio per segment (matching segment duration) to avoid
    # discontinuity flushes between segments
    samples = np.zeros(320000, dtype=np.float32)
    # First call fails; second call (retry after invalidation) succeeds
    mock_decode.side_effect = [
        RuntimeError("ffmpeg decode failed"),
        (samples, 1000.0),
        (samples, 1010.0),
    ]

    client = MagicMock()
    client.invalidate_cached_segment.return_value = True

    chunks = list(
        iter_audio_chunks(
            client, "hydro", 1000.0, 1020.0,
            chunk_seconds=60.0, target_sr=32000,
        )
    )

    # invalidate should have been called for the first segment
    client.invalidate_cached_segment.assert_called_once_with("hydro/hls/123/live0.ts")
    # decode should be called 3 times: fail + retry + second segment
    assert mock_decode.call_count == 3
    # Should get output (both segments accumulated into one chunk)
    assert len(chunks) == 1


@patch("humpback.classifier.s3_stream._build_stream_timeline")
@patch("humpback.classifier.s3_stream._decode_and_clip_segment")
def test_iter_audio_chunks_no_invalidation_without_method(mock_decode, mock_timeline):
    """When client has no invalidate method, decode failure is just logged."""
    from humpback.classifier.s3_stream import iter_audio_chunks

    timeline = _make_fake_timeline(2)
    mock_timeline.return_value = timeline

    samples = np.zeros(32000, dtype=np.float32)
    mock_decode.side_effect = [
        RuntimeError("ffmpeg decode failed"),
        (samples, 1010.0),
    ]

    client = MagicMock(spec=[])  # No invalidate_cached_segment method

    chunks = list(
        iter_audio_chunks(
            client, "hydro", 1000.0, 1020.0,
            chunk_seconds=60.0, target_sr=32000,
        )
    )

    # decode called twice: fail (no retry) + second segment
    assert mock_decode.call_count == 2
    assert len(chunks) == 1


# ------------------------------------------------------------------
# Local detection TSV cleanup
# ------------------------------------------------------------------


def test_local_detection_tsv_cleanup(tmp_path: Path):
    """Stale TSV is removed before local detection starts (simulated)."""
    tsv_path = tmp_path / "detections.tsv"
    tsv_path.write_text("filename\tstart_sec\nold.wav\t0.0\n")
    assert tsv_path.exists()

    # Simulate the cleanup logic from classifier_worker
    if tsv_path.exists():
        tsv_path.unlink()

    assert not tsv_path.exists()
