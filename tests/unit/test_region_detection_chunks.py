"""Unit tests for region detection chunk artifacts and manifest I/O.

Covers the manifest and chunk parquet helpers in ``call_parsing.storage``
and the resume/verify logic in ``region_detection_worker``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from humpback.call_parsing.storage import (
    chunk_parquet_path,
    read_all_chunk_traces,
    read_manifest,
    update_manifest_chunk,
    write_chunk_trace,
    write_manifest,
)
from humpback.call_parsing.types import WindowScore
from humpback.workers.region_detection_worker import (
    _build_manifest,
    _verify_manifest_chunks,
)
from humpback.schemas.call_parsing import RegionDetectionConfig


# ---- Manifest I/O --------------------------------------------------------


def test_write_read_manifest_round_trip(tmp_path: Path):
    manifest = {
        "job_id": "test-123",
        "config": {"stream_chunk_sec": 1800},
        "chunks": [
            {"index": 0, "status": "pending", "trace_rows": None},
            {"index": 1, "status": "pending", "trace_rows": None},
        ],
    }
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    assert loaded is not None
    assert loaded["job_id"] == "test-123"
    assert len(loaded["chunks"]) == 2


def test_read_manifest_returns_none_when_missing(tmp_path: Path):
    assert read_manifest(tmp_path) is None


def test_update_manifest_chunk(tmp_path: Path):
    manifest = {
        "job_id": "test-456",
        "chunks": [
            {"index": 0, "status": "pending", "trace_rows": None},
            {"index": 1, "status": "pending", "trace_rows": None},
        ],
    }
    write_manifest(tmp_path, manifest)
    update_manifest_chunk(tmp_path, 0, {"status": "complete", "trace_rows": 42})

    loaded = read_manifest(tmp_path)
    assert loaded is not None
    assert loaded["chunks"][0]["status"] == "complete"
    assert loaded["chunks"][0]["trace_rows"] == 42
    assert loaded["chunks"][1]["status"] == "pending"


def test_update_manifest_chunk_raises_when_no_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        update_manifest_chunk(tmp_path, 0, {"status": "complete"})


# ---- Chunk parquet I/O ---------------------------------------------------


def test_write_and_read_chunk_traces(tmp_path: Path):
    scores_0 = [
        WindowScore(time_sec=0.0, score=0.8),
        WindowScore(time_sec=1.0, score=0.6),
    ]
    scores_1 = [
        WindowScore(time_sec=2.0, score=0.9),
        WindowScore(time_sec=3.0, score=0.4),
    ]

    write_chunk_trace(tmp_path, 0, scores_0)
    write_chunk_trace(tmp_path, 1, scores_1)

    assert chunk_parquet_path(tmp_path, 0).exists()
    assert chunk_parquet_path(tmp_path, 1).exists()

    all_scores = read_all_chunk_traces(tmp_path, 2)
    assert len(all_scores) == 4
    assert all_scores[0].time_sec == 0.0
    assert all_scores[2].time_sec == 2.0


def test_chunk_parquet_path_format(tmp_path: Path):
    assert chunk_parquet_path(tmp_path, 0) == tmp_path / "chunks" / "0000.parquet"
    assert chunk_parquet_path(tmp_path, 47) == tmp_path / "chunks" / "0047.parquet"


def test_write_empty_chunk(tmp_path: Path):
    write_chunk_trace(tmp_path, 0, [])
    all_scores = read_all_chunk_traces(tmp_path, 1)
    assert all_scores == []


# ---- Build manifest -------------------------------------------------------


def test_build_manifest_structure():
    config = RegionDetectionConfig(
        stream_chunk_sec=1800, window_size_seconds=5.0, hop_seconds=1.0
    )
    edges = [(0.0, 1800.0), (1800.0, 3600.0)]
    manifest = _build_manifest("job-abc", edges, config)

    assert manifest["job_id"] == "job-abc"
    assert manifest["config"]["stream_chunk_sec"] == 1800
    assert len(manifest["chunks"]) == 2
    assert manifest["chunks"][0]["index"] == 0
    assert manifest["chunks"][0]["start_sec"] == 0.0
    assert manifest["chunks"][0]["end_sec"] == 1800.0
    assert manifest["chunks"][0]["status"] == "pending"
    assert manifest["chunks"][1]["index"] == 1
    assert manifest["chunks"][1]["start_sec"] == 1800.0


# ---- Verify manifest chunks (resume logic) --------------------------------


def test_verify_all_complete_chunks_present(tmp_path: Path):
    scores = [WindowScore(time_sec=0.0, score=0.5)]
    write_chunk_trace(tmp_path, 0, scores)
    write_chunk_trace(tmp_path, 1, scores)

    manifest = {
        "chunks": [
            {
                "index": 0,
                "status": "complete",
                "completed_at": "t",
                "trace_rows": 1,
                "elapsed_sec": 1.0,
            },
            {
                "index": 1,
                "status": "complete",
                "completed_at": "t",
                "trace_rows": 1,
                "elapsed_sec": 1.0,
            },
        ]
    }
    verified = _verify_manifest_chunks(manifest, tmp_path)
    assert verified == 2
    assert manifest["chunks"][0]["status"] == "complete"
    assert manifest["chunks"][1]["status"] == "complete"


def test_verify_resets_missing_chunk_to_pending(tmp_path: Path):
    scores = [WindowScore(time_sec=0.0, score=0.5)]
    write_chunk_trace(tmp_path, 0, scores)
    # chunk 1 parquet is intentionally missing

    manifest = {
        "chunks": [
            {
                "index": 0,
                "status": "complete",
                "completed_at": "t",
                "trace_rows": 1,
                "elapsed_sec": 1.0,
            },
            {
                "index": 1,
                "status": "complete",
                "completed_at": "t",
                "trace_rows": 1,
                "elapsed_sec": 1.0,
            },
        ]
    }
    verified = _verify_manifest_chunks(manifest, tmp_path)
    assert verified == 1
    assert manifest["chunks"][0]["status"] == "complete"
    assert manifest["chunks"][1]["status"] == "pending"
    assert manifest["chunks"][1]["completed_at"] is None
    assert manifest["chunks"][1]["trace_rows"] is None


def test_verify_skips_pending_chunks(tmp_path: Path):
    manifest = {
        "chunks": [
            {
                "index": 0,
                "status": "pending",
                "completed_at": None,
                "trace_rows": None,
                "elapsed_sec": None,
            },
        ]
    }
    verified = _verify_manifest_chunks(manifest, tmp_path)
    assert verified == 0


# ---- Timeline filtering for per-chunk slicing ------------------------------


def test_filter_timeline_selects_overlapping_segments():
    """Segments overlapping a chunk window are selected; others excluded."""
    from humpback.classifier.archive import StreamSegment

    full_timeline = [
        StreamSegment(key="a", start_ts=100.0, duration_sec=50.0),  # 100-150
        StreamSegment(key="b", start_ts=150.0, duration_sec=50.0),  # 150-200
        StreamSegment(key="c", start_ts=200.0, duration_sec=50.0),  # 200-250
        StreamSegment(key="d", start_ts=300.0, duration_sec=50.0),  # 300-350
    ]

    # Chunk covers 140-210 — should include a (ends at 150>140), b, c (starts at 200<210)
    filtered = [s for s in full_timeline if s.start_ts < 210.0 and s.end_ts > 140.0]
    assert [s.key for s in filtered] == ["a", "b", "c"]

    # Chunk covers 250-300 — no segments overlap
    filtered = [s for s in full_timeline if s.start_ts < 300.0 and s.end_ts > 250.0]
    assert filtered == []

    # Chunk covers 300-400 — only d
    filtered = [s for s in full_timeline if s.start_ts < 400.0 and s.end_ts > 300.0]
    assert [s.key for s in filtered] == ["d"]


# ---- Trace dedup at merge time -----------------------------------------------


def test_merge_deduplicates_by_time_sec_keeping_highest_score(tmp_path: Path):
    """When chunks produce overlapping timestamps, dedup keeps highest score."""
    # Chunk 0: timestamps 0-4, chunk 1: overlaps at 3-4 then continues 5-7
    chunk_0 = [
        WindowScore(time_sec=0.0, score=0.1),
        WindowScore(time_sec=1.0, score=0.5),
        WindowScore(time_sec=2.0, score=0.9),
        WindowScore(time_sec=3.0, score=0.3),
        WindowScore(time_sec=4.0, score=0.2),
    ]
    chunk_1 = [
        WindowScore(time_sec=3.0, score=0.8),  # higher than chunk 0's 0.3
        WindowScore(time_sec=4.0, score=0.1),  # lower than chunk 0's 0.2
        WindowScore(time_sec=5.0, score=0.7),
        WindowScore(time_sec=6.0, score=0.6),
    ]
    write_chunk_trace(tmp_path, 0, chunk_0)
    write_chunk_trace(tmp_path, 1, chunk_1)

    raw = read_all_chunk_traces(tmp_path, 2)
    assert len(raw) == 9  # 5 + 4, with duplicates

    # Apply the same dedup logic as the worker
    seen: dict[float, WindowScore] = {}
    for ws in raw:
        existing = seen.get(ws.time_sec)
        if existing is None or ws.score > existing.score:
            seen[ws.time_sec] = ws
    deduped = sorted(seen.values(), key=lambda ws: ws.time_sec)

    assert len(deduped) == 7  # 0,1,2,3,4,5,6
    scores_by_time = {ws.time_sec: ws.score for ws in deduped}
    assert scores_by_time[3.0] == 0.8  # kept chunk_1's higher score
    assert scores_by_time[4.0] == 0.2  # kept chunk_0's higher score


def test_merge_no_duplicates_passes_through(tmp_path: Path):
    """Non-overlapping chunks produce no dedup changes."""
    chunk_0 = [
        WindowScore(time_sec=0.0, score=0.5),
        WindowScore(time_sec=1.0, score=0.6),
    ]
    chunk_1 = [
        WindowScore(time_sec=2.0, score=0.7),
        WindowScore(time_sec=3.0, score=0.8),
    ]
    write_chunk_trace(tmp_path, 0, chunk_0)
    write_chunk_trace(tmp_path, 1, chunk_1)

    raw = read_all_chunk_traces(tmp_path, 2)
    assert len(raw) == 4

    seen: dict[float, WindowScore] = {}
    for ws in raw:
        existing = seen.get(ws.time_sec)
        if existing is None or ws.score > existing.score:
            seen[ws.time_sec] = ws
    deduped = sorted(seen.values(), key=lambda ws: ws.time_sec)

    assert len(deduped) == 4  # no change


# ---- Offset fix: score_audio_windows with correct time_offset_sec ----------


def test_score_audio_windows_offset_produces_non_overlapping_traces():
    """Two audio buffers with different time_offset_sec produce non-overlapping traces.

    This simulates the fixed behavior where iter_audio_chunks yields two
    buffers due to an HLS discontinuity. Each buffer is scored with the
    correct time_offset_sec derived from seg_start_utc, not the chunk start.
    """
    import numpy as np

    from humpback.classifier.detector import score_audio_windows
    from humpback.classifier.trainer import train_binary_classifier
    from humpback.processing.inference import FakeTFLiteModel

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64).astype(np.float32)
    neg = rng.randn(20, 64).astype(np.float32) - 5.0
    classifier, _ = train_binary_classifier(pos, neg)

    sr = 16000
    config = {"window_size_seconds": 5.0, "hop_seconds": 1.0}

    # Buffer 1: 8 seconds of audio at offset 100s
    buf1 = np.random.randn(sr * 8).astype(np.float32)
    records1 = score_audio_windows(
        buf1, sr, model, classifier, config, time_offset_sec=100.0
    )

    # Buffer 2: 8 seconds of audio at offset 115s (gap from 108 to 115)
    buf2 = np.random.randn(sr * 8).astype(np.float32)
    records2 = score_audio_windows(
        buf2, sr, model, classifier, config, time_offset_sec=115.0
    )

    times1 = [r["offset_sec"] for r in records1]
    times2 = [r["offset_sec"] for r in records2]

    assert len(times1) > 0
    assert len(times2) > 0
    assert times1[0] == pytest.approx(100.0)
    assert times2[0] == pytest.approx(115.0)
    assert max(times1) < min(times2), (
        f"Buffers should not overlap: buf1 max={max(times1)}, buf2 min={min(times2)}"
    )


def test_build_archive_detection_provider_accepts_force_refresh():
    """build_archive_detection_provider passes force_refresh to CachingHLSProvider."""
    from unittest.mock import patch

    from humpback.classifier.providers import build_archive_detection_provider
    from humpback.classifier.providers.orcasound_hls import CachingHLSProvider

    with patch("humpback.classifier.providers.orcasound_hls.CachingS3Client"):
        provider = build_archive_detection_provider(
            "rpi_orcasound_lab",
            local_cache_path=None,
            s3_cache_path="/tmp/fake-cache",
            force_refresh=False,
        )
    assert isinstance(provider, CachingHLSProvider)
    assert provider._force_refresh is False
