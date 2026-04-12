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
