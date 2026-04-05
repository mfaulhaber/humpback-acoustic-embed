"""Unit tests for timeline export service."""

import json
import math
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import insert

from humpback.classifier.detection_rows import ROW_STORE_FIELDNAMES
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.vocalization import VocalizationType
from humpback.processing.timeline_tiles import ZOOM_LEVELS, tile_count
from humpback.services.timeline_export import (
    ExportError,
    _flatten_label,
    _read_confidence_scores,
    _read_detections,
    export_timeline,
)
from humpback.storage import detection_diagnostics_path, detection_row_store_path

# ---- Fixtures ----


@pytest.fixture
def model_id():
    return str(uuid.uuid4())


@pytest.fixture
def job_id():
    return str(uuid.uuid4())


@pytest.fixture
def job_duration():
    return 100.0  # 100-second job


@pytest.fixture
async def completed_job(session, settings, engine, model_id, job_id, job_duration):
    """Create a completed hydrophone detection job with diagnostics and row store."""
    async with session.begin():
        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="humpback_test",
                model_path="/fake/path",
                model_version="v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id=model_id,
                detection_mode="windowed",
                window_selection="tiling",
                hydrophone_id="test_hydro",
                hydrophone_name="Test Hydrophone",
                start_timestamp=1000.0,
                end_timestamp=1000.0 + job_duration,
                local_cache_path="/fake/cache",
            )
        )

    # Create diagnostics parquet
    n_windows = int(job_duration / 5)
    diag_path = detection_diagnostics_path(settings.storage_root, job_id)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "filename": pa.array(["stream.ts"] * n_windows, type=pa.string()),
            "window_index": pa.array(list(range(n_windows)), type=pa.int32()),
            "offset_sec": pa.array(
                [float(i * 5) for i in range(n_windows)], type=pa.float32()
            ),
            "end_sec": pa.array(
                [float(i * 5 + 5) for i in range(n_windows)], type=pa.float32()
            ),
            "confidence": pa.array(
                [0.5 + 0.4 * (i / max(1, n_windows - 1)) for i in range(n_windows)],
                type=pa.float32(),
            ),
            "is_overlapped": pa.array([False] * n_windows, type=pa.bool_()),
            "overlap_sec": pa.array([0.0] * n_windows, type=pa.float32()),
        }
    )
    pq.write_table(table, str(diag_path))

    # Create row store with a couple detection rows
    rs_path = detection_row_store_path(settings.storage_root, job_id)
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    row1_id = str(uuid.uuid4())
    row2_id = str(uuid.uuid4())
    rows = {field: ["", ""] for field in ROW_STORE_FIELDNAMES}
    rows["row_id"] = [row1_id, row2_id]
    rows["start_utc"] = ["1010.0", "1050.0"]
    rows["end_utc"] = ["1015.0", "1055.0"]
    rows["avg_confidence"] = ["0.85", "0.72"]
    rows["peak_confidence"] = ["0.95", "0.80"]
    rows["humpback"] = ["1", ""]
    rs_table = pa.table({k: pa.array(v, type=pa.string()) for k, v in rows.items()})
    pq.write_table(rs_table, str(rs_path))

    return job_id


@pytest.fixture
def populated_tile_cache(settings, job_id, job_duration):
    """Create fake tile PNGs in the timeline cache directory."""
    cache_dir = settings.storage_root / "timeline_cache" / job_id
    for zoom in ZOOM_LEVELS:
        n = tile_count(zoom, job_duration_sec=job_duration)
        zoom_dir = cache_dir / zoom
        zoom_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            # Minimal valid PNG (1x1 pixel)
            (zoom_dir / f"tile_{i:04d}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
            )


# ---- Tests: _flatten_label ----


def test_flatten_label_humpback():
    assert (
        _flatten_label({"humpback": 1, "orca": None, "ship": None, "background": None})
        == "humpback"
    )


def test_flatten_label_orca():
    assert (
        _flatten_label({"humpback": None, "orca": 1, "ship": None, "background": None})
        == "orca"
    )


def test_flatten_label_none():
    assert (
        _flatten_label(
            {"humpback": None, "orca": None, "ship": None, "background": None}
        )
        is None
    )


def test_flatten_label_zero_is_none():
    assert (
        _flatten_label({"humpback": 0, "orca": None, "ship": None, "background": None})
        is None
    )


# ---- Tests: _read_confidence_scores ----


async def test_read_confidence_scores(completed_job, settings, job_duration):
    scores = _read_confidence_scores(
        job_id=completed_job,
        job_start=1000.0,
        job_end=1000.0 + job_duration,
        window_sec=5.0,
        settings=settings,
    )
    expected_buckets = max(1, int(job_duration / 5.0))
    assert len(scores) == expected_buckets
    # All buckets should have data (non-null)
    assert all(s is not None for s in scores)


async def test_read_confidence_scores_missing_diagnostics(settings):
    scores = _read_confidence_scores(
        job_id="nonexistent",
        job_start=0.0,
        job_end=100.0,
        window_sec=5.0,
        settings=settings,
    )
    assert scores == []


# ---- Tests: _read_detections ----


async def test_read_detections(completed_job, settings):
    detections = _read_detections(job_id=completed_job, settings=settings)
    assert len(detections) == 2

    d1 = detections[0]
    assert d1["row_id"] != ""
    assert d1["start_utc"] == 1010.0
    assert d1["end_utc"] == 1015.0
    assert d1["avg_confidence"] == 0.85
    assert d1["label"] == "humpback"

    d2 = detections[1]
    assert d2["label"] is None  # Not labeled


# ---- Tests: export_timeline ----


async def test_export_job_not_found(session, settings, tmp_path):
    with pytest.raises(ExportError, match="not found"):
        await export_timeline("nonexistent", tmp_path, session, settings)


async def test_export_job_not_complete(session, settings, engine, tmp_path, model_id):
    job_id = str(uuid.uuid4())
    async with session.begin():
        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="test",
                model_path="/fake",
                model_version="v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="running",
                classifier_model_id=model_id,
                hydrophone_id="h1",
                start_timestamp=0.0,
                end_timestamp=100.0,
            )
        )

    with pytest.raises(ExportError, match="must be complete"):
        await export_timeline(job_id, tmp_path, session, settings)


async def test_export_tiles_not_prepared(completed_job, session, settings, tmp_path):
    """Export fails when tiles haven't been rendered."""
    with pytest.raises(ExportError, match="not fully rendered"):
        await export_timeline(completed_job, tmp_path, session, settings)


async def test_export_full_pipeline(
    completed_job, session, settings, tmp_path, populated_tile_cache, job_duration
):
    """Full export pipeline with mocked audio resolution."""
    output_dir = tmp_path / "export_output"

    # Mock resolve_timeline_audio to return silence
    fake_audio = np.zeros(int(32000 * 300), dtype=np.float32)

    with (
        patch(
            "humpback.processing.timeline_audio.resolve_timeline_audio",
            return_value=fake_audio,
        ),
        patch(
            "humpback.processing.audio_encoding.encode_mp3",
            return_value=b"fake-mp3-data",
        ),
    ):
        result = await export_timeline(completed_job, output_dir, session, settings)

    # Verify result
    assert result.job_id == completed_job
    assert result.tile_count > 0
    assert result.audio_chunk_count == math.ceil(job_duration / 300)

    # Verify directory structure
    job_dir = Path(result.output_path)
    assert (job_dir / "manifest.json").exists()

    # Check tiles copied
    for zoom in ZOOM_LEVELS:
        n = tile_count(zoom, job_duration_sec=job_duration)
        for i in range(n):
            assert (job_dir / "tiles" / zoom / f"tile_{i:04d}.png").exists()

    # Check audio chunks
    for i in range(result.audio_chunk_count):
        assert (job_dir / "audio" / f"chunk_{i:04d}.mp3").exists()

    # Verify manifest structure
    manifest = json.loads((job_dir / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["job"]["id"] == completed_job
    assert manifest["job"]["hydrophone_name"] == "Test Hydrophone"
    assert manifest["job"]["window_selection"] == "tiling"
    assert manifest["tiles"]["zoom_levels"] == list(ZOOM_LEVELS)
    assert manifest["tiles"]["tile_size"] == [512, 256]
    assert manifest["audio"]["chunk_duration_sec"] == 300
    assert manifest["audio"]["format"] == "mp3"
    assert manifest["audio"]["sample_rate"] == 32000
    assert isinstance(manifest["confidence"]["scores"], list)
    assert len(manifest["detections"]) == 2
    assert manifest["detections"][0]["label"] == "humpback"
    assert manifest["detections"][1]["label"] is None
    assert isinstance(manifest["vocalization_labels"], list)
    assert isinstance(manifest["vocalization_types"], list)


async def test_export_with_vocalization_types(
    completed_job, session, settings, tmp_path, populated_tile_cache, engine
):
    """Vocalization types appear in the manifest."""
    async with session.begin():
        await session.execute(
            insert(VocalizationType).values(id=str(uuid.uuid4()), name="moan")
        )
        await session.execute(
            insert(VocalizationType).values(id=str(uuid.uuid4()), name="whup")
        )

    output_dir = tmp_path / "export_voc"
    fake_audio = np.zeros(int(32000 * 300), dtype=np.float32)

    with (
        patch(
            "humpback.processing.timeline_audio.resolve_timeline_audio",
            return_value=fake_audio,
        ),
        patch(
            "humpback.processing.audio_encoding.encode_mp3",
            return_value=b"fake-mp3-data",
        ),
    ):
        result = await export_timeline(completed_job, output_dir, session, settings)

    manifest = json.loads((Path(result.output_path) / "manifest.json").read_text())
    type_names = {t["name"] for t in manifest["vocalization_types"]}
    assert "moan" in type_names
    assert "whup" in type_names
