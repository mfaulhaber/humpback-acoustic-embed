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
from humpback.processing.timeline_renderers import DEFAULT_TIMELINE_RENDERER
from humpback.processing.timeline_repository import (
    TimelineSourceRef,
    TimelineTileRequest,
)
from humpback.processing.timeline_tiles import ZOOM_LEVELS, tile_count
from humpback.services.timeline_tile_service import (
    repository_from_settings,
    source_ref_from_job,
)
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
    source_ref = TimelineSourceRef(
        hydrophone_id="test_hydro",
        source_identity="/fake/cache",
        job_start_timestamp=1000.0,
        job_end_timestamp=1000.0 + job_duration,
    )
    _populate_shared_tiles(settings, source_ref, job_duration)


def _populate_shared_tiles(settings, source_ref, job_duration):
    repository = repository_from_settings(settings)
    renderer = DEFAULT_TIMELINE_RENDERER
    for zoom in ZOOM_LEVELS:
        n = tile_count(zoom, job_duration_sec=job_duration)
        for i in range(n):
            request = TimelineTileRequest(
                zoom_level=zoom,
                tile_index=i,
                freq_min=0,
                freq_max=3000,
                width_px=settings.timeline_tile_width_px,
                height_px=settings.timeline_tile_height_px,
            )
            repository.put(
                source_ref,
                renderer.renderer_id,
                renderer.version,
                request,
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
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


async def test_export_auto_prepares_tiles(
    completed_job, session, settings, tmp_path, job_duration
):
    """Export auto-prepares tiles when they are missing."""
    output_dir = tmp_path / "export_output"
    fake_audio = np.zeros(int(32000 * 300), dtype=np.float32)

    # _prepare_tiles_sync is called to render tiles; mock it to populate the
    # shared span repository.
    def fake_prepare(*, job, settings, cache):
        _populate_shared_tiles(
            settings, source_ref_from_job(job, settings), job_duration
        )
        return 0

    with (
        patch(
            "humpback.api.routers.timeline._prepare_tiles_sync",
            side_effect=fake_prepare,
        ),
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

    assert result.tile_count > 0
    assert Path(result.output_path, "manifest.json").exists()


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


async def test_export_audio_is_rms_normalized(
    completed_job, session, settings, tmp_path, populated_tile_cache
):
    """Exported audio chunks should be RMS-normalized to the target level.

    This pins the fix for the pre-PCEN inconsistency where exported
    chunks bypassed the viewer's normalization and shipped raw audio.
    """
    import subprocess

    output_dir = tmp_path / "export_rms"
    sr = 32000
    duration_sec = 300
    # Build an input signal at a known RMS far from the target so any
    # regression that bypasses ``normalize_for_playback`` is visible.
    t = np.arange(sr * duration_sec, dtype=np.float32) / sr
    fake_audio = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    # Pass the real encode_mp3 through by not patching it. Tiles still
    # need to be provided via the ``populated_tile_cache`` fixture.
    with patch(
        "humpback.processing.timeline_audio.resolve_timeline_audio",
        return_value=fake_audio,
    ):
        result = await export_timeline(completed_job, output_dir, session, settings)

    chunk_path = Path(result.output_path) / "audio" / "chunk_0000.mp3"
    assert chunk_path.exists()

    # Decode the MP3 back to raw float samples via ffmpeg so we can
    # measure the RMS of the exported data.
    ffmpeg = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(chunk_path),
            "-ac",
            "1",
            "-f",
            "f32le",
            "-ar",
            str(sr),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    decoded = np.frombuffer(ffmpeg.stdout, dtype=np.float32)
    assert decoded.size > 0

    rms = float(np.sqrt(np.mean(decoded.astype(np.float64) ** 2)))
    rms_dbfs = 20.0 * np.log10(rms) if rms > 0 else -np.inf
    # MP3 encoding introduces some drift; accept ±2 dB.
    assert abs(rms_dbfs - settings.playback_target_rms_dbfs) < 2.0, (
        f"Exported chunk RMS {rms_dbfs:.2f} dBFS far from target "
        f"{settings.playback_target_rms_dbfs} dBFS"
    )


async def test_export_writes_cache_version_marker(
    completed_job, session, settings, tmp_path
):
    """Exporting a job should leave a current-version marker in its cache
    directory (set by the first ``_prepare_tiles_sync`` call), and must
    not leave any legacy gain-profile or ref_db sidecars behind.
    """
    from humpback.processing.timeline_cache import TIMELINE_CACHE_VERSION

    output_dir = tmp_path / "export_version"
    fake_audio = np.zeros(int(32000 * 300), dtype=np.float32)

    # _prepare_tiles_sync will run because the tile cache for this job
    # does not yet exist — we intercept the per-tile renderer to avoid
    # pulling in real STFT work. Mirror the real prepare pass: migrate
    # first (writing the version marker) before generating any tiles so
    # migration does not sweep them away.
    def fake_prepare(*, job, settings, cache):
        job_dir = cache.cache_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        cache.ensure_job_cache_current(job.id)
        _populate_shared_tiles(settings, source_ref_from_job(job, settings), 100.0)
        return 0

    with (
        patch(
            "humpback.api.routers.timeline._prepare_tiles_sync",
            side_effect=fake_prepare,
        ),
        patch(
            "humpback.processing.timeline_audio.resolve_timeline_audio",
            return_value=fake_audio,
        ),
        patch(
            "humpback.processing.audio_encoding.encode_mp3",
            return_value=b"fake-mp3-data",
        ),
    ):
        await export_timeline(completed_job, output_dir, session, settings)

    job_cache_dir = settings.storage_root / "timeline_cache" / completed_job
    version_path = job_cache_dir / ".cache_version"
    assert version_path.exists()
    assert int(version_path.read_text().strip()) == TIMELINE_CACHE_VERSION
    assert not (job_cache_dir / ".ref_db.json").exists()
    assert not (job_cache_dir / ".gain_profile.json").exists()


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
