"""Integration tests for the call parsing API router."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from httpx import AsyncClient
from sklearn.pipeline import Pipeline

from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.call_parsing.storage import region_job_dir
from humpback.classifier.trainer import train_binary_classifier
from humpback.database import create_engine, create_session_factory
from humpback.ml.checkpointing import save_checkpoint
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.models.classifier import ClassifierModel
from humpback.models.model_registry import ModelConfig
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingSample,
)
from humpback.processing.inference import FakeTFLiteModel
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
)
from humpback.storage import ensure_dir

BASE = "/call-parsing"


def _write_sine_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    n = int(sample_rate * duration_sec)
    samples = [
        int(32767 * 0.7 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _synthetic_classifier() -> Pipeline:
    rng = np.random.RandomState(42)
    seed_embedding = np.sin(np.arange(64) * 2 / 64).astype(np.float32)
    pos = np.tile(seed_embedding, (20, 1)) + rng.randn(20, 64) * 0.01
    neg = rng.randn(20, 64) * 0.5 - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)
    return pipeline


async def _seed_source_fixtures(
    app_settings, *, with_real_audio: bool = False
) -> dict[str, str]:
    """Insert AudioFile + ModelConfig + ClassifierModel rows via the app DB.

    POST /call-parsing/runs now validates every FK (audio_file_id /
    model_config_id / classifier_model_id), so every router test that
    hits the endpoint needs these rows to exist first. Tests that drive
    the worker synchronously pass ``with_real_audio=True`` so the helper
    also writes a short WAV under ``storage_root/audio_src/`` and
    configures the ``AudioFile`` row's ``source_folder`` to point at it;
    this keeps the classifier/model rows dimensioned to match the
    deterministic ``FakeTFLiteModel`` the worker test monkeypatches in.
    """
    if with_real_audio:
        storage_root = Path(app_settings.storage_root)
        audio_dir = storage_root / "audio_src"
        audio_path = audio_dir / "router-fixture.wav"
        _write_sine_wav(audio_path, duration_sec=12.0)
        filename = audio_path.name
        source_folder: str | None = str(audio_dir)
        vector_dim = 64
        target_sample_rate = 16000
    else:
        filename = "router-fixture.wav"
        source_folder = None
        vector_dim = 1536
        target_sample_rate = 32000

    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            audio = AudioFile(
                filename=filename,
                folder_path="",
                source_folder=source_folder,
                checksum_sha256="router-fixture-sha",
            )
            model_config = ModelConfig(
                name="perch_v2_router_fixture",
                display_name="Perch v2 (router fixture)",
                path="/tmp/perch.tflite",
                vector_dim=vector_dim,
            )
            classifier = ClassifierModel(
                name="router-fixture-binary",
                model_path="/tmp/binary.joblib",
                model_version="perch_v2_router_fixture",
                vector_dim=vector_dim,
                window_size_seconds=5.0,
                target_sample_rate=target_sample_rate,
            )
            session.add_all([audio, model_config, classifier])
            await session.commit()
            return {
                "audio_file_id": audio.id,
                "model_config_id": model_config.id,
                "classifier_model_id": classifier.id,
            }
    finally:
        await engine.dispose()


def _patch_worker_ml(monkeypatch) -> None:
    """Replace the worker's model loaders with deterministic fakes.

    Mirrors the monkeypatch pattern in
    ``tests/integration/test_region_detection_worker.py`` so the router
    tests that run the worker synchronously don't need a real Perch
    TFLite binary or a real sklearn joblib file on disk.
    """
    pipeline = _synthetic_classifier()
    fake_perch = FakeTFLiteModel(vector_dim=64)

    def _fake_joblib_load(_path):
        return pipeline

    async def _fake_get_model_by_version(_session, _model_version, _settings):
        return fake_perch, "spectrogram"

    monkeypatch.setattr(
        "humpback.workers.region_detection_worker.joblib.load",
        _fake_joblib_load,
    )
    monkeypatch.setattr(
        "humpback.workers.region_detection_worker.get_model_by_version",
        _fake_get_model_by_version,
    )


async def _run_worker_once(app_settings) -> RegionDetectionJob | None:
    """Drive one iteration of the region-detection worker synchronously.

    Opens a fresh session against the same SQLite DB as the test client
    so the worker sees the job the API just created. Returns the claimed
    job (or ``None`` when the queue is empty) for assertions.
    """
    from humpback.workers.region_detection_worker import run_one_iteration

    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            return await run_one_iteration(session, app_settings)
    finally:
        await engine.dispose()


def _run_body(ids: dict[str, str], **overrides) -> dict:
    body = {
        "audio_file_id": ids["audio_file_id"],
        "model_config_id": ids["model_config_id"],
        "classifier_model_id": ids["classifier_model_id"],
    }
    body.update(overrides)
    return body


# ---- Segmentation fixture helpers ----------------------------------------


async def _seed_segmentation_training_dataset(
    app_settings,
) -> str:
    """Insert a ``SegmentationTrainingDataset`` and return its id."""
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            ds = SegmentationTrainingDataset(
                name="router-test-dataset",
                description="for router tests",
            )
            session.add(ds)
            await session.commit()
            return ds.id
    finally:
        await engine.dispose()


def _save_tiny_checkpoint(storage_root: Path) -> tuple[Path, dict]:
    """Build and save a tiny CRNN checkpoint. Returns (checkpoint_path, config_dict)."""
    model_config: dict = {
        "model_type": "SegmentationCRNN",
        "n_mels": 16,
        "conv_channels": [4],
        "gru_hidden": 4,
        "gru_layers": 1,
        "feature_config": SegmentationFeatureConfig(n_mels=16).model_dump(),
    }
    model = SegmentationCRNN(n_mels=16, conv_channels=[4], gru_hidden=4, gru_layers=1)
    with torch.no_grad():
        model.frame_head.bias.fill_(10.0)
    checkpoint_dir = ensure_dir(storage_root / "segmentation_models" / "tiny-rt")
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    save_checkpoint(
        path=checkpoint_path, model=model, optimizer=None, config=model_config
    )
    return checkpoint_path, model_config


async def _seed_segmentation_model(
    app_settings,
) -> tuple[str, Path]:
    """Insert a ``SegmentationModel`` row backed by a tiny checkpoint.

    Returns ``(model_id, checkpoint_dir)``.
    """
    checkpoint_path, model_config = _save_tiny_checkpoint(app_settings.storage_root)
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            sm = SegmentationModel(
                name="router-test-model",
                model_family="pytorch_crnn",
                model_path=str(checkpoint_path),
                config_json=json.dumps(model_config),
            )
            session.add(sm)
            await session.commit()
            return sm.id, checkpoint_path.parent
    finally:
        await engine.dispose()


async def _seed_complete_pass1_and_segmentation_model(
    app_settings,
) -> tuple[str, str, str]:
    """Insert AudioFile + completed RegionDetectionJob + SegmentationModel.

    Also writes a synthetic ``regions.parquet`` into the Pass 1 job dir
    and a tiny checkpoint for the model. Returns
    ``(upstream_job_id, seg_model_id, audio_file_id)``.
    """
    from humpback.call_parsing.storage import write_regions
    from humpback.call_parsing.types import Region

    storage_root = Path(app_settings.storage_root)
    audio_dir = storage_root / "audio_src"
    audio_path = audio_dir / "seg-fixture.wav"
    _write_sine_wav(audio_path, duration_sec=4.0)

    checkpoint_path, model_config = _save_tiny_checkpoint(storage_root)

    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            audio_file = AudioFile(
                filename="seg-fixture.wav",
                folder_path="",
                source_folder=str(audio_dir),
                checksum_sha256="seg-fixture-sha",
                duration_seconds=4.0,
            )
            session.add(audio_file)
            await session.flush()

            upstream = RegionDetectionJob(
                audio_file_id=audio_file.id,
                status="complete",
                config_json="{}",
            )
            session.add(upstream)
            await session.flush()

            # Write regions.parquet
            regions_dir = ensure_dir(region_job_dir(storage_root, upstream.id))
            write_regions(
                regions_dir / "regions.parquet",
                [
                    Region(
                        region_id="r-0",
                        start_sec=0.5,
                        end_sec=2.0,
                        padded_start_sec=0.0,
                        padded_end_sec=2.5,
                        max_score=0.9,
                        mean_score=0.7,
                        n_windows=3,
                    ),
                ],
            )

            sm = SegmentationModel(
                name="router-test-model",
                model_family="pytorch_crnn",
                model_path=str(checkpoint_path),
                config_json=json.dumps(model_config),
            )
            session.add(sm)
            await session.commit()
            return upstream.id, sm.id, audio_file.id
    finally:
        await engine.dispose()


# ---- Parent runs ---------------------------------------------------------


@pytest.mark.asyncio
async def test_create_run_creates_parent_and_pass1_job(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    resp = await client.post(f"{BASE}/runs", json=_run_body(ids))
    assert resp.status_code == 201
    data = resp.json()
    assert data["audio_file_id"] == ids["audio_file_id"]
    assert data["status"] == "queued"
    # The parent config snapshot is the serialized Pass 1 config.
    assert data["config_snapshot"] is not None
    assert '"padding_sec":1.0' in data["config_snapshot"]
    assert data["region_detection_job"] is not None
    assert data["region_detection_job"]["status"] == "queued"
    assert data["region_detection_job"]["parent_run_id"] == data["id"]
    assert data["region_detection_job"]["audio_file_id"] == ids["audio_file_id"]
    assert data["region_detection_job"]["model_config_id"] == ids["model_config_id"]
    assert (
        data["region_detection_job"]["classifier_model_id"]
        == ids["classifier_model_id"]
    )
    assert data["event_segmentation_job"] is None
    assert data["event_classification_job"] is None


@pytest.mark.asyncio
async def test_create_run_hydrophone_source(client: AsyncClient, app_settings) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = {
        "hydrophone_id": "rpi_north_sjc",
        "start_timestamp": 1_700_000_000.0,
        "end_timestamp": 1_700_086_400.0,
        "model_config_id": ids["model_config_id"],
        "classifier_model_id": ids["classifier_model_id"],
    }
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 201
    data = resp.json()
    assert data["audio_file_id"] is None
    assert data["hydrophone_id"] == "rpi_north_sjc"
    assert data["start_timestamp"] == 1_700_000_000.0
    assert data["end_timestamp"] == 1_700_086_400.0
    assert data["region_detection_job"]["hydrophone_id"] == "rpi_north_sjc"


@pytest.mark.asyncio
async def test_create_run_rejects_both_sources(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = _run_body(
        ids,
        hydrophone_id="rpi_north_sjc",
        start_timestamp=1.0,
        end_timestamp=2.0,
    )
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_run_rejects_neither_source(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = {
        "model_config_id": ids["model_config_id"],
        "classifier_model_id": ids["classifier_model_id"],
    }
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_run_rejects_inverted_timestamps(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = {
        "hydrophone_id": "rpi_north_sjc",
        "start_timestamp": 2.0,
        "end_timestamp": 1.0,
        "model_config_id": ids["model_config_id"],
        "classifier_model_id": ids["classifier_model_id"],
    }
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_run_404_on_missing_audio_file(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = _run_body(ids, audio_file_id="no-such-audio")
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 404
    assert "audio_file_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_run_404_on_missing_model_config(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = _run_body(ids, model_config_id="no-such-mc")
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 404
    assert "model_config_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_run_404_on_missing_classifier_model(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = _run_body(ids, classifier_model_id="no-such-cm")
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 404
    assert "classifier_model_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_run_404_on_missing_hydrophone(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = {
        "hydrophone_id": "not-a-registered-hydrophone",
        "start_timestamp": 1.0,
        "end_timestamp": 2.0,
        "model_config_id": ids["model_config_id"],
        "classifier_model_id": ids["classifier_model_id"],
    }
    resp = await client.post(f"{BASE}/runs", json=body)
    assert resp.status_code == 404
    assert "hydrophone_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_run_returns_nested_pass_status(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    create = await client.post(f"{BASE}/runs", json=_run_body(ids))
    run_id = create.json()["id"]

    resp = await client.get(f"{BASE}/runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == run_id
    assert data["region_detection_job"]["status"] == "queued"
    assert data["event_segmentation_job"] is None
    assert data["event_classification_job"] is None


@pytest.mark.asyncio
async def test_list_runs_returns_empty_then_populated(
    client: AsyncClient, app_settings
) -> None:
    empty = await client.get(f"{BASE}/runs")
    assert empty.status_code == 200
    assert empty.json() == []

    ids = await _seed_source_fixtures(app_settings)
    await client.post(f"{BASE}/runs", json=_run_body(ids))
    await client.post(f"{BASE}/runs", json=_run_body(ids))

    resp = await client.get(f"{BASE}/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_delete_run_cascades_children(client: AsyncClient, app_settings) -> None:
    ids = await _seed_source_fixtures(app_settings)
    create = await client.post(f"{BASE}/runs", json=_run_body(ids))
    run_id = create.json()["id"]
    region_id = create.json()["region_detection_job"]["id"]

    del_resp = await client.delete(f"{BASE}/runs/{run_id}")
    assert del_resp.status_code == 204

    missing_run = await client.get(f"{BASE}/runs/{run_id}")
    assert missing_run.status_code == 404

    missing_region = await client.get(f"{BASE}/region-jobs/{region_id}")
    assert missing_region.status_code == 404


@pytest.mark.asyncio
async def test_get_missing_run_returns_404(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/runs/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_region_jobs_empty(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/region-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_missing_region_job_returns_404(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/region-jobs/not-a-real-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_region_jobs_reflects_parent_run_creation(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    await client.post(f"{BASE}/runs", json=_run_body(ids))
    resp = await client.get(f"{BASE}/region-jobs")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_create_region_job_happy_path(client: AsyncClient, app_settings) -> None:
    ids = await _seed_source_fixtures(app_settings)
    resp = await client.post(f"{BASE}/region-jobs", json=_run_body(ids))
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["audio_file_id"] == ids["audio_file_id"]
    assert data["classifier_model_id"] == ids["classifier_model_id"]
    assert data["model_config_id"] == ids["model_config_id"]
    assert data["parent_run_id"] is None


@pytest.mark.asyncio
async def test_create_region_job_rejects_both_sources(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = _run_body(
        ids,
        hydrophone_id="rpi_north_sjc",
        start_timestamp=1.0,
        end_timestamp=2.0,
    )
    resp = await client.post(f"{BASE}/region-jobs", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_region_job_rejects_neither_source(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = {
        "model_config_id": ids["model_config_id"],
        "classifier_model_id": ids["classifier_model_id"],
    }
    resp = await client.post(f"{BASE}/region-jobs", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_region_job_rejects_inverted_timestamps(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = {
        "hydrophone_id": "rpi_north_sjc",
        "start_timestamp": 2.0,
        "end_timestamp": 1.0,
        "model_config_id": ids["model_config_id"],
        "classifier_model_id": ids["classifier_model_id"],
    }
    resp = await client.post(f"{BASE}/region-jobs", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_region_job_404_on_missing_audio_file(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = _run_body(ids, audio_file_id="no-such-audio")
    resp = await client.post(f"{BASE}/region-jobs", json=body)
    assert resp.status_code == 404
    assert "audio_file_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_region_job_404_on_missing_classifier_model(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    body = _run_body(ids, classifier_model_id="no-such-cm")
    resp = await client.post(f"{BASE}/region-jobs", json=body)
    assert resp.status_code == 404
    assert "classifier_model_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_region_trace_and_regions_return_409_while_queued(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    create = await client.post(f"{BASE}/region-jobs", json=_run_body(ids))
    job_id = create.json()["id"]

    trace_resp = await client.get(f"{BASE}/region-jobs/{job_id}/trace")
    regions_resp = await client.get(f"{BASE}/region-jobs/{job_id}/regions")
    assert trace_resp.status_code == 409
    assert regions_resp.status_code == 409


@pytest.mark.asyncio
async def test_region_trace_returns_404_on_missing_job(client: AsyncClient) -> None:
    trace_resp = await client.get(f"{BASE}/region-jobs/not-a-real-id/trace")
    regions_resp = await client.get(f"{BASE}/region-jobs/not-a-real-id/regions")
    assert trace_resp.status_code == 404
    assert regions_resp.status_code == 404


@pytest.mark.asyncio
async def test_region_trace_and_regions_happy_path_after_worker(
    client: AsyncClient, app_settings, monkeypatch
) -> None:
    ids = await _seed_source_fixtures(app_settings, with_real_audio=True)
    create = await client.post(f"{BASE}/region-jobs", json=_run_body(ids))
    assert create.status_code == 201
    job_id = create.json()["id"]

    _patch_worker_ml(monkeypatch)
    claimed = await _run_worker_once(app_settings)
    assert claimed is not None
    assert claimed.id == job_id

    detail_resp = await client.get(f"{BASE}/region-jobs/{job_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "complete"

    trace_resp = await client.get(f"{BASE}/region-jobs/{job_id}/trace")
    assert trace_resp.status_code == 200
    trace_rows = trace_resp.json()
    assert isinstance(trace_rows, list)
    assert len(trace_rows) > 0
    first = trace_rows[0]
    assert set(first.keys()) == {"time_sec", "score"}

    regions_resp = await client.get(f"{BASE}/region-jobs/{job_id}/regions")
    assert regions_resp.status_code == 200
    regions = regions_resp.json()
    assert len(regions) >= 1
    for r in regions:
        assert set(r.keys()) == {
            "region_id",
            "start_sec",
            "end_sec",
            "padded_start_sec",
            "padded_end_sec",
            "max_score",
            "mean_score",
            "n_windows",
        }
        assert r["start_sec"] <= r["end_sec"]
        assert 0.0 <= r["padded_start_sec"] <= r["padded_end_sec"] <= 12.0
    # Sorted by start_sec.
    starts = [r["start_sec"] for r in regions]
    assert starts == sorted(starts)


@pytest.mark.asyncio
async def test_delete_region_job_removes_parquet_dir(
    client: AsyncClient, app_settings, monkeypatch
) -> None:
    ids = await _seed_source_fixtures(app_settings, with_real_audio=True)
    create = await client.post(f"{BASE}/region-jobs", json=_run_body(ids))
    job_id = create.json()["id"]

    _patch_worker_ml(monkeypatch)
    await _run_worker_once(app_settings)

    job_dir = region_job_dir(app_settings.storage_root, job_id)
    assert job_dir.exists()
    assert (job_dir / "trace.parquet").exists()

    del_resp = await client.delete(f"{BASE}/region-jobs/{job_id}")
    assert del_resp.status_code == 204
    assert not job_dir.exists()

    missing = await client.get(f"{BASE}/region-jobs/{job_id}")
    assert missing.status_code == 404


# ---- Region tile endpoint ------------------------------------------------


@pytest.mark.asyncio
async def test_region_tile_404_for_missing_job(client: AsyncClient) -> None:
    resp = await client.get(
        f"{BASE}/region-jobs/no-such-id/tile",
        params={"zoom_level": "5m", "tile_index": 0},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_region_tile_409_for_non_complete_job(
    client: AsyncClient, app_settings
) -> None:
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            job = RegionDetectionJob(
                status="queued",
                hydrophone_id="test-hydro",
                start_timestamp=1_000_000.0,
                end_timestamp=1_000_300.0,
                config_json="{}",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
    finally:
        await engine.dispose()

    resp = await client.get(
        f"{BASE}/region-jobs/{job_id}/tile",
        params={"zoom_level": "5m", "tile_index": 0},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_region_tile_returns_png(
    client: AsyncClient, app_settings, monkeypatch
) -> None:
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            job = RegionDetectionJob(
                status="complete",
                hydrophone_id="test-hydro",
                start_timestamp=1_000_000.0,
                end_timestamp=1_000_300.0,
                config_json="{}",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
    finally:
        await engine.dispose()

    # Mock resolve_timeline_audio to return a sine wave
    def _fake_resolve(**kwargs):
        sr = kwargs.get("target_sr", 32000)
        duration = kwargs.get("duration_sec", 50.0)
        n = int(sr * duration)
        t = np.arange(n) / sr
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    monkeypatch.setattr(
        "humpback.processing.timeline_audio.resolve_timeline_audio",
        _fake_resolve,
    )

    resp = await client.get(
        f"{BASE}/region-jobs/{job_id}/tile",
        params={"zoom_level": "5m", "tile_index": 0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # PNG magic bytes
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_region_tile_rejects_invalid_zoom(
    client: AsyncClient, app_settings
) -> None:
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            job = RegionDetectionJob(
                status="complete",
                hydrophone_id="test-hydro",
                start_timestamp=1_000_000.0,
                end_timestamp=1_000_300.0,
                config_json="{}",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
    finally:
        await engine.dispose()

    resp = await client.get(
        f"{BASE}/region-jobs/{job_id}/tile",
        params={"zoom_level": "99h", "tile_index": 0},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_region_tile_rejects_out_of_range_index(
    client: AsyncClient, app_settings
) -> None:
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            job = RegionDetectionJob(
                status="complete",
                hydrophone_id="test-hydro",
                start_timestamp=1_000_000.0,
                end_timestamp=1_000_300.0,  # 300s = 6 tiles at 5m (50s/tile)
                config_json="{}",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
    finally:
        await engine.dispose()

    resp = await client.get(
        f"{BASE}/region-jobs/{job_id}/tile",
        params={"zoom_level": "5m", "tile_index": 100},
    )
    assert resp.status_code == 400


# ---- Region audio slice ---------------------------------------------------


@pytest.mark.asyncio
async def test_region_audio_slice_404_for_missing_job(client: AsyncClient) -> None:
    resp = await client.get(
        f"{BASE}/region-jobs/no-such-id/audio-slice",
        params={"start_sec": 0, "duration_sec": 2},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_region_audio_slice_409_for_non_complete_job(
    client: AsyncClient, app_settings
) -> None:
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            job = RegionDetectionJob(
                status="queued",
                hydrophone_id="test-hydro",
                start_timestamp=1_000_000.0,
                end_timestamp=1_000_300.0,
                config_json="{}",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
    finally:
        await engine.dispose()

    resp = await client.get(
        f"{BASE}/region-jobs/{job_id}/audio-slice",
        params={"start_sec": 0, "duration_sec": 2},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_region_audio_slice_returns_wav(
    client: AsyncClient, app_settings, monkeypatch
) -> None:
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            job = RegionDetectionJob(
                status="complete",
                hydrophone_id="test-hydro",
                start_timestamp=1_000_000.0,
                end_timestamp=1_000_300.0,
                config_json="{}",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
    finally:
        await engine.dispose()

    def _fake_resolve(**kwargs):
        sr = kwargs.get("target_sr", 16000)
        duration = kwargs.get("duration_sec", 2.0)
        n = int(sr * duration)
        t = np.arange(n) / sr
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    monkeypatch.setattr(
        "humpback.processing.timeline_audio.resolve_timeline_audio",
        _fake_resolve,
    )

    resp = await client.get(
        f"{BASE}/region-jobs/{job_id}/audio-slice",
        params={"start_sec": 10, "duration_sec": 2},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    # WAV RIFF header
    assert resp.content[:4] == b"RIFF"
    assert resp.content[8:12] == b"WAVE"


# ---- Pass 2 segmentation jobs (creation + events) -----------------------


@pytest.mark.asyncio
async def test_post_segmentation_job_happy_path(
    client: AsyncClient, app_settings
) -> None:
    upstream_id, model_id, _ = await _seed_complete_pass1_and_segmentation_model(
        app_settings
    )
    resp = await client.post(
        f"{BASE}/segmentation-jobs",
        json={
            "region_detection_job_id": upstream_id,
            "segmentation_model_id": model_id,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["region_detection_job_id"] == upstream_id
    assert data["segmentation_model_id"] == model_id


@pytest.mark.asyncio
async def test_post_segmentation_job_404_on_unknown_fk(
    client: AsyncClient, app_settings
) -> None:
    upstream_id, model_id, _ = await _seed_complete_pass1_and_segmentation_model(
        app_settings
    )
    resp_bad_upstream = await client.post(
        f"{BASE}/segmentation-jobs",
        json={
            "region_detection_job_id": "no-such-id",
            "segmentation_model_id": model_id,
        },
    )
    assert resp_bad_upstream.status_code == 404
    assert "region_detection_job_id" in resp_bad_upstream.json()["detail"]

    resp_bad_model = await client.post(
        f"{BASE}/segmentation-jobs",
        json={
            "region_detection_job_id": upstream_id,
            "segmentation_model_id": "no-such-model",
        },
    )
    assert resp_bad_model.status_code == 404
    assert "segmentation_model_id" in resp_bad_model.json()["detail"]


@pytest.mark.asyncio
async def test_post_segmentation_job_409_when_upstream_not_complete(
    client: AsyncClient, app_settings
) -> None:
    _, model_id, _ = await _seed_complete_pass1_and_segmentation_model(app_settings)
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            queued_upstream = RegionDetectionJob(
                audio_file_id="dummy", status="queued", config_json="{}"
            )
            session.add(queued_upstream)
            await session.commit()
            queued_id = queued_upstream.id
    finally:
        await engine.dispose()

    resp = await client.post(
        f"{BASE}/segmentation-jobs",
        json={
            "region_detection_job_id": queued_id,
            "segmentation_model_id": model_id,
        },
    )
    assert resp.status_code == 409
    assert "not 'complete'" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_segmentation_events_409_while_not_complete(
    client: AsyncClient, app_settings
) -> None:
    upstream_id, model_id, _ = await _seed_complete_pass1_and_segmentation_model(
        app_settings
    )
    create = await client.post(
        f"{BASE}/segmentation-jobs",
        json={
            "region_detection_job_id": upstream_id,
            "segmentation_model_id": model_id,
        },
    )
    job_id = create.json()["id"]
    resp = await client.get(f"{BASE}/segmentation-jobs/{job_id}/events")
    assert resp.status_code == 409
    assert "not 'complete'" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_segmentation_events_happy_path_after_worker(
    client: AsyncClient, app_settings
) -> None:
    upstream_id, model_id, _ = await _seed_complete_pass1_and_segmentation_model(
        app_settings
    )
    create = await client.post(
        f"{BASE}/segmentation-jobs",
        json={
            "region_detection_job_id": upstream_id,
            "segmentation_model_id": model_id,
        },
    )
    assert create.status_code == 201
    job_id = create.json()["id"]

    from humpback.workers.event_segmentation_worker import (
        run_one_iteration as seg_run,
    )

    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            claimed = await seg_run(session, app_settings)
    finally:
        await engine.dispose()
    assert claimed is not None
    assert claimed.id == job_id

    detail_resp = await client.get(f"{BASE}/segmentation-jobs/{job_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "complete"

    events_resp = await client.get(f"{BASE}/segmentation-jobs/{job_id}/events")
    assert events_resp.status_code == 200
    events = events_resp.json()
    assert isinstance(events, list)
    assert len(events) >= 1
    for e in events:
        assert set(e.keys()) == {
            "event_id",
            "region_id",
            "start_sec",
            "end_sec",
            "center_sec",
            "segmentation_confidence",
        }
        assert e["start_sec"] <= e["end_sec"]


# ---- Pass 2 segmentation training datasets -------------------------------


async def _seed_training_samples(app_settings, dataset_id: str, count: int) -> None:
    """Insert ``count`` dummy training samples for the given dataset."""
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            for _ in range(count):
                session.add(
                    SegmentationTrainingSample(
                        training_dataset_id=dataset_id,
                        audio_file_id="fake-audio",
                        crop_start_sec=0.0,
                        crop_end_sec=10.0,
                        events_json="[]",
                        source="test",
                    )
                )
            await session.commit()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_list_segmentation_training_datasets_empty(
    client: AsyncClient,
) -> None:
    resp = await client.get(f"{BASE}/segmentation-training-datasets")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_segmentation_training_datasets_with_counts(
    client: AsyncClient, app_settings
) -> None:
    ds_id = await _seed_segmentation_training_dataset(app_settings)
    await _seed_training_samples(app_settings, ds_id, 5)

    resp = await client.get(f"{BASE}/segmentation-training-datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    match = next(d for d in data if d["id"] == ds_id)
    assert match["name"] == "router-test-dataset"
    assert match["sample_count"] == 5
    assert "created_at" in match


@pytest.mark.asyncio
async def test_list_segmentation_training_datasets_zero_samples(
    client: AsyncClient, app_settings
) -> None:
    ds_id = await _seed_segmentation_training_dataset(app_settings)

    resp = await client.get(f"{BASE}/segmentation-training-datasets")
    assert resp.status_code == 200
    data = resp.json()
    match = next(d for d in data if d["id"] == ds_id)
    assert match["sample_count"] == 0


# ---- Segmentation jobs with correction counts ----------------------------


@pytest.mark.asyncio
async def test_segmentation_jobs_with_correction_counts(
    client: AsyncClient, app_settings
) -> None:
    """Jobs with and without corrections both appear, with correct counts."""
    from humpback.models.feedback_training import EventBoundaryCorrection

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(
            hydrophone_id="orcasound_lab",
            start_timestamp=1000.0,
            end_timestamp=2000.0,
            status="complete",
            config_json="{}",
        )
        session.add(rd)
        await session.flush()

        es1 = EventSegmentationJob(
            region_detection_job_id=rd.id, status="complete", event_count=5
        )
        es2 = EventSegmentationJob(
            region_detection_job_id=rd.id, status="complete", event_count=3
        )
        session.add_all([es1, es2])
        await session.flush()

        # Add corrections only for es1
        for i in range(3):
            session.add(
                EventBoundaryCorrection(
                    event_segmentation_job_id=es1.id,
                    event_id=f"e{i}",
                    region_id="r1",
                    correction_type="adjust",
                    start_sec=float(i),
                    end_sec=float(i + 1),
                )
            )
        await session.commit()
        es1_id, es2_id = es1.id, es2.id
    await engine.dispose()

    resp = await client.get(f"{BASE}/segmentation-jobs/with-correction-counts")
    assert resp.status_code == 200
    data = resp.json()

    job_map = {j["id"]: j for j in data}
    assert es1_id in job_map
    assert es2_id in job_map
    assert job_map[es1_id]["correction_count"] == 3
    assert job_map[es2_id]["correction_count"] == 0
    assert job_map[es1_id]["hydrophone_id"] == "orcasound_lab"
    assert job_map[es1_id]["start_timestamp"] == 1000.0


@pytest.mark.asyncio
async def test_segmentation_jobs_with_correction_counts_excludes_incomplete(
    client: AsyncClient, app_settings
) -> None:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="running")
        session.add(es)
        await session.commit()
        es_id = es.id
    await engine.dispose()

    resp = await client.get(f"{BASE}/segmentation-jobs/with-correction-counts")
    assert resp.status_code == 200
    job_ids = [j["id"] for j in resp.json()]
    assert es_id not in job_ids


# ---- Pass 2 segmentation models -----------------------------------------


@pytest.mark.asyncio
async def test_delete_segmentation_model_happy_path(
    client: AsyncClient, app_settings
) -> None:
    model_id, checkpoint_dir = await _seed_segmentation_model(app_settings)
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "checkpoint.pt").exists()

    resp = await client.delete(f"{BASE}/segmentation-models/{model_id}")
    assert resp.status_code == 204

    assert not checkpoint_dir.exists()

    get_resp = await client.get(f"{BASE}/segmentation-models/{model_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_segmentation_model_409_when_referenced_by_in_flight_job(
    client: AsyncClient, app_settings
) -> None:
    upstream_id, model_id, _ = await _seed_complete_pass1_and_segmentation_model(
        app_settings
    )
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            in_flight_job = EventSegmentationJob(
                region_detection_job_id=upstream_id,
                segmentation_model_id=model_id,
                status="running",
                config_json=SegmentationDecoderConfig().model_dump_json(),
            )
            session.add(in_flight_job)
            await session.commit()
    finally:
        await engine.dispose()

    resp = await client.delete(f"{BASE}/segmentation-models/{model_id}")
    assert resp.status_code == 409
    assert "in-flight" in resp.json()["detail"]


# ---- Pass 3 — event classification endpoints ----------------------------


@pytest.mark.asyncio
async def test_post_classification_jobs_missing_fields_returns_422(
    client: AsyncClient,
) -> None:
    resp = await client.post(f"{BASE}/classification-jobs", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_classification_jobs_missing_model_returns_404(
    client: AsyncClient, app_settings
) -> None:
    resp = await client.post(
        f"{BASE}/classification-jobs",
        json={
            "event_segmentation_job_id": "nonexistent",
            "vocalization_model_id": "nonexistent",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_post_classification_jobs_wrong_model_family_returns_422(
    client: AsyncClient, app_settings
) -> None:
    """Model with sklearn family is rejected with 422."""
    from humpback.models.vocalization import VocalizationClassifierModel

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="a1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
        session.add(es)
        await session.flush()
        vm = VocalizationClassifierModel(
            name="sklearn-model",
            model_dir_path="/tmp/fake",
            vocabulary_snapshot="[]",
            per_class_thresholds="{}",
            model_family="sklearn_perch_embedding",
            input_mode="detection_row",
        )
        session.add(vm)
        await session.commit()
        es_id, vm_id = es.id, vm.id
    await engine.dispose()

    resp = await client.post(
        f"{BASE}/classification-jobs",
        json={
            "event_segmentation_job_id": es_id,
            "vocalization_model_id": vm_id,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_classification_jobs_upstream_not_complete_returns_409(
    client: AsyncClient, app_settings
) -> None:
    """Non-complete upstream segmentation job is rejected with 409."""
    from humpback.models.vocalization import VocalizationClassifierModel

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="a1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="running")
        session.add(es)
        await session.flush()
        vm = VocalizationClassifierModel(
            name="cnn-model",
            model_dir_path="/tmp/fake",
            vocabulary_snapshot="[]",
            per_class_thresholds="{}",
            model_family="pytorch_event_cnn",
            input_mode="segmented_event",
        )
        session.add(vm)
        await session.commit()
        es_id, vm_id = es.id, vm.id
    await engine.dispose()

    resp = await client.post(
        f"{BASE}/classification-jobs",
        json={
            "event_segmentation_job_id": es_id,
            "vocalization_model_id": vm_id,
        },
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_post_classification_jobs_valid_returns_200(
    client: AsyncClient, app_settings
) -> None:
    """Valid inputs create a queued classification job."""
    from humpback.models.vocalization import VocalizationClassifierModel

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="a1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
        session.add(es)
        await session.flush()
        vm = VocalizationClassifierModel(
            name="cnn-model",
            model_dir_path="/tmp/fake",
            vocabulary_snapshot="[]",
            per_class_thresholds="{}",
            model_family="pytorch_event_cnn",
            input_mode="segmented_event",
        )
        session.add(vm)
        await session.commit()
        es_id, vm_id = es.id, vm.id
    await engine.dispose()

    resp = await client.post(
        f"{BASE}/classification-jobs",
        json={
            "event_segmentation_job_id": es_id,
            "vocalization_model_id": vm_id,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["event_segmentation_job_id"] == es_id
    assert data["vocalization_model_id"] == vm_id


@pytest.mark.asyncio
async def test_get_typed_events_on_complete_job_returns_sorted(
    client: AsyncClient, app_settings
) -> None:
    """GET typed-events on a complete job returns rows sorted by start_sec."""
    from humpback.call_parsing.storage import classification_job_dir, write_typed_events
    from humpback.call_parsing.types import TypedEvent

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="a1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
        session.add(es)
        await session.flush()
        from humpback.models.call_parsing import EventClassificationJob

        ec = EventClassificationJob(
            event_segmentation_job_id=es.id,
            vocalization_model_id="vm-fake",
            status="complete",
            typed_event_count=3,
        )
        session.add(ec)
        await session.commit()
        ec_id = ec.id
    await engine.dispose()

    # Write typed_events.parquet (out of order to test sorting)
    job_dir = classification_job_dir(app_settings.storage_root, ec_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(
        job_dir / "typed_events.parquet",
        [
            TypedEvent("e2", 5.0, 7.0, "moan", 0.8, True),
            TypedEvent("e1", 1.0, 3.0, "upcall", 0.9, True),
            TypedEvent("e1", 1.0, 3.0, "moan", 0.3, False),
        ],
    )

    resp = await client.get(f"{BASE}/classification-jobs/{ec_id}/typed-events")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 3
    assert rows[0]["start_sec"] <= rows[1]["start_sec"] <= rows[2]["start_sec"]
    assert all(0.0 <= r["score"] <= 1.0 for r in rows)
    assert all("type_name" in r for r in rows)


@pytest.mark.asyncio
async def test_sequence_endpoint_returns_501_naming_pass4(
    client: AsyncClient, app_settings
) -> None:
    ids = await _seed_source_fixtures(app_settings)
    create = await client.post(f"{BASE}/runs", json=_run_body(ids))
    run_id = create.json()["id"]
    resp = await client.get(f"{BASE}/runs/{run_id}/sequence")
    assert resp.status_code == 501
    assert "Pass 4" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_classification_typed_events_not_complete_returns_409(
    client: AsyncClient, app_settings
) -> None:
    from humpback.database import create_engine, create_session_factory
    from humpback.models.call_parsing import (
        EventClassificationJob,
        EventSegmentationJob,
        RegionDetectionJob,
    )

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="a1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
        session.add(es)
        await session.flush()
        ec = EventClassificationJob(event_segmentation_job_id=es.id, status="running")
        session.add(ec)
        await session.commit()
        ec_id = ec.id
    await engine.dispose()

    resp = await client.get(f"{BASE}/classification-jobs/{ec_id}/typed-events")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_classification_typed_events_nonexistent_returns_404(
    client: AsyncClient,
) -> None:
    resp = await client.get(f"{BASE}/classification-jobs/nonexistent/typed-events")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_segmentation_and_classification_lists_empty(
    client: AsyncClient,
) -> None:
    s = await client.get(f"{BASE}/segmentation-jobs")
    c = await client.get(f"{BASE}/classification-jobs")
    assert s.status_code == 200 and s.json() == []
    assert c.status_code == 200 and c.json() == []


# ---- Boundary corrections (Task 5) ----------------------------------------


async def _seed_complete_segmentation_job(app_settings) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
        session.add(es)
        await session.commit()
        es_id = es.id
    await engine.dispose()
    return es_id


async def _seed_complete_classification_job(app_settings) -> str:
    from humpback.models.call_parsing import EventClassificationJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
        session.add(es)
        await session.flush()
        ec = EventClassificationJob(event_segmentation_job_id=es.id, status="complete")
        session.add(ec)
        await session.commit()
        ec_id = ec.id
    await engine.dispose()
    return ec_id


@pytest.mark.asyncio
async def test_boundary_corrections_post_happy(
    client: AsyncClient, app_settings
) -> None:
    es_id = await _seed_complete_segmentation_job(app_settings)
    resp = await client.post(
        f"{BASE}/segmentation-jobs/{es_id}/corrections",
        json={
            "corrections": [
                {
                    "event_id": "e1",
                    "region_id": "r1",
                    "correction_type": "adjust",
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                }
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


@pytest.mark.asyncio
async def test_boundary_corrections_post_404(client: AsyncClient) -> None:
    resp = await client.post(
        f"{BASE}/segmentation-jobs/nonexistent/corrections",
        json={
            "corrections": [
                {
                    "event_id": "e1",
                    "region_id": "r1",
                    "correction_type": "delete",
                }
            ]
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_boundary_corrections_get_and_delete(
    client: AsyncClient, app_settings
) -> None:
    es_id = await _seed_complete_segmentation_job(app_settings)
    await client.post(
        f"{BASE}/segmentation-jobs/{es_id}/corrections",
        json={
            "corrections": [
                {
                    "event_id": "e1",
                    "region_id": "r1",
                    "correction_type": "add",
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                }
            ]
        },
    )

    get_resp = await client.get(f"{BASE}/segmentation-jobs/{es_id}/corrections")
    assert get_resp.status_code == 200
    assert len(get_resp.json()) == 1

    del_resp = await client.delete(f"{BASE}/segmentation-jobs/{es_id}/corrections")
    assert del_resp.status_code == 204

    get_resp2 = await client.get(f"{BASE}/segmentation-jobs/{es_id}/corrections")
    assert get_resp2.json() == []


# ---- Type corrections (Task 5) --------------------------------------------


@pytest.mark.asyncio
async def test_type_corrections_post_happy(client: AsyncClient, app_settings) -> None:
    ec_id = await _seed_complete_classification_job(app_settings)
    resp = await client.post(
        f"{BASE}/classification-jobs/{ec_id}/corrections",
        json={"corrections": [{"event_id": "e1", "type_name": "upcall"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


@pytest.mark.asyncio
async def test_type_corrections_post_404(client: AsyncClient) -> None:
    resp = await client.post(
        f"{BASE}/classification-jobs/nonexistent/corrections",
        json={"corrections": [{"event_id": "e1", "type_name": "upcall"}]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_type_corrections_overwrite_on_repeat(
    client: AsyncClient, app_settings
) -> None:
    ec_id = await _seed_complete_classification_job(app_settings)
    await client.post(
        f"{BASE}/classification-jobs/{ec_id}/corrections",
        json={"corrections": [{"event_id": "e1", "type_name": "upcall"}]},
    )
    await client.post(
        f"{BASE}/classification-jobs/{ec_id}/corrections",
        json={"corrections": [{"event_id": "e1", "type_name": "moan"}]},
    )
    get_resp = await client.get(f"{BASE}/classification-jobs/{ec_id}/corrections")
    data = get_resp.json()
    assert len(data) == 1
    assert data[0]["type_name"] == "moan"


@pytest.mark.asyncio
async def test_type_corrections_delete(client: AsyncClient, app_settings) -> None:
    ec_id = await _seed_complete_classification_job(app_settings)
    await client.post(
        f"{BASE}/classification-jobs/{ec_id}/corrections",
        json={"corrections": [{"event_id": "e1", "type_name": "upcall"}]},
    )
    del_resp = await client.delete(f"{BASE}/classification-jobs/{ec_id}/corrections")
    assert del_resp.status_code == 204


# ---- Classifier feedback training jobs (Task 6) ----------------------------


@pytest.mark.asyncio
async def test_classifier_training_crud(client: AsyncClient, app_settings) -> None:
    ec_id = await _seed_complete_classification_job(app_settings)

    post_resp = await client.post(
        f"{BASE}/classifier-training-jobs",
        json={"source_job_ids": [ec_id]},
    )
    assert post_resp.status_code == 201
    job_id = post_resp.json()["id"]
    assert post_resp.json()["status"] == "queued"

    list_resp = await client.get(f"{BASE}/classifier-training-jobs")
    assert list_resp.status_code == 200
    assert any(j["id"] == job_id for j in list_resp.json())

    get_resp = await client.get(f"{BASE}/classifier-training-jobs/{job_id}")
    assert get_resp.status_code == 200

    del_resp = await client.delete(f"{BASE}/classifier-training-jobs/{job_id}")
    assert del_resp.status_code == 204


@pytest.mark.asyncio
async def test_classifier_training_404_missing_source(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        f"{BASE}/classifier-training-jobs",
        json={"source_job_ids": ["nonexistent"]},
    )
    assert resp.status_code == 404


# ---- Classifier model management (Task 6) ---------------------------------


@pytest.mark.asyncio
async def test_classifier_models_list(client: AsyncClient, app_settings) -> None:
    from humpback.models.vocalization import VocalizationClassifierModel

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        session.add(
            VocalizationClassifierModel(
                name="event-cnn",
                model_dir_path="/tmp/event",
                vocabulary_snapshot="[]",
                per_class_thresholds="{}",
                model_family="pytorch_event_cnn",
                input_mode="segmented_event",
            )
        )
        session.add(
            VocalizationClassifierModel(
                name="sklearn-model",
                model_dir_path="/tmp/sklearn",
                vocabulary_snapshot="[]",
                per_class_thresholds="{}",
                model_family="sklearn_perch_embedding",
            )
        )
        await session.commit()
    await engine.dispose()

    resp = await client.get(f"{BASE}/classifier-models")
    assert resp.status_code == 200
    data = resp.json()
    assert all(m["model_family"] == "pytorch_event_cnn" for m in data)
    assert len(data) >= 1


@pytest.mark.asyncio
async def test_classifier_model_delete_404(client: AsyncClient) -> None:
    resp = await client.delete(f"{BASE}/classifier-models/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_classifier_model_delete_409_in_flight(
    client: AsyncClient, app_settings
) -> None:
    from humpback.models.call_parsing import EventClassificationJob
    from humpback.models.vocalization import VocalizationClassifierModel

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        vm = VocalizationClassifierModel(
            name="event-cnn",
            model_dir_path="/tmp/event",
            vocabulary_snapshot="[]",
            per_class_thresholds="{}",
            model_family="pytorch_event_cnn",
            input_mode="segmented_event",
        )
        session.add(vm)
        await session.flush()
        rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
        session.add(rd)
        await session.flush()
        es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
        session.add(es)
        await session.flush()
        ec = EventClassificationJob(
            event_segmentation_job_id=es.id,
            vocalization_model_id=vm.id,
            status="running",
        )
        session.add(ec)
        await session.commit()
        vm_id = vm.id
    await engine.dispose()

    resp = await client.delete(f"{BASE}/classifier-models/{vm_id}")
    assert resp.status_code == 409
