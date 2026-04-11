"""Integration tests for the call parsing API router."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
from httpx import AsyncClient
from sklearn.pipeline import Pipeline

from humpback.call_parsing.storage import region_job_dir
from humpback.classifier.trainer import train_binary_classifier
from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.classifier import ClassifierModel
from humpback.models.model_registry import ModelConfig
from humpback.processing.inference import FakeTFLiteModel

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


@pytest.mark.asyncio
async def test_post_segmentation_jobs_returns_501_naming_pass2(
    client: AsyncClient,
) -> None:
    resp = await client.post(f"{BASE}/segmentation-jobs", json={})
    assert resp.status_code == 501
    assert "Pass 2" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_post_classification_jobs_returns_501_naming_pass3(
    client: AsyncClient,
) -> None:
    resp = await client.post(f"{BASE}/classification-jobs", json={})
    assert resp.status_code == 501
    assert "Pass 3" in resp.json()["detail"]


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
async def test_segmentation_events_endpoint_is_501(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/segmentation-jobs/anything/events")
    assert resp.status_code == 501
    assert "Pass 2" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_classification_typed_events_endpoint_is_501(
    client: AsyncClient,
) -> None:
    resp = await client.get(f"{BASE}/classification-jobs/anything/typed-events")
    assert resp.status_code == 501
    assert "Pass 3" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_segmentation_and_classification_lists_empty(
    client: AsyncClient,
) -> None:
    s = await client.get(f"{BASE}/segmentation-jobs")
    c = await client.get(f"{BASE}/classification-jobs")
    assert s.status_code == 200 and s.json() == []
    assert c.status_code == 200 and c.json() == []
