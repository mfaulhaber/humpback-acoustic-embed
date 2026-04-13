"""End-to-end smoke test: upload → process → cluster → verify."""

import io
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.pipeline import Pipeline

from humpback.api.app import create_app
from humpback.call_parsing.storage import region_job_dir
from humpback.classifier.trainer import train_binary_classifier
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.classifier import ClassifierModel
from humpback.models.model_registry import ModelConfig
from humpback.processing.inference import FakeTFLiteModel
from humpback.workers.classifier_worker import run_detection_job, run_training_job
from humpback.workers.clustering_worker import run_clustering_job
from humpback.workers.processing_worker import run_processing_job
from humpback.workers.queue import (
    claim_clustering_job,
    claim_detection_job,
    claim_processing_job,
    claim_training_job,
)
from humpback.workers.region_detection_worker import run_one_iteration


def make_wav_bytes(duration: float = 10.0, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * duration)
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


@pytest.fixture
def e2e_settings(tmp_path):
    db_path = tmp_path / "e2e.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
        vector_dim=64,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )


@pytest.fixture
async def e2e_client(e2e_settings):
    app = create_app(e2e_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await app.router.startup()
        yield ac
        await app.router.shutdown()


async def test_full_workflow(e2e_settings, e2e_client):
    """Full E2E: upload → process → cluster → verify."""
    client = e2e_client
    settings = e2e_settings

    # 1. Upload audio
    wav_data = make_wav_bytes(duration=10.0, sample_rate=16000)
    resp = await client.post(
        "/audio/upload",
        files={"file": ("whale_song.wav", wav_data, "audio/wav")},
    )
    assert resp.status_code == 201
    audio = resp.json()
    audio_id = audio["id"]
    assert audio["filename"] == "whale_song.wav"

    # 2. Create processing job
    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201
    job_data = resp.json()
    assert job_data["status"] == "queued"
    assert job_data["skipped"] is False
    job_id = job_data["id"]

    # 3. Run the processing job directly (deterministic, no polling)
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    model = FakeTFLiteModel(vector_dim=settings.vector_dim)

    async with session_factory() as session:
        claimed = await claim_processing_job(session)
        assert claimed is not None
        assert claimed.id == job_id
        await run_processing_job(session, claimed, settings, model)

    # 4. Verify job is complete
    resp = await client.get(f"/processing/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "complete"

    # 5. Verify embedding set exists
    resp = await client.get("/processing/embedding-sets")
    assert resp.status_code == 200
    es_list = resp.json()
    assert len(es_list) == 1
    es = es_list[0]
    assert es["audio_file_id"] == audio_id
    assert es["vector_dim"] == settings.vector_dim

    # Verify parquet is readable
    table = pq.read_table(es["parquet_path"])
    assert len(table) > 0
    assert "embedding" in table.column_names
    assert "row_index" in table.column_names

    # 6. Test idempotency: re-queue same config should be skipped
    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["skipped"] is True
    assert resp.json()["status"] == "complete"

    # 7. Create clustering job
    resp = await client.post(
        "/clustering/jobs",
        json={
            "embedding_set_ids": [es["id"]],
            "parameters": {"use_umap": True, "min_cluster_size": 2},
        },
    )
    assert resp.status_code == 201
    cjob_data = resp.json()
    cjob_id = cjob_data["id"]
    assert cjob_data["status"] == "queued"

    # 8. Run clustering job directly
    async with session_factory() as session:
        claimed = await claim_clustering_job(session)
        assert claimed is not None
        assert claimed.id == cjob_id
        await run_clustering_job(session, claimed, settings)

    # 9. Verify clustering job complete
    resp = await client.get(f"/clustering/jobs/{cjob_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "complete"

    # 10. Verify clusters exist
    resp = await client.get(f"/clustering/jobs/{cjob_id}/clusters")
    assert resp.status_code == 200
    clusters = resp.json()
    assert len(clusters) >= 1

    # Verify total assignments equal embedding count
    total_size = sum(c["size"] for c in clusters)
    assert total_size == len(table)

    # Verify cluster assignments are accessible
    for cluster in clusters:
        resp = await client.get(f"/clustering/clusters/{cluster['id']}/assignments")
        assert resp.status_code == 200
        assignments = resp.json()
        assert len(assignments) == cluster["size"]

    # Verify output files
    output_dir = settings.storage_root / "clusters" / cjob_id
    assert (output_dir / "clusters.json").exists()
    assert (output_dir / "assignments.parquet").exists()

    await engine.dispose()


async def test_classifier_workflow(e2e_settings, e2e_client, tmp_path):
    """E2E: process embeddings → train classifier → run detection → download TSV."""
    client = e2e_client
    settings = e2e_settings

    # 1. Upload and process audio (positive samples)
    wav_data = make_wav_bytes(duration=10.0, sample_rate=16000)
    resp = await client.post(
        "/audio/upload",
        files={"file": ("whale.wav", wav_data, "audio/wav")},
    )
    assert resp.status_code == 201
    audio_id = resp.json()["id"]

    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201
    resp.json()["id"]

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        claimed = await claim_processing_job(session)
        assert claimed is not None
        await run_processing_job(session, claimed, settings)

    resp = await client.get("/processing/embedding-sets")
    es_list = resp.json()
    assert len(es_list) >= 1
    es_id = es_list[0]["id"]

    # 2. Upload and process negative audio
    neg_wav_data = make_wav_bytes(duration=10.0, sample_rate=16000)
    resp = await client.post(
        "/audio/upload",
        files={"file": ("noise.wav", neg_wav_data, "audio/wav")},
        data={"folder_path": "negatives"},
    )
    assert resp.status_code == 201
    neg_audio_id = resp.json()["id"]

    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": neg_audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201

    async with session_factory() as session:
        claimed = await claim_processing_job(session)
        assert claimed is not None
        await run_processing_job(session, claimed, settings)

    resp = await client.get("/processing/embedding-sets")
    neg_es_list = resp.json()
    # Find the embedding set for the negative audio
    neg_es_id = next(
        es["id"] for es in neg_es_list if es["audio_file_id"] == neg_audio_id
    )

    # 3. Create training job
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "test-classifier",
            "positive_embedding_set_ids": [es_id],
            "negative_embedding_set_ids": [neg_es_id],
        },
    )
    assert resp.status_code == 201
    tjob_data = resp.json()
    tjob_id = tjob_data["id"]
    assert tjob_data["status"] == "queued"

    # 4. Run training job
    async with session_factory() as session:
        claimed = await claim_training_job(session)
        assert claimed is not None
        assert claimed.id == tjob_id
        await run_training_job(session, claimed, settings)

    # 5. Verify training complete
    resp = await client.get(f"/classifier/training-jobs/{tjob_id}")
    assert resp.status_code == 200
    tjob = resp.json()
    assert tjob["status"] == "complete"
    assert tjob["classifier_model_id"] is not None

    # 6. Verify classifier model exists
    resp = await client.get("/classifier/models")
    models = resp.json()
    assert len(models) >= 1
    model_id = models[0]["id"]
    assert models[0]["name"] == "test-classifier"
    assert models[0]["training_summary"] is not None

    # 7. Create detection job (scan the negative folder itself)
    detect_dir = tmp_path / "detect_audio"
    detect_dir.mkdir()
    (detect_dir / "test.wav").write_bytes(
        make_wav_bytes(duration=5.0, sample_rate=16000)
    )

    resp = await client.post(
        "/classifier/detection-jobs",
        json={
            "classifier_model_id": model_id,
            "audio_folder": str(detect_dir),
            "confidence_threshold": 0.5,
        },
    )
    assert resp.status_code == 201
    djob_id = resp.json()["id"]

    # 8. Run detection job
    async with session_factory() as session:
        claimed = await claim_detection_job(session)
        assert claimed is not None
        assert claimed.id == djob_id
        await run_detection_job(session, claimed, settings)

    # 9. Verify detection complete
    resp = await client.get(f"/classifier/detection-jobs/{djob_id}")
    assert resp.status_code == 200
    djob = resp.json()
    assert djob["status"] == "complete"
    assert djob["result_summary"] is not None
    assert djob["result_summary"]["n_files"] == 1

    # 10. Download TSV
    resp = await client.get(f"/classifier/detection-jobs/{djob_id}/download")
    assert resp.status_code == 200
    tsv_content = resp.text
    assert "filename" in tsv_content  # header row

    await engine.dispose()


async def test_short_audio_warns_no_embedding_set(e2e_settings, e2e_client):
    """Short audio (<window_size) completes with warning, no EmbeddingSet created."""
    client = e2e_client
    settings = e2e_settings

    # Upload a 2-second audio file (shorter than 5s window)
    wav_data = make_wav_bytes(duration=2.0, sample_rate=16000)
    resp = await client.post(
        "/audio/upload",
        files={"file": ("short.wav", wav_data, "audio/wav")},
    )
    assert resp.status_code == 201
    audio_id = resp.json()["id"]

    # Create processing job
    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    # Run processing job
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        claimed = await claim_processing_job(session)
        assert claimed is not None
        assert claimed.id == job_id
        await run_processing_job(session, claimed, settings)

    # Job should be complete with a warning
    resp = await client.get(f"/processing/jobs/{job_id}")
    assert resp.status_code == 200
    job = resp.json()
    assert job["status"] == "complete"
    assert job["warning_message"] is not None
    assert "too short" in job["warning_message"]
    assert f"samples @ {settings.target_sample_rate} Hz" in job["warning_message"]
    assert "5.0s < 5.0s" not in job["warning_message"]

    # No embedding set should have been created
    resp = await client.get("/processing/embedding-sets")
    assert resp.status_code == 200
    es_list = [es for es in resp.json() if es["audio_file_id"] == audio_id]
    assert len(es_list) == 0

    await engine.dispose()


def _write_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
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


async def test_call_parsing_pass1_smoke(
    e2e_settings, e2e_client, tmp_path, monkeypatch
):
    """E2E: create Pass 1 job → run worker → trace/regions endpoints → delete cleanup."""
    client = e2e_client
    settings = e2e_settings

    audio_dir = tmp_path / "call_parsing_audio"
    audio_path = audio_dir / "sample.wav"
    _write_wav(audio_path, duration_sec=12.0)

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        audio = AudioFile(
            filename=audio_path.name,
            folder_path="",
            source_folder=str(audio_dir),
            checksum_sha256="call-parsing-smoke-sha",
        )
        model_config = ModelConfig(
            name="perch_v2_smoke",
            display_name="Perch v2 (smoke)",
            path="/tmp/perch.tflite",
            vector_dim=64,
        )
        classifier = ClassifierModel(
            name="pass1-smoke-binary",
            model_path="/tmp/binary.joblib",
            model_version="perch_v2_smoke",
            vector_dim=64,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )
        session.add_all([audio, model_config, classifier])
        await session.commit()
        audio_id = audio.id
        model_config_id = model_config.id
        classifier_id = classifier.id

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

    resp = await client.post(
        "/call-parsing/region-jobs",
        json={
            "audio_file_id": audio_id,
            "model_config_id": model_config_id,
            "classifier_model_id": classifier_id,
        },
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]
    assert resp.json()["status"] == "queued"

    # Trace and regions are 409 while the job is still queued.
    queued_trace = await client.get(f"/call-parsing/region-jobs/{job_id}/trace")
    queued_regions = await client.get(f"/call-parsing/region-jobs/{job_id}/regions")
    assert queued_trace.status_code == 409
    assert queued_regions.status_code == 409

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None
    assert claimed.id == job_id

    detail = await client.get(f"/call-parsing/region-jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "complete"
    assert detail.json()["trace_row_count"] > 0
    assert detail.json()["region_count"] >= 1

    trace_resp = await client.get(f"/call-parsing/region-jobs/{job_id}/trace")
    assert trace_resp.status_code == 200
    trace_rows = trace_resp.json()
    assert len(trace_rows) > 0
    assert set(trace_rows[0].keys()) == {"time_sec", "score"}

    regions_resp = await client.get(f"/call-parsing/region-jobs/{job_id}/regions")
    assert regions_resp.status_code == 200
    regions = regions_resp.json()
    assert len(regions) >= 1
    for region in regions:
        assert 0.0 <= region["padded_start_sec"] <= region["padded_end_sec"] <= 12.0
        assert region["start_sec"] <= region["end_sec"]
        assert region["n_windows"] >= 1

    job_dir = region_job_dir(settings.storage_root, job_id)
    assert job_dir.exists()
    assert (job_dir / "trace.parquet").exists()
    assert (job_dir / "regions.parquet").exists()

    delete_resp = await client.delete(f"/call-parsing/region-jobs/{job_id}")
    assert delete_resp.status_code == 204
    assert not job_dir.exists()

    missing = await client.get(f"/call-parsing/region-jobs/{job_id}")
    assert missing.status_code == 404

    await engine.dispose()


async def test_call_parsing_pass2_smoke(
    e2e_settings, e2e_client, tmp_path, monkeypatch
):
    """E2E: Pass 1 → train segmentation model → Pass 2 inference → events endpoint → delete cleanup."""
    import json

    from humpback.call_parsing.storage import segmentation_job_dir
    from humpback.models.segmentation_training import (
        SegmentationTrainingDataset,
        SegmentationTrainingSample,
    )
    from humpback.workers.event_segmentation_worker import (
        run_one_iteration as seg_run_one_iteration,
    )

    client = e2e_client
    settings = e2e_settings

    # ---- Step 1: Run Pass 1 to produce regions.parquet ----

    audio_dir = tmp_path / "call_parsing_audio"
    audio_path = audio_dir / "sample.wav"
    _write_wav(audio_path, duration_sec=12.0)

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        audio = AudioFile(
            filename=audio_path.name,
            folder_path="",
            source_folder=str(audio_dir),
            checksum_sha256="pass2-smoke-sha",
            duration_seconds=12.0,
        )
        model_config = ModelConfig(
            name="perch_v2_pass2_smoke",
            display_name="Perch v2 (pass2 smoke)",
            path="/tmp/perch.tflite",
            vector_dim=64,
        )
        classifier = ClassifierModel(
            name="pass2-smoke-binary",
            model_path="/tmp/binary.joblib",
            model_version="perch_v2_pass2_smoke",
            vector_dim=64,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )
        session.add_all([audio, model_config, classifier])
        await session.commit()
        audio_id = audio.id
        model_config_id = model_config.id
        classifier_id = classifier.id

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

    resp = await client.post(
        "/call-parsing/region-jobs",
        json={
            "audio_file_id": audio_id,
            "model_config_id": model_config_id,
            "classifier_model_id": classifier_id,
        },
    )
    assert resp.status_code == 201
    pass1_job_id = resp.json()["id"]

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None
    assert claimed.id == pass1_job_id

    detail = await client.get(f"/call-parsing/region-jobs/{pass1_job_id}")
    assert detail.json()["status"] == "complete"
    assert detail.json()["region_count"] >= 1

    # ---- Step 2: Create synthetic training dataset + samples ----

    async with session_factory() as session:
        dataset = SegmentationTrainingDataset(
            name="pass2-smoke-dataset",
            description="synthetic tone + silence for smoke test",
        )
        session.add(dataset)
        await session.flush()
        dataset_id = dataset.id

        for i in range(4):
            pos_path = audio_dir / f"train_pos_{i}.wav"
            neg_path = audio_dir / f"train_neg_{i}.wav"
            _write_wav(pos_path, duration_sec=1.0)
            _write_wav(neg_path, duration_sec=1.0)

            pos_af = AudioFile(
                filename=pos_path.name,
                folder_path="",
                source_folder=str(audio_dir),
                checksum_sha256=f"pass2-pos-{i}",
                duration_seconds=1.0,
            )
            neg_af = AudioFile(
                filename=neg_path.name,
                folder_path="",
                source_folder=str(audio_dir),
                checksum_sha256=f"pass2-neg-{i}",
                duration_seconds=1.0,
            )
            session.add_all([pos_af, neg_af])
            await session.flush()

            session.add(
                SegmentationTrainingSample(
                    training_dataset_id=dataset_id,
                    audio_file_id=pos_af.id,
                    crop_start_sec=0.0,
                    crop_end_sec=1.0,
                    events_json=json.dumps([{"start_sec": 0.0, "end_sec": 1.0}]),
                    source="smoke_test",
                )
            )
            session.add(
                SegmentationTrainingSample(
                    training_dataset_id=dataset_id,
                    audio_file_id=neg_af.id,
                    crop_start_sec=0.0,
                    crop_end_sec=1.0,
                    events_json="[]",
                    source="smoke_test",
                )
            )

        await session.commit()

    # ---- Step 3: Train segmentation model directly via trainer ----

    import asyncio
    import uuid

    from humpback.call_parsing.segmentation.trainer import train_model
    from humpback.ml.device import select_device
    from humpback.models.call_parsing import SegmentationModel
    from humpback.processing.audio_io import decode_audio, resample
    from humpback.schemas.call_parsing import (
        SegmentationDecoderConfig,
        SegmentationFeatureConfig,
        SegmentationTrainingConfig,
    )
    from humpback.storage import resolve_audio_path

    async with session_factory() as session:
        from sqlalchemy import select as sa_select

        sample_result = await session.execute(
            sa_select(SegmentationTrainingSample).where(
                SegmentationTrainingSample.training_dataset_id == dataset_id
            )
        )
        samples = list(sample_result.scalars().all())

        af_ids = sorted({s.audio_file_id for s in samples if s.audio_file_id})
        af_result = await session.execute(
            sa_select(AudioFile).where(AudioFile.id.in_(af_ids))
        )
        audio_files_by_id = {af.id: af for af in af_result.scalars().all()}

    feature_config = SegmentationFeatureConfig()
    target_sr = feature_config.sample_rate
    audio_cache: dict[str, "np.ndarray"] = {}

    def _audio_loader(sample):
        import numpy as np

        af = audio_files_by_id[sample.audio_file_id]
        if sample.audio_file_id not in audio_cache:
            path = resolve_audio_path(af, settings.storage_root)
            raw, sr = decode_audio(path)
            resampled = resample(raw, sr, target_sr)
            audio_cache[sample.audio_file_id] = np.asarray(resampled, dtype=np.float32)
        audio = audio_cache[sample.audio_file_id]
        start = max(0, int(round(float(sample.crop_start_sec) * target_sr)))
        end = max(start, int(round(float(sample.crop_end_sec) * target_sr)))
        return audio[start:end].copy()

    training_config = SegmentationTrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        early_stopping_patience=100,
        grad_clip=1.0,
        seed=0,
        val_fraction=0.25,
        conv_channels=[8],
        gru_hidden=8,
        gru_layers=1,
    )
    decoder_config = SegmentationDecoderConfig()
    seg_model_id = uuid.uuid4().hex
    model_dir = settings.storage_root / "segmentation_models" / seg_model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "checkpoint.pt"
    device = select_device()

    await asyncio.to_thread(
        train_model,
        samples=samples,
        feature_config=feature_config,
        decoder_config=decoder_config,
        audio_loader=_audio_loader,
        config=training_config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    async with session_factory() as session:
        seg_model = SegmentationModel(
            id=seg_model_id,
            name="segmentation-smoke",
            model_family="pytorch_crnn",
            model_path=str(checkpoint_path),
            config_json=json.dumps(
                {
                    "model_type": "SegmentationCRNN",
                    "n_mels": training_config.n_mels,
                    "conv_channels": list(training_config.conv_channels),
                    "gru_hidden": training_config.gru_hidden,
                    "gru_layers": training_config.gru_layers,
                    "feature_config": feature_config.model_dump(),
                }
            ),
        )
        session.add(seg_model)
        await session.commit()

    # ---- Step 4: Verify model persisted ----

    models_resp = await client.get("/call-parsing/segmentation-models")
    assert models_resp.status_code == 200
    model_ids = [m["id"] for m in models_resp.json()]
    assert seg_model_id in model_ids

    # ---- Step 5: Run Pass 2 event segmentation against the trained model ----

    seg_resp = await client.post(
        "/call-parsing/segmentation-jobs",
        json={
            "region_detection_job_id": pass1_job_id,
            "segmentation_model_id": seg_model_id,
            "config": {
                "high_threshold": 0.01,
                "low_threshold": 0.005,
                "min_event_sec": 0.0,
            },
        },
    )
    assert seg_resp.status_code == 201
    seg_job_id = seg_resp.json()["id"]
    assert seg_resp.json()["status"] == "queued"

    events_409 = await client.get(
        f"/call-parsing/segmentation-jobs/{seg_job_id}/events"
    )
    assert events_409.status_code == 409

    async with session_factory() as session:
        claimed = await seg_run_one_iteration(session, settings)
    assert claimed is not None
    assert claimed.id == seg_job_id

    # ---- Step 6: Verify events via API ----

    seg_detail = await client.get(f"/call-parsing/segmentation-jobs/{seg_job_id}")
    assert seg_detail.status_code == 200
    assert seg_detail.json()["status"] == "complete"
    assert seg_detail.json()["event_count"] >= 1

    events_resp = await client.get(
        f"/call-parsing/segmentation-jobs/{seg_job_id}/events"
    )
    assert events_resp.status_code == 200
    events = events_resp.json()
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
        assert 0.0 <= e["segmentation_confidence"] <= 1.0

    # ---- Step 7: Delete segmentation job and verify cleanup ----

    job_dir = segmentation_job_dir(settings.storage_root, seg_job_id)
    assert job_dir.exists()
    assert (job_dir / "events.parquet").exists()

    del_resp = await client.delete(f"/call-parsing/segmentation-jobs/{seg_job_id}")
    assert del_resp.status_code == 204
    assert not (job_dir / "events.parquet").exists()

    missing = await client.get(f"/call-parsing/segmentation-jobs/{seg_job_id}")
    assert missing.status_code == 404

    await engine.dispose()
