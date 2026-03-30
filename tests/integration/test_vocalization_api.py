"""Integration tests for vocalization type classification API."""

import json

import pytest

from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.processing import EmbeddingSet
from humpback.models.vocalization import VocalizationClassifierModel


# ---- Vocabulary CRUD ----


@pytest.mark.asyncio
async def test_vocabulary_crud(client):
    """Create, list, update, delete vocalization types."""
    # Create
    resp = await client.post(
        "/vocalization/types", json={"name": "Whup", "description": "Low freq"}
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Whup"  # normalized Title Case
    assert data["description"] == "Low freq"
    type_id = data["id"]

    # List
    resp2 = await client.get("/vocalization/types")
    assert resp2.status_code == 200
    types = resp2.json()
    assert len(types) >= 1
    assert any(t["id"] == type_id for t in types)

    # Update
    resp3 = await client.put(f"/vocalization/types/{type_id}", json={"name": "moan"})
    assert resp3.status_code == 200
    assert resp3.json()["name"] == "Moan"

    # Delete
    resp4 = await client.delete(f"/vocalization/types/{type_id}")
    assert resp4.status_code == 204


@pytest.mark.asyncio
async def test_vocabulary_duplicate(client):
    """Creating duplicate type returns 409."""
    await client.post("/vocalization/types", json={"name": "whup"})
    resp = await client.post("/vocalization/types", json={"name": "whup"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_vocabulary_delete_nonexistent(client):
    resp = await client.delete("/vocalization/types/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_vocabulary_import(client, app_settings):
    """Import types from embedding set folder structure."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        af = AudioFile(
            filename="call.wav",
            folder_path="curated/shriek",
            checksum_sha256="xyz123",
        )
        session.add(af)
        await session.flush()

        es = EmbeddingSet(
            audio_file_id=af.id,
            encoding_signature="sig-test",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=128,
            parquet_path="/fake/path.parquet",
        )
        session.add(es)
        await session.commit()
        es_id = es.id

    resp = await client.post(
        "/vocalization/types/import",
        json={"embedding_set_ids": [es_id]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Shriek" in data["added"]


# ---- Models ----


@pytest.mark.asyncio
async def test_models_list_empty(client):
    resp = await client.get("/vocalization/models")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_model_activate(client, app_settings):
    """Activate a model and verify it's marked active."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        m = VocalizationClassifierModel(
            name="test-model",
            model_dir_path="/fake/dir",
            vocabulary_snapshot=json.dumps(["whup"]),
            per_class_thresholds=json.dumps({"whup": 0.5}),
            is_active=False,
        )
        session.add(m)
        await session.commit()
        model_id = m.id

    resp = await client.put(f"/vocalization/models/{model_id}/activate")
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True

    # Verify via list
    resp2 = await client.get("/vocalization/models")
    models = resp2.json()
    active = [m for m in models if m["is_active"]]
    assert len(active) == 1
    assert active[0]["id"] == model_id


@pytest.mark.asyncio
async def test_model_get_nonexistent(client):
    resp = await client.get("/vocalization/models/nonexistent")
    assert resp.status_code == 404


# ---- Training Jobs ----


@pytest.mark.asyncio
async def test_training_job_lifecycle(client):
    """Create and fetch a training job."""
    resp = await client.post(
        "/vocalization/training-jobs",
        json={
            "source_config": {
                "embedding_set_ids": ["es-1"],
                "detection_job_ids": [],
            },
            "parameters": {"min_examples_per_type": 4},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    job_id = data["id"]

    # Fetch by ID
    resp2 = await client.get(f"/vocalization/training-jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == job_id

    # List
    resp3 = await client.get("/vocalization/training-jobs")
    assert resp3.status_code == 200
    assert any(j["id"] == job_id for j in resp3.json())


@pytest.mark.asyncio
async def test_training_job_nonexistent(client):
    resp = await client.get("/vocalization/training-jobs/nonexistent")
    assert resp.status_code == 404


# ---- Inference Jobs ----


@pytest.mark.asyncio
async def test_inference_job_requires_valid_model(client):
    """Inference job creation fails without a valid model."""
    resp = await client.post(
        "/vocalization/inference-jobs",
        json={
            "vocalization_model_id": "nonexistent",
            "source_type": "embedding_set",
            "source_id": "es-1",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_inference_job_lifecycle(client, app_settings):
    """Create and list inference jobs."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        m = VocalizationClassifierModel(
            name="model-for-inference",
            model_dir_path="/fake/dir",
            vocabulary_snapshot=json.dumps(["whup"]),
            per_class_thresholds=json.dumps({"whup": 0.5}),
            is_active=True,
        )
        session.add(m)
        await session.commit()
        model_id = m.id

    resp = await client.post(
        "/vocalization/inference-jobs",
        json={
            "vocalization_model_id": model_id,
            "source_type": "embedding_set",
            "source_id": "es-1",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["vocalization_model_id"] == model_id
    job_id = data["id"]

    # List
    resp2 = await client.get("/vocalization/inference-jobs")
    assert resp2.status_code == 200
    assert any(j["id"] == job_id for j in resp2.json())

    # Get
    resp3 = await client.get(f"/vocalization/inference-jobs/{job_id}")
    assert resp3.status_code == 200


@pytest.mark.asyncio
async def test_inference_results_not_complete(client, app_settings):
    """Results endpoint returns 400 if job not complete."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        m = VocalizationClassifierModel(
            name="m",
            model_dir_path="/fake",
            vocabulary_snapshot=json.dumps(["whup"]),
            per_class_thresholds=json.dumps({"whup": 0.5}),
        )
        session.add(m)
        await session.commit()
        model_id = m.id

    resp = await client.post(
        "/vocalization/inference-jobs",
        json={
            "vocalization_model_id": model_id,
            "source_type": "embedding_set",
            "source_id": "es-1",
        },
    )
    job_id = resp.json()["id"]

    resp2 = await client.get(f"/vocalization/inference-jobs/{job_id}/results")
    assert resp2.status_code == 400


# ---- Training Source ----


@pytest.mark.asyncio
async def test_model_training_source(client, app_settings):
    """GET training-source returns source_config from the producing training job."""
    from humpback.models.vocalization import VocalizationTrainingJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        m = VocalizationClassifierModel(
            name="sourced-model",
            model_dir_path="/fake/dir",
            vocabulary_snapshot=json.dumps(["whup"]),
            per_class_thresholds=json.dumps({"whup": 0.5}),
        )
        session.add(m)
        await session.flush()

        tj = VocalizationTrainingJob(
            source_config=json.dumps(
                {
                    "embedding_set_ids": ["es-1"],
                    "detection_job_ids": ["dj-1"],
                }
            ),
            parameters=json.dumps({"C": 2.0}),
            status="complete",
            vocalization_model_id=m.id,
        )
        session.add(tj)
        await session.commit()
        model_id = m.id

    resp = await client.get(f"/vocalization/models/{model_id}/training-source")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_config"]["detection_job_ids"] == ["dj-1"]
    assert data["parameters"]["C"] == 2.0


@pytest.mark.asyncio
async def test_model_training_source_no_training_job(client, app_settings):
    """GET training-source returns nulls when no training job is linked."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        m = VocalizationClassifierModel(
            name="orphan-model",
            model_dir_path="/fake/dir",
            vocabulary_snapshot=json.dumps(["whup"]),
            per_class_thresholds=json.dumps({"whup": 0.5}),
        )
        session.add(m)
        await session.commit()
        model_id = m.id

    resp = await client.get(f"/vocalization/models/{model_id}/training-source")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_config"] is None
    assert data["parameters"] is None


# ---- Confidence Sort ----


@pytest.mark.asyncio
async def test_inference_results_confidence_sort(client, app_settings):
    """Results with sort=confidence_desc return rows ordered by confidence."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from humpback.models.vocalization import VocalizationInferenceJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        m = VocalizationClassifierModel(
            name="conf-sort-model",
            model_dir_path="/fake/dir",
            vocabulary_snapshot=json.dumps(["whup"]),
            per_class_thresholds=json.dumps({"whup": 0.5}),
        )
        session.add(m)
        await session.commit()
        model_id = m.id

    # Write a predictions parquet with known confidence values
    output_dir = app_settings.storage_root / "vocalization_inference" / "conf-job"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.parquet"

    table = pa.table(
        {
            "filename": ["a.wav", "b.wav", "c.wav"],
            "start_sec": [0.0, 5.0, 10.0],
            "end_sec": [5.0, 10.0, 15.0],
            "confidence": [0.3, 0.9, 0.6],
            "whup": [0.8, 0.2, 0.5],
        }
    )
    pq.write_table(table, str(output_path))

    # Create a completed inference job pointing to that parquet
    async with sf() as session:
        job = VocalizationInferenceJob(
            id="conf-job",
            vocalization_model_id=model_id,
            source_type="embedding_set",
            source_id="es-fake",
            status="complete",
            output_path=str(output_path),
        )
        session.add(job)
        await session.commit()

    # Without sort — parquet insertion order
    resp = await client.get("/vocalization/inference-jobs/conf-job/results")
    assert resp.status_code == 200
    rows = resp.json()
    confs_unsorted = [r["confidence"] for r in rows]
    assert confs_unsorted == [0.3, 0.9, 0.6]  # insertion order

    # With sort=confidence_desc
    resp2 = await client.get(
        "/vocalization/inference-jobs/conf-job/results?sort=confidence_desc"
    )
    assert resp2.status_code == 200
    rows2 = resp2.json()
    confs_sorted = [r["confidence"] for r in rows2]
    assert confs_sorted == [0.9, 0.6, 0.3]


@pytest.mark.asyncio
async def test_inference_results_confidence_null(client, app_settings):
    """Results without confidence column return null and sort degrades gracefully."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from humpback.models.vocalization import VocalizationInferenceJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        m = VocalizationClassifierModel(
            name="no-conf-model",
            model_dir_path="/fake/dir",
            vocabulary_snapshot=json.dumps(["whup"]),
            per_class_thresholds=json.dumps({"whup": 0.5}),
        )
        session.add(m)
        await session.commit()
        model_id = m.id

    # Write predictions WITHOUT confidence column
    output_dir = app_settings.storage_root / "vocalization_inference" / "noconf-job"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.parquet"

    table = pa.table(
        {
            "filename": ["a.wav", "b.wav"],
            "start_sec": [0.0, 5.0],
            "end_sec": [5.0, 10.0],
            "whup": [0.8, 0.2],
        }
    )
    pq.write_table(table, str(output_path))

    async with sf() as session:
        job = VocalizationInferenceJob(
            id="noconf-job",
            vocalization_model_id=model_id,
            source_type="embedding_set",
            source_id="es-fake",
            status="complete",
            output_path=str(output_path),
        )
        session.add(job)
        await session.commit()

    # confidence should be null
    resp = await client.get("/vocalization/inference-jobs/noconf-job/results")
    assert resp.status_code == 200
    rows = resp.json()
    assert all(r["confidence"] is None for r in rows)

    # sort=confidence_desc should not error
    resp2 = await client.get(
        "/vocalization/inference-jobs/noconf-job/results?sort=confidence_desc"
    )
    assert resp2.status_code == 200
