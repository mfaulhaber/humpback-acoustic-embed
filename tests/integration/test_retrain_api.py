"""Integration tests for retrain workflow API endpoints."""

import json
import uuid


from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob
from humpback.models.processing import EmbeddingSet


async def _seed_model_with_training(app_settings, tmp_path_factory):
    """Insert a classifier model with training provenance into the test DB."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    pos_dir = tmp_path_factory.mktemp("positive")
    neg_dir = tmp_path_factory.mktemp("negative")

    async with sf() as session:
        af_pos = AudioFile(
            filename="song.wav",
            folder_path="positive",
            source_folder=str(pos_dir),
            checksum_sha256=f"pos_{uuid.uuid4().hex[:8]}",
        )
        af_neg = AudioFile(
            filename="noise.wav",
            folder_path="negative",
            source_folder=str(neg_dir),
            checksum_sha256=f"neg_{uuid.uuid4().hex[:8]}",
        )
        session.add_all([af_pos, af_neg])
        await session.flush()

        es_pos = EmbeddingSet(
            audio_file_id=af_pos.id,
            encoding_signature="sig1",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=1280,
            parquet_path="/fake/pos.parquet",
        )
        es_neg = EmbeddingSet(
            audio_file_id=af_neg.id,
            encoding_signature="sig1",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=1280,
            parquet_path="/fake/neg.parquet",
        )
        session.add_all([es_pos, es_neg])
        await session.flush()

        tj = ClassifierTrainingJob(
            name="test-classifier",
            positive_embedding_set_ids=json.dumps([es_pos.id]),
            negative_embedding_set_ids=json.dumps([es_neg.id]),
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            status="complete",
            parameters=json.dumps({"classifier_type": "logistic_regression"}),
        )
        session.add(tj)
        await session.flush()

        cm = ClassifierModel(
            name="test-classifier",
            model_path="/fake/model.joblib",
            model_version="perch_v1",
            vector_dim=1280,
            window_size_seconds=5.0,
            target_sample_rate=32000,
            training_job_id=tj.id,
        )
        session.add(cm)
        await session.flush()

        tj.classifier_model_id = cm.id
        await session.commit()

        model_id = cm.id

    await engine.dispose()
    return model_id


# ---- Retrain Info ----


async def test_retrain_info(client, app_settings, tmp_path_factory):
    model_id = await _seed_model_with_training(app_settings, tmp_path_factory)

    resp = await client.get(f"/classifier/models/{model_id}/retrain-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == model_id
    assert data["model_version"] == "perch_v1"
    assert len(data["positive_folder_roots"]) == 1
    assert len(data["negative_folder_roots"]) == 1
    assert data["parameters"]["classifier_type"] == "logistic_regression"


async def test_retrain_info_not_found(client):
    resp = await client.get("/classifier/models/nonexistent/retrain-info")
    assert resp.status_code == 404


# ---- Create Retrain Workflow ----


async def test_create_retrain_workflow(client, app_settings, tmp_path_factory):
    model_id = await _seed_model_with_training(app_settings, tmp_path_factory)

    resp = await client.post(
        "/classifier/retrain",
        json={
            "source_model_id": model_id,
            "new_model_name": "retrained-model",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["source_model_id"] == model_id
    assert data["new_model_name"] == "retrained-model"
    assert data["model_version"] == "perch_v1"
    assert len(data["positive_folder_roots"]) == 1
    assert len(data["negative_folder_roots"]) == 1


async def test_create_retrain_workflow_with_overrides(
    client, app_settings, tmp_path_factory
):
    model_id = await _seed_model_with_training(app_settings, tmp_path_factory)

    resp = await client.post(
        "/classifier/retrain",
        json={
            "source_model_id": model_id,
            "new_model_name": "retrained-mlp",
            "parameters": {"classifier_type": "mlp"},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["parameters"]["classifier_type"] == "mlp"


async def test_create_retrain_workflow_bad_model(client):
    resp = await client.post(
        "/classifier/retrain",
        json={
            "source_model_id": "nonexistent",
            "new_model_name": "name",
        },
    )
    assert resp.status_code == 400


# ---- List/Get Retrain Workflows ----


async def test_list_retrain_workflows_empty(client):
    resp = await client.get("/classifier/retrain-workflows")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_and_get_retrain_workflow(client, app_settings, tmp_path_factory):
    model_id = await _seed_model_with_training(app_settings, tmp_path_factory)

    # Create a workflow
    resp = await client.post(
        "/classifier/retrain",
        json={
            "source_model_id": model_id,
            "new_model_name": "retrained",
        },
    )
    assert resp.status_code == 201
    wf_id = resp.json()["id"]

    # List should include it
    resp = await client.get("/classifier/retrain-workflows")
    assert resp.status_code == 200
    ids = [wf["id"] for wf in resp.json()]
    assert wf_id in ids

    # Get by ID
    resp = await client.get(f"/classifier/retrain-workflows/{wf_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == wf_id
    assert resp.json()["status"] == "queued"


async def test_get_retrain_workflow_not_found(client):
    resp = await client.get("/classifier/retrain-workflows/nonexistent")
    assert resp.status_code == 404
