"""Integration tests for retired classifier retrain endpoints."""

import json

from humpback.database import create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel
from humpback.models.retrain import RetrainWorkflow


RETIREMENT_ERROR = (
    "Classifier retrain is retired because it depended on the legacy "
    "audio/processing workflow"
)


async def _seed_classifier_model(app_settings) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        model = ClassifierModel(
            name="test-classifier",
            model_path="/fake/model.joblib",
            model_version="perch_v1",
            vector_dim=1280,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(model)
        await session.commit()
        model_id = model.id
    await engine.dispose()
    return model_id


async def _seed_retrain_workflow(app_settings) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        workflow = RetrainWorkflow(
            status="failed",
            source_model_id="source-model",
            new_model_name="legacy-retrain",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            parameters=json.dumps({"classifier_type": "logistic_regression"}),
            positive_folder_roots=json.dumps(["/tmp/positive"]),
            negative_folder_roots=json.dumps(["/tmp/negative"]),
            error_message=RETIREMENT_ERROR,
        )
        session.add(workflow)
        await session.commit()
        workflow_id = workflow.id
    await engine.dispose()
    return workflow_id


async def test_retrain_info_returns_404_for_existing_model(client, app_settings):
    model_id = await _seed_classifier_model(app_settings)

    resp = await client.get(f"/classifier/models/{model_id}/retrain-info")
    assert resp.status_code == 404


async def test_retrain_info_not_found(client):
    resp = await client.get("/classifier/models/nonexistent/retrain-info")
    assert resp.status_code == 404


async def test_create_retrain_workflow_returns_retirement_error(client):
    resp = await client.post(
        "/classifier/retrain",
        json={
            "source_model_id": "any-model-id",
            "new_model_name": "retrained-model",
        },
    )
    assert resp.status_code == 400
    assert RETIREMENT_ERROR in resp.json()["detail"]


async def test_list_retrain_workflows_empty(client):
    resp = await client.get("/classifier/retrain-workflows")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_and_get_retain_existing_retrain_workflow(client, app_settings):
    workflow_id = await _seed_retrain_workflow(app_settings)

    resp = await client.get("/classifier/retrain-workflows")
    assert resp.status_code == 200
    assert [wf["id"] for wf in resp.json()] == [workflow_id]
    assert resp.json()[0]["error_message"] == RETIREMENT_ERROR

    resp = await client.get(f"/classifier/retrain-workflows/{workflow_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == workflow_id
    assert resp.json()["status"] == "failed"


async def test_get_retrain_workflow_not_found(client):
    resp = await client.get("/classifier/retrain-workflows/nonexistent")
    assert resp.status_code == 404
