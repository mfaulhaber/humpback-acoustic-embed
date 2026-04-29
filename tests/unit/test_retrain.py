"""Unit tests for retired classifier retrain behavior."""

import json

import pytest

from humpback.models.classifier import ClassifierModel
from humpback.models.retrain import RetrainWorkflow
from humpback.services.classifier_service import (
    collect_embedding_sets_for_folders,
    create_retrain_workflow,
    get_retrain_info,
    get_retrain_workflow,
    list_retrain_workflows,
)


RETIREMENT_ERROR = (
    "Classifier retrain is retired because it depended on the legacy "
    "audio/processing workflow"
)


async def test_collect_embedding_sets_for_folders_is_retired(session):
    with pytest.raises(ValueError, match="Classifier retrain is retired"):
        await collect_embedding_sets_for_folders(session, ["/tmp/positive"], "perch_v1")


async def test_get_retrain_info_returns_none_for_existing_model(session):
    model = ClassifierModel(
        name="legacy-model",
        model_path="/fake/model.joblib",
        model_version="perch_v1",
        vector_dim=1280,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(model)
    await session.commit()

    assert await get_retrain_info(session, model.id) is None


async def test_get_retrain_info_returns_none_for_missing_model(session):
    assert await get_retrain_info(session, "missing") is None


async def test_create_retrain_workflow_is_retired(session):
    with pytest.raises(ValueError, match="Classifier retrain is retired"):
        await create_retrain_workflow(session, "model-id", "retrained-model")


async def test_list_and_get_existing_retrain_workflows(session):
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

    workflows = await list_retrain_workflows(session)
    assert [wf.id for wf in workflows] == [workflow.id]

    fetched = await get_retrain_workflow(session, workflow.id)
    assert fetched is not None
    assert fetched.error_message == RETIREMENT_ERROR


async def test_get_retrain_workflow_not_found(session):
    assert await get_retrain_workflow(session, "missing") is None
