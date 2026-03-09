"""Unit tests for retrain workflow: folder tracing, embedding collection, state machine."""

import json
from pathlib import Path

import pytest

from humpback.models.audio import AudioFile
from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob
from humpback.models.processing import EmbeddingSet
from humpback.models.retrain import RetrainWorkflow
from humpback.services.classifier_service import (
    collect_embedding_sets_for_folders,
    create_retrain_workflow,
    get_retrain_info,
    trace_folder_roots,
)


# ---- Helpers ----


async def _seed_training_scenario(session, tmp_path):
    """Create a full training scenario: audio files, embedding sets, training job, model."""
    # Create audio files as if imported from /data/positive and /data/negative
    pos_dir = tmp_path / "positive"
    pos_dir.mkdir()
    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()

    af_pos = AudioFile(
        filename="song.wav",
        folder_path="positive",
        source_folder=str(pos_dir),
        checksum_sha256="pos_hash_1",
    )
    af_neg = AudioFile(
        filename="noise.wav",
        folder_path="negative",
        source_folder=str(neg_dir),
        checksum_sha256="neg_hash_1",
    )
    session.add_all([af_pos, af_neg])
    await session.flush()

    # Create embedding sets
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

    # Create training job
    tj = ClassifierTrainingJob(
        name="test-classifier",
        positive_embedding_set_ids=json.dumps([es_pos.id]),
        negative_embedding_set_ids=json.dumps([es_neg.id]),
        model_version="perch_v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status="complete",
        classifier_model_id="will-set-below",
        parameters=json.dumps({"classifier_type": "logistic_regression"}),
    )
    session.add(tj)
    await session.flush()

    # Create classifier model
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

    # Update training job with model ID
    tj.classifier_model_id = cm.id
    await session.commit()

    return cm, tj, es_pos, es_neg, af_pos, af_neg


# ---- Folder Tracing ----


async def test_trace_folder_roots_basic(session, tmp_path):
    cm, tj, es_pos, es_neg, af_pos, af_neg = await _seed_training_scenario(
        session, tmp_path
    )

    roots = await trace_folder_roots(session, tj)

    assert str(tmp_path / "positive") in roots["positive_folder_roots"]
    assert str(tmp_path / "negative") in roots["negative_folder_roots"]


async def test_trace_folder_roots_nested_folders(session, tmp_path):
    """Audio in subfolders should trace back to the import root."""
    root_dir = tmp_path / "whale_songs"
    sub_dir = root_dir / "deep"
    sub_dir.mkdir(parents=True)

    af = AudioFile(
        filename="call.wav",
        folder_path="whale_songs/deep",
        source_folder=str(sub_dir),
        checksum_sha256="nested_hash",
    )
    session.add(af)
    await session.flush()

    es = EmbeddingSet(
        audio_file_id=af.id,
        encoding_signature="sig_nested",
        model_version="perch_v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        vector_dim=1280,
        parquet_path="/fake/nested.parquet",
    )
    session.add(es)
    await session.flush()

    tj = ClassifierTrainingJob(
        name="nested-test",
        positive_embedding_set_ids=json.dumps([es.id]),
        negative_embedding_set_ids=json.dumps([]),
        model_version="perch_v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status="complete",
    )
    session.add(tj)
    await session.commit()

    roots = await trace_folder_roots(session, tj)
    assert str(root_dir) in roots["positive_folder_roots"]
    assert roots["negative_folder_roots"] == []


# ---- Embedding Set Collection ----


async def test_collect_embedding_sets_for_folders(session, tmp_path):
    cm, tj, es_pos, es_neg, af_pos, af_neg = await _seed_training_scenario(
        session, tmp_path
    )

    pos_ids = await collect_embedding_sets_for_folders(
        session, [str(tmp_path / "positive")], "perch_v1"
    )
    assert es_pos.id in pos_ids

    neg_ids = await collect_embedding_sets_for_folders(
        session, [str(tmp_path / "negative")], "perch_v1"
    )
    assert es_neg.id in neg_ids


async def test_collect_embedding_sets_wrong_model_version(session, tmp_path):
    cm, tj, es_pos, es_neg, af_pos, af_neg = await _seed_training_scenario(
        session, tmp_path
    )

    ids = await collect_embedding_sets_for_folders(
        session, [str(tmp_path / "positive")], "wrong_model"
    )
    assert ids == []


async def test_collect_embedding_sets_includes_new_files(session, tmp_path):
    """New audio files added after initial training are included."""
    cm, tj, es_pos, es_neg, af_pos, af_neg = await _seed_training_scenario(
        session, tmp_path
    )

    # Add a new audio file in the positive folder
    af_new = AudioFile(
        filename="new_song.wav",
        folder_path="positive",
        source_folder=str(tmp_path / "positive"),
        checksum_sha256="new_pos_hash",
    )
    session.add(af_new)
    await session.flush()

    es_new = EmbeddingSet(
        audio_file_id=af_new.id,
        encoding_signature="sig1",
        model_version="perch_v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        vector_dim=1280,
        parquet_path="/fake/new_pos.parquet",
    )
    session.add(es_new)
    await session.commit()

    pos_ids = await collect_embedding_sets_for_folders(
        session, [str(tmp_path / "positive")], "perch_v1"
    )
    assert es_pos.id in pos_ids
    assert es_new.id in pos_ids
    assert len(pos_ids) == 2


# ---- Retrain Info ----


async def test_get_retrain_info(session, tmp_path):
    cm, tj, es_pos, es_neg, af_pos, af_neg = await _seed_training_scenario(
        session, tmp_path
    )

    info = await get_retrain_info(session, cm.id)
    assert info is not None
    assert info["model_id"] == cm.id
    assert info["model_version"] == "perch_v1"
    assert len(info["positive_folder_roots"]) == 1
    assert len(info["negative_folder_roots"]) == 1
    assert info["parameters"] == {"classifier_type": "logistic_regression"}


async def test_get_retrain_info_not_found(session):
    info = await get_retrain_info(session, "nonexistent")
    assert info is None


async def test_get_retrain_info_no_training_job(session):
    cm = ClassifierModel(
        name="orphan",
        model_path="/fake",
        model_version="v1",
        vector_dim=1280,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(cm)
    await session.commit()

    info = await get_retrain_info(session, cm.id)
    assert info is None


# ---- Create Retrain Workflow ----


async def test_create_retrain_workflow(session, tmp_path):
    cm, tj, es_pos, es_neg, af_pos, af_neg = await _seed_training_scenario(
        session, tmp_path
    )

    wf = await create_retrain_workflow(session, cm.id, "retrained-model")
    assert wf.status == "queued"
    assert wf.source_model_id == cm.id
    assert wf.new_model_name == "retrained-model"
    assert wf.model_version == "perch_v1"
    assert json.loads(wf.positive_folder_roots) == [str(tmp_path / "positive")]
    assert json.loads(wf.negative_folder_roots) == [str(tmp_path / "negative")]


async def test_create_retrain_workflow_with_overrides(session, tmp_path):
    cm, tj, es_pos, es_neg, af_pos, af_neg = await _seed_training_scenario(
        session, tmp_path
    )

    wf = await create_retrain_workflow(
        session, cm.id, "retrained-v2", {"classifier_type": "mlp"}
    )
    params = json.loads(wf.parameters)
    assert params["classifier_type"] == "mlp"


async def test_create_retrain_workflow_nonexistent_model(session):
    with pytest.raises(ValueError, match="not found"):
        await create_retrain_workflow(session, "fake-id", "name")


async def test_create_retrain_workflow_no_training_job(session):
    cm = ClassifierModel(
        name="orphan",
        model_path="/fake",
        model_version="v1",
        vector_dim=1280,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(cm)
    await session.commit()

    with pytest.raises(ValueError, match="no associated training job"):
        await create_retrain_workflow(session, cm.id, "name")
