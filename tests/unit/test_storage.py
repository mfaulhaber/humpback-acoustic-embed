"""Tests for humpback.storage path helpers."""

from pathlib import Path

from humpback.storage import (
    continuous_embedding_dir,
    continuous_embedding_manifest_path,
    continuous_embedding_parquet_path,
    detection_embeddings_path,
    event_encoder_dir,
    event_encoder_kmeans_path,
    event_encoder_manifest_path,
    event_encoder_preprocess_path,
    event_encoder_report_path,
    event_encoder_sequences_path,
    event_encoder_tokens_path,
    event_encoder_vectors_path,
)


def test_detection_embeddings_path_includes_model_version():
    root = Path("/tmp/storage")
    job_id = "det-abc"
    p = detection_embeddings_path(root, job_id, "perch_v2")
    assert "perch_v2" in p.parts
    assert job_id in p.parts
    assert p.name == "detection_embeddings.parquet"


def test_detection_embeddings_path_differs_by_model_version():
    root = Path("/tmp/storage")
    job_id = "det-abc"
    a = detection_embeddings_path(root, job_id, "perch_v2")
    b = detection_embeddings_path(root, job_id, "birdnet_tf2")
    assert a != b
    assert a.parent != b.parent


def test_continuous_embedding_paths_under_root():
    root = Path("/tmp/storage")
    job_id = "cej-1"
    d = continuous_embedding_dir(root, job_id)
    assert d == root / "continuous_embeddings" / job_id

    parquet = continuous_embedding_parquet_path(root, job_id)
    assert parquet == d / "embeddings.parquet"

    manifest = continuous_embedding_manifest_path(root, job_id)
    assert manifest == d / "manifest.json"


def test_event_encoder_paths_under_root():
    root = Path("/tmp/storage")
    job_id = "eej-1"
    d = event_encoder_dir(root, job_id)
    assert d == root / "event_encoders" / job_id
    assert event_encoder_manifest_path(root, job_id) == d / "manifest.json"
    assert event_encoder_report_path(root, job_id) == d / "report.json"
    assert event_encoder_vectors_path(root, job_id) == d / "event_vectors.parquet"
    assert event_encoder_tokens_path(root, job_id) == d / "event_tokens.parquet"
    assert event_encoder_sequences_path(root, job_id) == d / "token_sequences.parquet"
    assert event_encoder_preprocess_path(root, job_id) == d / "preprocess.joblib"
    assert event_encoder_kmeans_path(root, job_id, 50) == d / "kmeans_k50.joblib"
