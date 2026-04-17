"""Tests for humpback.storage path helpers."""

from pathlib import Path

from humpback.storage import detection_embeddings_path


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
