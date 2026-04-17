"""Integration tests for classifier API endpoints."""

import csv
import json
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Epoch anchors for test fixtures.
# 2024-06-15T08:00:00Z — generic base for non-filename-anchored tests.
BASE_EPOCH = 1718438400.0
# 2025-06-15T08:00:00Z — matches "20250615T080000Z" in fixture filenames.
_20250615T080000Z = 1749974400.0
# 2025-07-02T08:01:18Z — matches "20250702T080118Z" in fixture filenames.
_20250702T080118Z = 1751443278.0


def _autoresearch_fixture_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "autoresearch"
        / "explicit-negatives"
    )


def _write_embedding_set_parquet(path: Path, rows: list[list[float]]) -> None:
    table = pa.table(
        {
            "row_index": pa.array(list(range(len(rows))), type=pa.int32()),
            "embedding": pa.array(rows, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _write_detection_embeddings_parquet(
    path: Path,
    row_ids: list[str],
    rows: list[list[float]],
) -> None:
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(rows, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


async def _import_promotable_candidate(client, app_settings) -> dict:
    from humpback.database import create_engine, create_session_factory
    from humpback.models.audio import AudioFile
    from humpback.models.classifier import ClassifierModel, DetectionJob
    from humpback.models.processing import EmbeddingSet

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    source_model_id = str(uuid.uuid4())
    detection_job_id = str(uuid.uuid4())
    af_id = str(uuid.uuid4())

    pos_path = app_settings.storage_root / "candidate-fixtures" / "pos.parquet"
    det_path = (
        app_settings.storage_root
        / "detections"
        / detection_job_id
        / "detection_embeddings.parquet"
    )
    artifact_dir = app_settings.storage_root / "candidate-fixtures" / "artifacts"

    _write_embedding_set_parquet(pos_path, [[2.0, 2.0], [2.5, 2.5]])
    _write_detection_embeddings_parquet(
        det_path,
        row_ids=["neg-1", "neg-2"],
        rows=[[-2.0, -2.0], [-2.5, -2.5]],
    )

    async with sf() as session:
        session.add(
            ClassifierModel(
                id=source_model_id,
                name="Local Source Model",
                model_path="/tmp/source-model.joblib",
                model_version="perch_v1",
                vector_dim=2,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        session.add(
            AudioFile(
                id=af_id,
                filename="pos.wav",
                folder_path="positive",
                checksum_sha256="candidate-pos",
            )
        )
        session.add(
            EmbeddingSet(
                id=str(uuid.uuid4()),
                audio_file_id=af_id,
                encoding_signature="sig",
                model_version="perch_v1",
                window_size_seconds=5.0,
                target_sample_rate=32000,
                vector_dim=2,
                parquet_path=str(pos_path),
            )
        )
        session.add(
            DetectionJob(
                id=detection_job_id,
                status="complete",
                classifier_model_id=source_model_id,
                audio_folder=str(app_settings.storage_root / "audio"),
                confidence_threshold=0.5,
                detection_mode="windowed",
            )
        )
        await session.commit()

    manifest_path = artifact_dir / "manifest.json"
    best_run_path = artifact_dir / "best_run.json"
    comparison_path = artifact_dir / "comparison.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "examples": [
                    {
                        "id": "pos-0",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(pos_path),
                        "row_index": 0,
                    },
                    {
                        "id": "pos-1",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(pos_path),
                        "row_index": 1,
                    },
                    {
                        "id": "neg-1",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-1",
                    },
                    {
                        "id": "neg-2",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-2",
                    },
                ],
            }
        )
    )
    best_run_path.write_text(
        json.dumps(
            {
                "config": {
                    "classifier": "logreg",
                    "feature_norm": "standard",
                    "pca_dim": None,
                    "class_weight_pos": 1.0,
                    "class_weight_neg": 1.0,
                    "hard_negative_fraction": 0.0,
                    "prob_calibration": "none",
                    "threshold": 0.5,
                    "context_pooling": "center",
                    "seed": 42,
                },
                "metrics": {
                    "threshold": 0.5,
                    "precision": 1.0,
                    "recall": 1.0,
                    "fp": 0,
                    "fn": 0,
                    "tp": 2,
                    "tn": 2,
                },
                "config_hash": "promotable123",
                "trial": 1,
            }
        )
    )
    comparison_path.write_text(
        json.dumps(
            {
                "objective_name": "default",
                "production": {
                    "id": source_model_id,
                    "name": "Local Source Model",
                    "model_version": "perch_v1",
                },
                "splits": {
                    "test": {
                        "autoresearch": {
                            "metrics": {"precision": 1.0, "recall": 1.0},
                            "top_false_positives": [],
                        },
                        "production": {
                            "metrics": {"precision": 0.5, "recall": 0.5},
                            "top_false_positives": [],
                        },
                        "delta": {"precision": 0.5, "recall": 0.5},
                        "prediction_disagreements": [],
                    }
                },
            }
        )
    )

    resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "name": "Promotable Candidate",
            "manifest_path": str(manifest_path),
            "best_run_path": str(best_run_path),
            "comparison_path": str(comparison_path),
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "promotable"

    await engine.dispose()
    return data


async def test_create_training_job_missing_embedding_sets(client):
    """400 when embedding sets don't exist."""
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "test",
            "positive_embedding_set_ids": ["nonexistent"],
            "negative_embedding_set_ids": ["nonexistent2"],
        },
    )
    assert resp.status_code == 400


async def test_create_training_job_missing_negative_sets(client):
    """400 when negative embedding sets don't exist."""
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "test",
            "positive_embedding_set_ids": ["nonexistent"],
            "negative_embedding_set_ids": ["also-nonexistent"],
        },
    )
    assert resp.status_code == 400


async def test_list_training_jobs_empty(client):
    resp = await client.get("/classifier/training-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_training_job_not_found(client):
    resp = await client.get("/classifier/training-jobs/nonexistent")
    assert resp.status_code == 404


async def test_list_models_empty(client):
    resp = await client.get("/classifier/models")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_model_not_found(client):
    resp = await client.get("/classifier/models/nonexistent")
    assert resp.status_code == 404


async def test_import_autoresearch_candidate_success(client, app_settings):
    fixture_dir = _autoresearch_fixture_dir()

    resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "name": "Explicit Negatives Phase 1",
            "manifest_path": str(fixture_dir / "manifest.json"),
            "best_run_path": str(fixture_dir / "phase1" / "best_run.json"),
            "comparison_path": str(fixture_dir / "phase1" / "lr-v12-comparison.json"),
            "top_false_positives_path": str(
                fixture_dir / "phase1" / "top_false_positives.json"
            ),
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Explicit Negatives Phase 1"
    assert data["status"] == "promotable"
    assert data["phase"] == "phase1"
    assert data["source_model_name"] == "LR-v12"
    assert data["comparison_target"] == "LR-v12"
    assert data["is_reproducible_exact"] is True
    assert len(data["warnings"]) == 0
    assert data["artifact_paths"]["manifest_path"].startswith(
        str(app_settings.storage_root)
    )
    assert data["artifact_paths"]["best_run_path"].startswith(
        str(app_settings.storage_root)
    )
    assert data["top_false_positives_preview"]["imported"]
    assert data["prediction_disagreements_preview"]["test"]
    assert data["split_metrics"]["test"]["autoresearch"]["precision"] == 0.987342
    assert data["metric_deltas"]["test"]["fp"] == -92.0
    assert data["source_counts"]["split_counts"]["train"] > 0

    list_resp = await client.get("/classifier/autoresearch-candidates")
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert len(listed) == 1
    assert listed[0]["id"] == data["id"]
    assert listed[0]["name"] == "Explicit Negatives Phase 1"

    detail_resp = await client.get(f"/classifier/autoresearch-candidates/{data['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == data["id"]
    assert detail["source_model_metadata"]["name"] == "LR-v12"


async def test_import_autoresearch_candidate_missing_manifest(client):
    fixture_dir = _autoresearch_fixture_dir()

    resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "manifest_path": str(fixture_dir / "missing-manifest.json"),
            "best_run_path": str(fixture_dir / "phase1" / "best_run.json"),
        },
    )
    assert resp.status_code == 400
    assert "manifest.json not found" in resp.text


async def test_import_autoresearch_candidate_rejects_malformed_best_run(
    client, tmp_path
):
    fixture_dir = _autoresearch_fixture_dir()
    malformed = tmp_path / "best_run.json"
    malformed.write_text("{not valid json")

    resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "manifest_path": str(fixture_dir / "manifest.json"),
            "best_run_path": str(malformed),
        },
    )
    assert resp.status_code == 400
    assert "best_run.json is not valid JSON" in resp.text


async def test_import_autoresearch_candidate_rejects_malformed_optional_comparison(
    client, tmp_path
):
    fixture_dir = _autoresearch_fixture_dir()
    malformed = tmp_path / "comparison.json"
    malformed.write_text(json.dumps({"not": "a comparison"}))

    resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "manifest_path": str(fixture_dir / "manifest.json"),
            "best_run_path": str(fixture_dir / "phase1" / "best_run.json"),
            "comparison_path": str(malformed),
        },
    )
    assert resp.status_code == 400
    assert "comparison JSON must contain 'splits'" in resp.text


async def test_import_autoresearch_candidate_accepts_comparison_summary_fixture(
    client,
):
    fixture_dir = _autoresearch_fixture_dir()

    resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "name": "Summary Comparison Candidate",
            "manifest_path": str(fixture_dir / "manifest.json"),
            "best_run_path": str(fixture_dir / "phase1" / "best_run.json"),
            "comparison_path": str(fixture_dir / "comparison_summary.json"),
            "top_false_positives_path": str(
                fixture_dir / "phase1" / "top_false_positives.json"
            ),
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Summary Comparison Candidate"
    assert data["comparison_target"] == "comparison_summary"
    assert data["split_metrics"] is None
    assert data["metric_deltas"] is None
    assert data["top_false_positives_preview"]["imported"]
    assert any(
        "split-level production deltas" in warning for warning in data["warnings"]
    )


async def test_get_autoresearch_candidate_not_found(client):
    resp = await client.get("/classifier/autoresearch-candidates/nonexistent")
    assert resp.status_code == 404


async def test_create_training_job_from_promotable_autoresearch_candidate(
    client, app_settings
):
    candidate = await _import_promotable_candidate(client, app_settings)

    resp = await client.post(
        f"/classifier/autoresearch-candidates/{candidate['id']}/training-jobs",
        json={"new_model_name": "candidate-backed-model", "notes": "ship it"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "candidate-backed-model"
    assert data["status"] == "queued"
    assert data["source_mode"] == "autoresearch_candidate"
    assert data["source_candidate_id"] == candidate["id"]
    assert data["manifest_path"] == candidate["artifact_paths"]["manifest_path"]
    assert data["training_split_name"] == "train"
    assert data["promoted_config"]["classifier"] == "logreg"
    assert data["source_comparison_context"]["candidate_id"] == candidate["id"]
    assert data["source_comparison_context"]["notes"] == "ship it"

    detail_resp = await client.get(
        f"/classifier/autoresearch-candidates/{candidate['id']}"
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["status"] == "training"
    assert detail["training_job_id"] == data["id"]


async def test_blocked_autoresearch_candidate_cannot_create_training_job(
    client, app_settings
):
    """Candidates with still-unsupported configs must remain blocked."""
    fixture_dir = _autoresearch_fixture_dir()

    # hard_negative_fraction > 0 is still blocked (ADR-043, not lifted).
    blocked_best_run = app_settings.storage_root / "blocked_best_run.json"
    blocked_best_run.parent.mkdir(parents=True, exist_ok=True)
    blocked_best_run.write_text(
        json.dumps(
            {
                "config": {
                    "classifier": "logreg",
                    "feature_norm": "l2",
                    "pca_dim": None,
                    "class_weight_pos": 1.0,
                    "class_weight_neg": 1.0,
                    "hard_negative_fraction": 0.25,
                    "prob_calibration": "none",
                    "threshold": 0.5,
                    "context_pooling": "center",
                    "seed": 42,
                },
                "metrics": {"threshold": 0.5, "precision": 0.9, "recall": 0.9},
                "config_hash": "blocked_hardneg",
            }
        )
    )

    import_resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "name": "Blocked Hard-Negative Candidate",
            "manifest_path": str(fixture_dir / "manifest.json"),
            "best_run_path": str(blocked_best_run),
        },
    )
    assert import_resp.status_code == 201
    candidate = import_resp.json()
    assert candidate["status"] == "blocked"

    promote_resp = await client.post(
        f"/classifier/autoresearch-candidates/{candidate['id']}/training-jobs",
        json={"new_model_name": "should-fail"},
    )
    assert promote_resp.status_code == 400
    assert "not promotable" in promote_resp.text


async def test_import_linear_svm_candidate_is_promotable(client, app_settings):
    """Linear SVM candidates are now reproducible via the shared replay module."""
    fixture_dir = _autoresearch_fixture_dir()

    svm_best_run = app_settings.storage_root / "svm_best_run.json"
    svm_best_run.parent.mkdir(parents=True, exist_ok=True)
    svm_best_run.write_text(
        json.dumps(
            {
                "config": {
                    "classifier": "linear_svm",
                    "feature_norm": "standard",
                    "pca_dim": None,
                    "class_weight_pos": 2.0,
                    "class_weight_neg": 1.0,
                    "hard_negative_fraction": 0.0,
                    "prob_calibration": "none",
                    "threshold": 0.5,
                    "context_pooling": "center",
                    "seed": 42,
                },
                "metrics": {"threshold": 0.5, "precision": 0.95, "recall": 0.93},
                "config_hash": "svm_promotable",
            }
        )
    )

    import_resp = await client.post(
        "/classifier/autoresearch-candidates/import",
        json={
            "name": "Promotable Linear SVM Candidate",
            "manifest_path": str(fixture_dir / "manifest.json"),
            "best_run_path": str(svm_best_run),
        },
    )
    assert import_resp.status_code == 201
    candidate = import_resp.json()
    assert candidate["status"] == "promotable"
    assert candidate["is_reproducible_exact"] is True
    # Warnings may include "no comparison artifact" since we don't supply one
    # in this test; the important thing is that no blockers prevent promotion.
    assert not any("linear_svm" in w for w in candidate["warnings"]), candidate[
        "warnings"
    ]


async def test_delete_model_not_found(client):
    resp = await client.delete("/classifier/models/nonexistent")
    assert resp.status_code == 404


async def test_list_detection_jobs_empty(client):
    resp = await client.get("/classifier/detection-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_detection_job_bad_model(client):
    resp = await client.post(
        "/classifier/detection-jobs",
        json={
            "classifier_model_id": "nonexistent",
            "audio_folder": "/tmp",
        },
    )
    assert resp.status_code == 400


async def test_create_detection_job_success(client, app_settings, test_wav):
    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import ClassifierModel

    model_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="local-detection-model",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/detection-jobs",
        json={
            "classifier_model_id": model_id,
            "audio_folder": str(test_wav.parent),
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["audio_folder"] == str(test_wav.parent)
    assert data["detection_mode"] == "windowed"

    get_resp = await client.get(f"/classifier/detection-jobs/{data['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["detection_mode"] == "windowed"

    await engine.dispose()


async def test_create_detection_job_rejects_detection_mode(
    client, app_settings, test_wav
):
    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import ClassifierModel

    model_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="local-detection-model-legacy",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/detection-jobs",
        json={
            "classifier_model_id": model_id,
            "audio_folder": str(test_wav.parent),
            "detection_mode": "merged",
        },
    )
    assert resp.status_code == 422
    assert "detection_mode" in resp.text

    await engine.dispose()


async def test_get_detection_job_not_found(client):
    resp = await client.get("/classifier/detection-jobs/nonexistent")
    assert resp.status_code == 404


async def test_download_detection_not_found(client):
    resp = await client.get("/classifier/detection-jobs/nonexistent/download")
    assert resp.status_code == 404


async def test_extraction_settings(client):
    resp = await client.get("/classifier/extraction-settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "positive_output_path" in data
    assert "negative_output_path" in data
    assert data["positive_selection_smoothing_window"] == 3
    assert data["positive_selection_min_score"] == 0.7
    assert data["positive_selection_extend_min_score"] == 0.6


async def test_extract_nonexistent_jobs(client):
    resp = await client.post(
        "/classifier/detection-jobs/extract",
        json={"job_ids": ["nonexistent"]},
    )
    assert resp.status_code == 404


async def test_extract_empty_job_ids(client):
    resp = await client.post(
        "/classifier/detection-jobs/extract",
        json={"job_ids": []},
    )
    # Empty list: all 0 jobs found, 0 expected → should succeed with count 0
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


async def test_extract_persists_positive_selection_config(client, app_settings):
    from sqlalchemy import insert, select

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/detection-jobs/extract",
        json={
            "job_ids": [job_id],
            "positive_selection_smoothing_window": 5,
            "positive_selection_min_score": 0.82,
            "positive_selection_extend_min_score": 0.61,
        },
    )
    assert resp.status_code == 200

    async with sf() as session:
        result = await session.execute(
            select(DetectionJob.extract_config).where(DetectionJob.id == job_id)
        )
        config = result.scalar_one()
        assert config is not None
        parsed = json.loads(config)

    assert parsed["positive_selection_smoothing_window"] == 5
    assert parsed["positive_selection_min_score"] == 0.82
    assert parsed["positive_selection_extend_min_score"] == 0.61
    await engine.dispose()


# ---- Diagnostics Endpoints ----


async def test_diagnostics_not_found(client):
    resp = await client.get("/classifier/detection-jobs/nonexistent/diagnostics")
    assert resp.status_code == 404


async def test_diagnostics_summary_not_found(client):
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/diagnostics/summary"
    )
    assert resp.status_code == 404


async def test_training_summary_not_found(client):
    resp = await client.get("/classifier/models/nonexistent/training-summary")
    assert resp.status_code == 404


# ---- Incremental Detection Content ----


async def test_content_endpoint_serves_running_job(client, app_settings):
    """GET /content returns partial results when job is running with row store on disk."""
    from pathlib import Path
    from sqlalchemy import insert
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob
    from humpback.classifier.detection_rows import write_detection_row_store

    # Create a running detection job directly in the DB
    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    # Write a partial row store to disk (as the worker now does during detection)
    storage_root = Path(app_settings.storage_root)
    ddir = storage_root / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH + 1.0),
                "end_utc": str(BASE_EPOCH + 6.0),
                "avg_confidence": "0.850000",
                "peak_confidence": "0.900000",
                "n_windows": "2",
                "raw_start_utc": str(BASE_EPOCH + 1.0),
                "raw_end_utc": str(BASE_EPOCH + 6.0),
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="running",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                files_processed=1,
                files_total=5,
            )
        )
        await session.commit()

    # Content endpoint should serve partial results
    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["start_utc"] == BASE_EPOCH + 1.0
    assert rows[0]["avg_confidence"] == 0.85
    assert rows[0]["raw_start_utc"] == BASE_EPOCH + 1.0
    assert rows[0]["raw_end_utc"] == BASE_EPOCH + 6.0
    assert rows[0]["merged_event_count"] == 1

    # Job list should include progress fields
    resp = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert resp.status_code == 200
    job_data = resp.json()
    assert job_data["files_processed"] == 1
    assert job_data["files_total"] == 5
    assert job_data["status"] == "running"

    await engine.dispose()


async def test_content_endpoint_parses_positive_selection_metadata(
    client, app_settings
):
    """GET /content should parse positive-selection metadata from row store columns."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(_20250615T080000Z),
                "end_utc": str(_20250615T080000Z + 10.0),
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "humpback": "1",
                "auto_positive_selection_score_source": "stored_diagnostics",
                "auto_positive_selection_decision": "positive",
                "auto_positive_selection_offsets": "[0,1,2]",
                "auto_positive_selection_raw_scores": "[0.2,0.9,0.95]",
                "auto_positive_selection_smoothed_scores": "[0.366667,0.683333,0.933333]",
                "auto_positive_selection_start_utc": str(_20250615T080000Z + 2.0),
                "auto_positive_selection_end_utc": str(_20250615T080000Z + 7.0),
                "auto_positive_selection_peak_score": "0.933333",
                "positive_selection_origin": "auto_selection",
                "positive_selection_score_source": "stored_diagnostics",
                "positive_selection_decision": "positive",
                "positive_selection_offsets": "[0,1,2]",
                "positive_selection_raw_scores": "[0.2,0.9,0.95]",
                "positive_selection_smoothed_scores": "[0.366667,0.683333,0.933333]",
                "positive_selection_start_utc": str(_20250615T080000Z + 2.0),
                "positive_selection_end_utc": str(_20250615T080000Z + 7.0),
                "positive_selection_peak_score": "0.933333",
                "positive_extract_filename": "20250615T080002Z_20250615T080007Z.flac",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert rows[0]["positive_selection_score_source"] == "stored_diagnostics"
    assert rows[0]["positive_selection_decision"] == "positive"
    assert rows[0]["auto_positive_selection_decision"] == "positive"
    assert rows[0]["positive_selection_offsets"] == [0.0, 1.0, 2.0]
    assert rows[0]["positive_selection_raw_scores"] == [0.2, 0.9, 0.95]
    assert rows[0]["positive_selection_smoothed_scores"] == [
        0.366667,
        0.683333,
        0.933333,
    ]
    assert rows[0]["positive_selection_start_utc"] == _20250615T080000Z + 2.0
    assert rows[0]["positive_selection_end_utc"] == _20250615T080000Z + 7.0
    assert (
        rows[0]["positive_extract_filename"] == "20250615T080002Z_20250615T080007Z.flac"
    )

    await engine.dispose()


async def test_content_uses_backfilled_auto_selection_for_legacy_job(
    client, app_settings
):
    """Legacy merged jobs should still surface backfilled auto selection on read paths."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detector import write_window_diagnostics
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    tsv_path = ddir / "detections.tsv"
    diagnostics_path = ddir / "window_diagnostics.parquet"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "humpback",
        "orca",
        "ship",
        "background",
    ]
    source_name = "20250615T080000Z.wav"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": source_name,
                "start_sec": "0.0",
                "end_sec": "10.0",
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
        )

    write_window_diagnostics(
        [
            {
                "filename": source_name,
                "window_index": idx,
                "offset_sec": float(offset),
                "end_sec": float(offset + 5),
                "confidence": conf,
                "is_overlapped": False,
                "overlap_sec": 0.0,
            }
            for idx, (offset, conf) in enumerate(
                [(0, 0.2), (1, 0.9), (2, 0.95), (3, 0.9), (4, 0.2), (5, 0.1)]
            )
        ],
        diagnostics_path,
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["start_utc"] == _20250615T080000Z
    assert rows[0]["auto_positive_selection_decision"] == "positive"
    assert rows[0]["auto_positive_selection_start_utc"] == _20250615T080000Z + 2.0
    assert rows[0]["auto_positive_selection_end_utc"] == _20250615T080000Z + 7.0
    assert rows[0]["positive_selection_origin"] is None
    assert rows[0]["positive_selection_start_utc"] is None

    await engine.dispose()


async def test_save_labels_rejects_legacy_merged_job(client, app_settings):
    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": BASE_EPOCH,
                "end_utc": BASE_EPOCH + 5.0,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 400
    assert "legacy merged-mode job" in resp.json()["detail"]

    await engine.dispose()


async def test_row_state_rejects_legacy_merged_job(client, app_settings):
    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/row-state",
        json={
            "start_utc": BASE_EPOCH,
            "end_utc": BASE_EPOCH + 5.0,
            "humpback": 1,
        },
    )
    assert resp.status_code == 400
    assert "legacy merged-mode job" in resp.json()["detail"]

    await engine.dispose()


async def test_extract_rejects_legacy_merged_job(client, app_settings):
    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/detection-jobs/extract",
        json={"job_ids": [job_id]},
    )
    assert resp.status_code == 400
    assert "legacy merged-mode job" in resp.json()["detail"]

    await engine.dispose()


async def test_overlap_rejected(client, app_settings):
    """Same embedding set ID in both pos and neg returns 400."""
    shared_id = "shared-set-id"
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "overlap-test",
            "positive_embedding_set_ids": [shared_id, "other-pos"],
            "negative_embedding_set_ids": [shared_id, "other-neg"],
        },
    )
    assert resp.status_code == 400
    assert "both positive and negative" in resp.json()["detail"]


async def test_content_endpoint_rejects_queued_job(client, app_settings):
    """GET /content returns 400 for queued jobs (no TSV yet)."""
    from sqlalchemy import insert
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="queued",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 400

    await engine.dispose()


async def test_save_labels_rejects_invalid_values(client, app_settings):
    """PUT /labels rejects values outside {0, 1, null}."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": BASE_EPOCH,
                "end_utc": BASE_EPOCH + 5.0,
                "humpback": 2,
            }
        ],
    )
    assert resp.status_code == 422

    await engine.dispose()


async def test_save_labels_preserves_positive_extract_filename(client, app_settings):
    """PUT /labels preserves positive_extract_filename in the row store."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import (
        read_detection_row_store,
        write_detection_row_store,
    )
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    row_store_path = ddir / "detection_rows.parquet"

    # Row identity uses UTC epochs derived from 20250702T080118Z + offsets
    row_start_utc = _20250702T080118Z + 37.0
    row_end_utc = _20250702T080118Z + 45.0
    write_detection_row_store(
        row_store_path,
        [
            {
                "start_utc": str(row_start_utc),
                "end_utc": str(row_end_utc),
                "avg_confidence": "0.951",
                "peak_confidence": "0.970",
                "n_windows": "4",
                "auto_positive_selection_score_source": "windowed_peak",
                "auto_positive_selection_decision": "positive",
                "auto_positive_selection_offsets": f"[{row_start_utc}]",
                "auto_positive_selection_start_utc": str(row_start_utc),
                "auto_positive_selection_end_utc": str(row_end_utc),
                "auto_positive_selection_peak_score": "0.970",
                "positive_extract_filename": "20250702T080157Z_20250702T080202Z.flac",
                "humpback": "",
                "ship": "",
                "background": "",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(row_store_path),
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": row_start_utc,
                "end_utc": row_end_utc,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 200

    # Verify the row store was updated
    _fieldnames, rows = read_detection_row_store(row_store_path)
    assert len(rows) == 1
    assert rows[0]["positive_selection_decision"] == "positive"
    assert rows[0]["positive_selection_origin"] == "auto_selection"
    # Effective selection copies from auto when humpback=1
    assert rows[0]["positive_selection_start_utc"]
    assert rows[0]["positive_selection_end_utc"]
    assert (
        rows[0]["positive_extract_filename"] == "20250702T080157Z_20250702T080202Z.flac"
    )
    assert rows[0]["humpback"] == "1"

    await engine.dispose()


async def test_row_state_endpoint_persists_manual_selection(client, app_settings):
    """PUT /row-state should atomically persist labels plus manual window bounds."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import (
        read_detection_row_store,
        write_detection_row_store,
    )
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    row_store_path = ddir / "detection_rows.parquet"

    row_start = _20250615T080000Z
    row_end = _20250615T080000Z + 10.0
    write_detection_row_store(
        row_store_path,
        [
            {
                "start_utc": str(row_start),
                "end_utc": str(row_end),
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(row_store_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    row = resp.json()[0]

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/row-state",
        json={
            "start_utc": row["start_utc"],
            "end_utc": row["end_utc"],
            "humpback": 1,
            "orca": None,
            "ship": None,
            "background": None,
            "manual_positive_selection_start_utc": row_start,
            "manual_positive_selection_end_utc": row_end,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["row"]["humpback"] == 1
    assert payload["row"]["positive_selection_origin"] == "manual_override"
    assert payload["row"]["positive_selection_start_utc"] == row_start
    assert payload["row"]["positive_selection_end_utc"] == row_end
    assert payload["row"]["manual_positive_selection_start_utc"] == row_start
    assert payload["row"]["manual_positive_selection_end_utc"] == row_end

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["positive_selection_origin"] == "manual_override"
    assert row["manual_positive_selection_start_utc"] == row_start
    assert row["manual_positive_selection_end_utc"] == row_end

    # Verify the row store was updated
    _fieldnames, saved_rows = read_detection_row_store(row_store_path)
    assert saved_rows[0]["manual_positive_selection_start_utc"] != ""
    assert saved_rows[0]["manual_positive_selection_end_utc"] != ""
    assert saved_rows[0]["positive_selection_origin"] == "manual_override"

    await engine.dispose()


async def test_row_state_accepts_non_edge_aligned_window_multiple(client, app_settings):
    """Manual bounds may start/end off the clip edges when duration stays at 5*N."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    row_store_path = ddir / "detection_rows.parquet"

    row_start = _20250615T080000Z
    row_end = _20250615T080000Z + 15.0  # 15s clip so manual 3..13 fits inside
    write_detection_row_store(
        row_store_path,
        [
            {
                "start_utc": str(row_start),
                "end_utc": str(row_end),
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(row_store_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    row = resp.json()[0]

    manual_start = row_start + 3.0
    manual_end = row_start + 13.0
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/row-state",
        json={
            "start_utc": row["start_utc"],
            "end_utc": row["end_utc"],
            "humpback": 1,
            "manual_positive_selection_start_utc": manual_start,
            "manual_positive_selection_end_utc": manual_end,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["row"]["positive_selection_origin"] == "manual_override"
    assert payload["row"]["manual_positive_selection_start_utc"] == manual_start
    assert payload["row"]["manual_positive_selection_end_utc"] == manual_end
    assert payload["row"]["positive_selection_start_utc"] == manual_start
    assert payload["row"]["positive_selection_end_utc"] == manual_end

    await engine.dispose()


async def test_content_and_download_use_row_store_when_tsv_missing(
    client, app_settings
):
    """Completed jobs should still serve content/download from the row store alone."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    row_store_path = ddir / "detection_rows.parquet"

    write_detection_row_store(
        row_store_path,
        [
            {
                "start_utc": str(_20250615T080000Z),
                "end_utc": str(_20250615T080000Z + 10.0),
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                output_row_store_path=str(row_store_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["start_utc"] == _20250615T080000Z
    assert rows[0]["end_utc"] == _20250615T080000Z + 10.0

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/download")
    assert resp.status_code == 200
    assert "start_utc\tend_utc" in resp.text

    await engine.dispose()


async def test_labels_and_row_state_update_row_store(client, app_settings):
    """Editing labels and row state should persist to the Parquet row store."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import (
        read_detection_row_store,
        write_detection_row_store,
    )
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    row_start = _20250615T080000Z
    row_end = _20250615T080000Z + 10.0

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    row_store_path = ddir / "detection_rows.parquet"

    write_detection_row_store(
        row_store_path,
        [
            {
                "start_utc": str(row_start),
                "end_utc": str(row_end),
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(row_store_path),
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": row_start,
                "end_utc": row_end,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 200
    _fieldnames, saved_rows = read_detection_row_store(row_store_path)
    assert saved_rows[0]["humpback"] == "1"

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/row-state",
        json={
            "start_utc": row_start,
            "end_utc": row_end,
            "humpback": 1,
            "manual_positive_selection_start_utc": row_start,
            "manual_positive_selection_end_utc": row_end,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["row"]["positive_selection_origin"] == "manual_override"
    assert payload["row"]["manual_positive_selection_start_utc"] == row_start
    assert payload["row"]["manual_positive_selection_end_utc"] == row_end

    _fieldnames, saved_rows = read_detection_row_store(row_store_path)
    assert saved_rows[0]["manual_positive_selection_start_utc"] != ""
    assert saved_rows[0]["manual_positive_selection_end_utc"] != ""
    assert saved_rows[0]["positive_selection_origin"] == "manual_override"

    await engine.dispose()


# ---- Spectrogram Endpoint ----


async def test_spectrogram_not_found(client):
    """404 for nonexistent job."""
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/spectrogram",
        params={"start_utc": BASE_EPOCH, "duration_sec": 5},
    )
    assert resp.status_code == 404


async def test_spectrogram_returns_png(client, app_settings, wav_bytes):
    """Spectrogram endpoint returns valid PNG for a local detection job."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    # Write a WAV file to a temp audio folder
    audio_folder = Path(app_settings.storage_root) / "audio_test"
    audio_folder.mkdir(parents=True)
    wav_path = audio_folder / "test.wav"
    wav_path.write_bytes(wav_bytes)

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder=str(audio_folder),
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    # test.wav has no parseable timestamp so base_epoch = file mtime;
    # use start_utc > 0 but rely on local resolution picking the only file
    import os

    wav_mtime = os.path.getmtime(wav_path)
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/spectrogram",
        params={"start_utc": wav_mtime, "duration_sec": 1},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b"\x89PNG"

    await engine.dispose()


async def test_extracted_sidecar_png_matches_spectrogram_endpoint(client, app_settings):
    """Extracted sidecar PNGs should match the marker-free UI spectrogram image."""
    import io
    import math
    import struct
    import wave
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detector import write_window_diagnostics
    from humpback.classifier.extractor import extract_labeled_samples
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    def _make_wav_bytes(duration: float, sample_rate: int = 16000) -> bytes:
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

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    audio_folder = Path(app_settings.storage_root) / "audio_test"
    audio_folder.mkdir(parents=True)
    source_name = "20250615T080000Z_test.wav"
    wav_path = audio_folder / source_name
    wav_path.write_bytes(_make_wav_bytes(duration=12.0))

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "start_sec",
                "end_sec",
                "avg_confidence",
                "peak_confidence",
                "humpback",
                "orca",
                "ship",
                "background",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": source_name,
                "start_sec": "0.0",
                "end_sec": "10.0",
                "avg_confidence": "0.9",
                "peak_confidence": "0.95",
                "humpback": "1",
                "orca": "",
                "ship": "",
                "background": "",
            }
        )

    diagnostics_path = tsv_dir / "window_diagnostics.parquet"
    write_window_diagnostics(
        [
            {
                "filename": source_name,
                "window_index": idx,
                "offset_sec": float(offset),
                "end_sec": float(offset + 5),
                "confidence": conf,
                "is_overlapped": False,
                "overlap_sec": 0.0,
            }
            for idx, (offset, conf) in enumerate(
                [(0, 0.2), (1, 0.9), (2, 0.95), (3, 0.9), (4, 0.2), (5, 0.1)]
            )
        ],
        diagnostics_path,
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder=str(audio_folder),
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    extract_labeled_samples(
        tsv_path=tsv_path,
        audio_folder=audio_folder,
        positive_output_path=Path(app_settings.storage_root) / "labeled" / "positives",
        negative_output_path=Path(app_settings.storage_root) / "labeled" / "negatives",
        window_diagnostics_path=diagnostics_path,
    )

    saved_pngs = list(
        (Path(app_settings.storage_root) / "labeled" / "positives").rglob("*.png")
    )
    assert len(saved_pngs) == 1

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/spectrogram",
        params={"start_utc": _20250615T080000Z + 2.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 200
    assert resp.content == saved_pngs[0].read_bytes()

    await engine.dispose()


# ---- has_positive_labels flag ----


async def test_save_labels_sets_has_positive_labels_true(client, app_settings):
    """Saving labels with humpback=1 sets has_positive_labels to True."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            },
            {
                "start_utc": str(BASE_EPOCH + 5.0),
                "end_utc": str(BASE_EPOCH + 10.0),
                "avg_confidence": "0.7",
                "peak_confidence": "0.8",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            },
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    # Save one row as humpback=1
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[{"start_utc": BASE_EPOCH, "end_utc": BASE_EPOCH + 5.0, "humpback": 1}],
    )
    assert resp.status_code == 200

    # Verify flag is True
    resp2 = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["has_positive_labels"] is True

    await engine.dispose()


async def test_save_labels_clears_has_positive_labels(client, app_settings):
    """Clearing all positive labels sets has_positive_labels to False."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "1",
                "ship": "",
                "background": "",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                has_positive_labels=True,
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    # Clear the humpback label
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[{"start_utc": BASE_EPOCH, "end_utc": BASE_EPOCH + 5.0, "humpback": None}],
    )
    assert resp.status_code == 200

    # Verify flag is now False
    resp2 = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["has_positive_labels"] is False

    await engine.dispose()


async def test_partial_save_preserves_positive_flag_from_other_rows(
    client, app_settings
):
    """Partial save updating non-humpback labels preserves flag from other rows."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "1",
                "ship": "",
                "background": "",
            },
            {
                "start_utc": str(BASE_EPOCH + 5.0),
                "end_utc": str(BASE_EPOCH + 10.0),
                "avg_confidence": "0.6",
                "peak_confidence": "0.7",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            },
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    # Only save ship label on second row (don't touch the humpback=1 row)
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[{"start_utc": BASE_EPOCH + 5.0, "end_utc": BASE_EPOCH + 10.0, "ship": 1}],
    )
    assert resp.status_code == 200

    # Flag should still be True because first row has humpback=1
    resp2 = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["has_positive_labels"] is True

    await engine.dispose()


async def test_save_orca_label_sets_has_positive_labels(client, app_settings):
    """Saving labels with orca=1 sets has_positive_labels to True."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    # Save orca=1
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[{"start_utc": BASE_EPOCH, "end_utc": BASE_EPOCH + 5.0, "orca": 1}],
    )
    assert resp.status_code == 200

    # Verify flag is True
    resp2 = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["has_positive_labels"] is True

    await engine.dispose()


async def test_orca_label_round_trip(client, app_settings):
    """Orca label round-trips through PUT labels and GET content."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
        ],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    # Save orca=1, humpback=0
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": BASE_EPOCH,
                "end_utc": BASE_EPOCH + 5.0,
                "humpback": 0,
                "orca": 1,
            }
        ],
    )
    assert resp.status_code == 200

    # Read back content and verify orca label
    resp2 = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp2.status_code == 200
    rows = resp2.json()
    assert len(rows) == 1
    assert rows[0]["orca"] == 1
    assert rows[0]["humpback"] == 0

    await engine.dispose()


# ---- Detection-Manifest Training ----


async def _seed_detection_manifest_fixtures(app_settings):
    """Seed DB + disk for a detection-manifest training job test.

    Returns (detection_job_id, model_version).
    """
    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import ClassifierModel, DetectionJob
    from humpback.models.model_registry import ModelConfig
    from humpback.storage import detection_embeddings_path, detection_row_store_path

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    model_version = "perch_v2"
    cm_id = str(uuid.uuid4())
    dj_id = str(uuid.uuid4())

    async with sf() as session:
        session.add(
            ModelConfig(
                name=model_version,
                display_name="Perch v2 (TFLite)",
                path="models/perch_v2.tflite",
                model_type="tflite",
                input_format="waveform",
                vector_dim=1536,
                is_default=False,
            )
        )
        session.add(
            ClassifierModel(
                id=cm_id,
                name="source-classifier",
                model_path="/tmp/fake.joblib",
                model_version=model_version,
                vector_dim=1536,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        session.add(
            DetectionJob(
                id=dj_id,
                status="complete",
                classifier_model_id=cm_id,
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
            )
        )
        await session.commit()

    # Write row store with one positive + one negative row.
    rs_path = detection_row_store_path(app_settings.storage_root, dj_id)
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.9",
                "peak_confidence": "0.95",
                "n_windows": "1",
                "humpback": "1",
            },
            {
                "start_utc": str(BASE_EPOCH + 5.0),
                "end_utc": str(BASE_EPOCH + 10.0),
                "avg_confidence": "0.3",
                "peak_confidence": "0.4",
                "n_windows": "1",
                "background": "1",
            },
        ],
    )

    # Write embeddings parquet at the model-versioned path.
    emb_path = detection_embeddings_path(
        app_settings.storage_root, dj_id, model_version
    )
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    _write_detection_embeddings_parquet(
        emb_path,
        row_ids=["r1", "r2"],
        rows=[[0.1] * 4, [0.2] * 4],
    )

    await engine.dispose()
    return dj_id, model_version


async def test_create_training_job_detection_manifest_success(client, app_settings):
    """POST /training-jobs with detection_job_ids creates a detection_manifest job."""
    dj_id, model_version = await _seed_detection_manifest_fixtures(app_settings)

    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "perch-v2-test",
            "detection_job_ids": [dj_id],
            "embedding_model_version": model_version,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["source_mode"] == "detection_manifest"
    assert data["model_version"] == model_version
    assert data["source_detection_job_ids"] == [dj_id]
    assert data["status"] == "queued"
    assert data["window_size_seconds"] == 5.0
    assert data["target_sample_rate"] == 32000


async def test_create_training_job_mixed_sources_rejected(client):
    """POST /training-jobs with both embedding sets and detection jobs -> 422."""
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "mixed",
            "positive_embedding_set_ids": ["es-1"],
            "negative_embedding_set_ids": ["es-2"],
            "detection_job_ids": ["dj-1"],
            "embedding_model_version": "perch_v2",
        },
    )
    assert resp.status_code == 422
    assert "Cannot mix" in resp.text


async def test_create_training_job_detection_manifest_missing_embeddings(
    client, app_settings
):
    """POST /training-jobs with detection_job_ids but no embeddings -> 400."""
    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import ClassifierModel, DetectionJob
    from humpback.models.model_registry import ModelConfig
    from humpback.storage import detection_row_store_path

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    cm_id = str(uuid.uuid4())
    dj_id = str(uuid.uuid4())

    async with sf() as session:
        session.add(
            ModelConfig(
                name="perch_v2_missing",
                display_name="Perch v2",
                path="models/perch_v2.tflite",
                model_type="tflite",
                input_format="waveform",
                vector_dim=1536,
                is_default=False,
            )
        )
        session.add(
            ClassifierModel(
                id=cm_id,
                name="no-emb-classifier",
                model_path="/tmp/fake.joblib",
                model_version="perch_v2_missing",
                vector_dim=1536,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        session.add(
            DetectionJob(
                id=dj_id,
                status="complete",
                classifier_model_id=cm_id,
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                detection_mode="windowed",
            )
        )
        await session.commit()

    # Write a row store but NO embeddings.
    rs_path = detection_row_store_path(app_settings.storage_root, dj_id)
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_row_store(
        rs_path,
        [
            {
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
                "avg_confidence": "0.9",
                "peak_confidence": "0.95",
                "n_windows": "1",
                "humpback": "1",
            },
        ],
    )

    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "no-embeddings",
            "detection_job_ids": [dj_id],
            "embedding_model_version": "perch_v2_missing",
        },
    )
    assert resp.status_code == 400
    assert "no embeddings" in resp.json()["detail"].lower()

    await engine.dispose()


async def test_create_training_job_detection_manifest_roundtrip(client, app_settings):
    """Detection-manifest job is visible via GET list and detail endpoints."""
    dj_id, model_version = await _seed_detection_manifest_fixtures(app_settings)

    create_resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "roundtrip-test",
            "detection_job_ids": [dj_id],
            "embedding_model_version": model_version,
        },
    )
    assert create_resp.status_code == 201
    job_id = create_resp.json()["id"]

    # GET detail
    detail_resp = await client.get(f"/classifier/training-jobs/{job_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["source_mode"] == "detection_manifest"
    assert detail["source_detection_job_ids"] == [dj_id]

    # GET list
    list_resp = await client.get("/classifier/training-jobs")
    assert list_resp.status_code == 200
    ids = [j["id"] for j in list_resp.json()]
    assert job_id in ids


# ---- Detection Embedding Endpoint ----


async def test_detection_embedding_404_for_nonexistent_job(client):
    """GET /classifier/detection-jobs/{id}/embedding returns 404 for missing job."""
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/embedding",
        params={"row_id": "some-row"},
    )
    assert resp.status_code == 404


async def test_detection_embedding_404_for_job_without_embeddings(client, app_settings):
    """GET /classifier/detection-jobs/{id}/embedding returns 404 when no embeddings exist."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    # Create a completed job with a row store but no embeddings parquet
    from humpback.classifier.detection_rows import write_detection_row_store

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    rs_path = ddir / "detection_rows.parquet"
    write_detection_row_store(
        rs_path,
        [{"start_utc": str(BASE_EPOCH), "end_utc": str(BASE_EPOCH + 5.0)}],
    )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                output_row_store_path=str(rs_path),
            )
        )
        await session.commit()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding",
        params={"row_id": "some-row"},
    )
    assert resp.status_code == 404
    assert "no stored embeddings" in resp.json()["detail"].lower()

    await engine.dispose()


async def test_detection_embedding_404_for_job_without_output(client, app_settings):
    """GET /classifier/detection-jobs/{id}/embedding returns 404 when no output path."""
    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding",
        params={"row_id": "some-row"},
    )
    assert resp.status_code == 404
    assert "no stored embeddings" in resp.json()["detail"].lower()

    await engine.dispose()


async def test_detection_embedding_returns_vector(client, app_settings):
    """GET /classifier/detection-jobs/{id}/embedding returns embedding when available."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detector import write_detection_embeddings
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import ClassifierModel, DetectionJob

    job_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    emb_dir = ddir / "embeddings" / "perch_v1"
    emb_dir.mkdir(parents=True)

    test_row_id = "emb-test-row-1"
    embedding_records = [
        {
            "row_id": test_row_id,
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "confidence": 0.85,
        }
    ]
    emb_path = emb_dir / "detection_embeddings.parquet"
    write_detection_embeddings(embedding_records, emb_path)

    async with sf() as session:
        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="test-model",
                model_path="/tmp/fake.pkl",
                model_version="perch_v1",
                vector_dim=4,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id=model_id,
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding",
        params={"row_id": test_row_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "perch_v1"
    assert data["vector_dim"] == 4
    assert len(data["vector"]) == 4
    import pytest as _pt

    assert data["vector"][0] == _pt.approx(0.1, abs=1e-5)

    await engine.dispose()


async def test_detection_embedding_hydrophone_job_row_id(client, app_settings):
    """Hydrophone embedding lookup works with row_id."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detector import write_detection_embeddings
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import ClassifierModel, DetectionJob

    job_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    job_start_ts = 1751439600.0
    hydro_row_id = "hydro-emb-row-1"

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    emb_dir = ddir / "embeddings" / "perch_v1"
    emb_dir.mkdir(parents=True)

    emb_path = emb_dir / "detection_embeddings.parquet"
    write_detection_embeddings(
        [
            {
                "row_id": hydro_row_id,
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.85,
            }
        ],
        emb_path,
    )

    async with sf() as session:
        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="test-model",
                model_path="/tmp/fake.pkl",
                model_version="perch_v1",
                vector_dim=4,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id=model_id,
                hydrophone_id="rpi_orcasound_lab",
                start_timestamp=job_start_ts,
                end_timestamp=job_start_ts + 3600,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding",
        params={"row_id": hydro_row_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "perch_v1"
    assert data["vector_dim"] == 4
    assert len(data["vector"]) == 4

    await engine.dispose()
