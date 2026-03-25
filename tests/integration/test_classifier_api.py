"""Integration tests for classifier API endpoints."""

import csv
import json
import uuid


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
                output_tsv_path="/tmp/fake.tsv",
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
                "row_id": "test-row-1",
                "filename": "test.wav",
                "start_sec": "1.000000",
                "end_sec": "6.000000",
                "avg_confidence": "0.850000",
                "peak_confidence": "0.900000",
                "n_windows": "2",
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
                output_tsv_path=str(ddir / "detections.tsv"),
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
    assert rows[0]["filename"] == "test.wav"
    assert rows[0]["avg_confidence"] == 0.85
    assert rows[0]["raw_start_sec"] == 1.0
    assert rows[0]["raw_end_sec"] == 6.0
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
    """GET /content should parse positive-selection metadata from TSV columns."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    tsv_path = ddir / "detections.tsv"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "humpback",
        "positive_selection_score_source",
        "positive_selection_decision",
        "positive_selection_offsets",
        "positive_selection_raw_scores",
        "positive_selection_smoothed_scores",
        "positive_selection_start_sec",
        "positive_selection_end_sec",
        "positive_selection_peak_score",
        "positive_extract_filename",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "20250615T080000Z.wav",
                "start_sec": "0.0",
                "end_sec": "10.0",
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "humpback": "1",
                "positive_selection_score_source": "stored_diagnostics",
                "positive_selection_decision": "positive",
                "positive_selection_offsets": "[0,1,2]",
                "positive_selection_raw_scores": "[0.2,0.9,0.95]",
                "positive_selection_smoothed_scores": "[0.366667,0.683333,0.933333]",
                "positive_selection_start_sec": "2.000000",
                "positive_selection_end_sec": "7.000000",
                "positive_selection_peak_score": "0.933333",
                "positive_extract_filename": "20250615T080002Z_20250615T080007Z.flac",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                output_tsv_path=str(tsv_path),
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
    assert rows[0]["positive_selection_start_sec"] == 2.0
    assert rows[0]["positive_selection_end_sec"] == 7.0
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["row_id"]
    assert rows[0]["auto_positive_selection_decision"] == "positive"
    assert rows[0]["auto_positive_selection_start_sec"] == 2.0
    assert rows[0]["auto_positive_selection_end_sec"] == 7.0
    assert rows[0]["positive_selection_origin"] is None
    assert rows[0]["positive_selection_start_sec"] is None

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
                "filename": "test.wav",
                "start_sec": 0.0,
                "end_sec": 5.0,
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
            "row_id": "row-1",
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

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "humpback",
        "ship",
        "background",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "test.wav",
                "start_sec": "0.0",
                "end_sec": "5.0",
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "filename": "test.wav",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "humpback": 2,
            }
        ],
    )
    assert resp.status_code == 422

    await engine.dispose()


async def test_save_labels_preserves_extract_filename_column(client, app_settings):
    """PUT /labels preserves extra TSV columns such as extraction provenance."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import read_detection_row_store
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    row_store_path = tsv_dir / "detection_rows.parquet"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "detection_filename",
        "extract_filename",
        "positive_selection_decision",
        "positive_selection_offsets",
        "positive_extract_filename",
        "humpback",
        "ship",
        "background",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "20250702T080118Z.wav",
                "start_sec": "37.0",
                "end_sec": "45.0",
                "avg_confidence": "0.951",
                "peak_confidence": "0.970",
                "n_windows": "4",
                "detection_filename": "20250702T080155Z_20250702T080203Z.flac",
                "extract_filename": "20250702T080155Z_20250702T080205Z.flac",
                "positive_selection_decision": "positive",
                "positive_selection_offsets": "[37,38,39]",
                "positive_extract_filename": "20250702T080157Z_20250702T080202Z.flac",
                "humpback": "",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "filename": "20250702T080118Z.wav",
                "start_sec": 37.0,
                "end_sec": 45.0,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 200

    # TSV is no longer synced on write; verify the row store was updated instead
    _fieldnames, rows = read_detection_row_store(row_store_path)
    assert len(rows) == 1
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080203Z.flac"
    assert rows[0]["extract_filename"] == "20250702T080155Z_20250702T080205Z.flac"
    assert rows[0]["positive_selection_decision"] == "positive"
    assert rows[0]["positive_selection_origin"] == "auto_selection"
    assert rows[0]["positive_selection_start_sec"] == "37.000000"
    assert rows[0]["positive_selection_end_sec"] == "45.000000"
    assert rows[0]["positive_selection_offsets"] == "[37.0]"
    assert (
        rows[0]["positive_extract_filename"] == "20250702T080157Z_20250702T080202Z.flac"
    )
    assert rows[0]["humpback"] == "1"

    await engine.dispose()


async def test_row_state_endpoint_persists_manual_selection(client, app_settings):
    """PUT /row-state should atomically persist labels plus manual window bounds."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.classifier.detection_rows import read_detection_row_store
    from humpback.classifier.detector import write_window_diagnostics
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    tsv_path = ddir / "detections.tsv"
    row_store_path = ddir / "detection_rows.parquet"
    diagnostics_path = ddir / "window_diagnostics.parquet"
    source_name = "20250615T080000Z.wav"

    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
            delimiter="\t",
        )
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
                detection_mode="windowed",
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    row = resp.json()[0]

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/row-state",
        json={
            "row_id": row["row_id"],
            "humpback": 1,
            "orca": None,
            "ship": None,
            "background": None,
            "manual_positive_selection_start_sec": 0.0,
            "manual_positive_selection_end_sec": 10.0,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["row"]["humpback"] == 1
    assert payload["row"]["positive_selection_origin"] == "manual_override"
    assert payload["row"]["positive_selection_start_sec"] == 0.0
    assert payload["row"]["positive_selection_end_sec"] == 10.0
    assert payload["row"]["manual_positive_selection_start_sec"] == 0.0
    assert payload["row"]["manual_positive_selection_end_sec"] == 10.0

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["positive_selection_origin"] == "manual_override"
    assert row["manual_positive_selection_start_sec"] == 0.0
    assert row["manual_positive_selection_end_sec"] == 10.0

    # TSV is no longer synced on write; verify the row store was updated instead
    _fieldnames, saved_rows = read_detection_row_store(row_store_path)
    assert saved_rows[0]["manual_positive_selection_start_sec"] == "0.000000"
    assert saved_rows[0]["manual_positive_selection_end_sec"] == "10.000000"
    assert saved_rows[0]["positive_selection_origin"] == "manual_override"

    await engine.dispose()


async def test_row_state_accepts_non_edge_aligned_window_multiple(client, app_settings):
    """Manual bounds may start/end off the clip edges when duration stays at 5*N."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    tsv_path = ddir / "detections.tsv"

    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "start_sec",
                "end_sec",
                "avg_confidence",
                "peak_confidence",
                "n_windows",
                "detection_filename",
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
                "filename": "20250615T080000Z.wav",
                "start_sec": "0.0",
                "end_sec": "10.0",
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "detection_filename": "20250615T080000Z_20250615T080015Z.flac",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    row = resp.json()[0]

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/row-state",
        json={
            "row_id": row["row_id"],
            "humpback": 1,
            "manual_positive_selection_start_sec": 3.0,
            "manual_positive_selection_end_sec": 13.0,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["row"]["positive_selection_origin"] == "manual_override"
    assert payload["row"]["manual_positive_selection_start_sec"] == 3.0
    assert payload["row"]["manual_positive_selection_end_sec"] == 13.0
    assert payload["row"]["positive_selection_start_sec"] == 3.0
    assert payload["row"]["positive_selection_end_sec"] == 13.0

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
    tsv_path = ddir / "detections.tsv"
    row_store_path = ddir / "detection_rows.parquet"

    write_detection_row_store(
        row_store_path,
        [
            {
                "row_id": "row-1",
                "filename": "20250615T080000Z.wav",
                "start_sec": "0.000000",
                "end_sec": "10.000000",
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "detection_filename": "20250615T080000Z_20250615T080010Z.flac",
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
                output_tsv_path=str(tsv_path),
                output_row_store_path=str(row_store_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["row_id"] == "row-1"
    assert rows[0]["filename"] == "20250615T080000Z.wav"

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/download")
    assert resp.status_code == 200
    assert "filename\tstart_sec\tend_sec" in resp.text
    assert "20250615T080000Z.wav" in resp.text

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

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ddir = Path(app_settings.storage_root) / "detections" / job_id
    ddir.mkdir(parents=True)
    tsv_path = ddir / "detections.tsv"
    row_store_path = ddir / "detection_rows.parquet"

    write_detection_row_store(
        row_store_path,
        [
            {
                "row_id": "row-1",
                "filename": "20250615T080000Z.wav",
                "start_sec": "0.000000",
                "end_sec": "10.000000",
                "avg_confidence": "0.85",
                "peak_confidence": "0.95",
                "n_windows": "6",
                "detection_filename": "20250615T080000Z_20250615T080010Z.flac",
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
                output_tsv_path=str(tsv_path),
                output_row_store_path=str(row_store_path),
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "filename": "20250615T080000Z.wav",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 200
    # TSV is no longer synced on write; verify the row store was updated instead
    assert not tsv_path.is_file()
    _fieldnames, saved_rows = read_detection_row_store(row_store_path)
    assert saved_rows[0]["humpback"] == "1"

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/row-state",
        json={
            "row_id": "row-1",
            "humpback": 1,
            "manual_positive_selection_start_sec": 0.0,
            "manual_positive_selection_end_sec": 10.0,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["row"]["positive_selection_origin"] == "manual_override"
    assert payload["row"]["manual_positive_selection_start_sec"] == 0.0
    assert payload["row"]["manual_positive_selection_end_sec"] == 10.0

    _fieldnames, saved_rows = read_detection_row_store(row_store_path)
    assert saved_rows[0]["manual_positive_selection_start_sec"] == "0.000000"
    assert saved_rows[0]["manual_positive_selection_end_sec"] == "10.000000"
    assert saved_rows[0]["positive_selection_origin"] == "manual_override"

    await engine.dispose()


# ---- Spectrogram Endpoint ----


async def test_spectrogram_not_found(client):
    """404 for nonexistent job."""
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/spectrogram",
        params={"filename": "test.wav", "start_sec": 0, "duration_sec": 5},
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

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/spectrogram",
        params={"filename": "test.wav", "start_sec": 0, "duration_sec": 1},
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
                output_tsv_path=str(tsv_path),
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
        params={"filename": source_name, "start_sec": 2.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 200
    assert resp.content == saved_pngs[0].read_bytes()

    await engine.dispose()


# ---- has_positive_labels flag ----


async def test_save_labels_sets_has_positive_labels_true(client, app_settings):
    """Saving labels with humpback=1 sets has_positive_labels to True."""
    from pathlib import Path

    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "humpback",
        "ship",
        "background",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "a.wav",
                "start_sec": "0.0",
                "end_sec": "5.0",
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            }
        )
        writer.writerow(
            {
                "filename": "a.wav",
                "start_sec": "5.0",
                "end_sec": "10.0",
                "avg_confidence": "0.7",
                "peak_confidence": "0.8",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    # Save one row as humpback=1
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[{"filename": "a.wav", "start_sec": 0.0, "end_sec": 5.0, "humpback": 1}],
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

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "humpback",
        "ship",
        "background",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "a.wav",
                "start_sec": "0.0",
                "end_sec": "5.0",
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "1",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
                has_positive_labels=True,
            )
        )
        await session.commit()

    # Clear the humpback label
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {"filename": "a.wav", "start_sec": 0.0, "end_sec": 5.0, "humpback": None}
        ],
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

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "humpback",
        "ship",
        "background",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "a.wav",
                "start_sec": "0.0",
                "end_sec": "5.0",
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "1",
                "ship": "",
                "background": "",
            }
        )
        writer.writerow(
            {
                "filename": "a.wav",
                "start_sec": "5.0",
                "end_sec": "10.0",
                "avg_confidence": "0.6",
                "peak_confidence": "0.7",
                "n_windows": "1",
                "humpback": "",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    # Only save ship label on second row (don't touch the humpback=1 row)
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[{"filename": "a.wav", "start_sec": 5.0, "end_sec": 10.0, "ship": 1}],
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

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
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
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "a.wav",
                "start_sec": "0.0",
                "end_sec": "5.0",
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    # Save orca=1
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[{"filename": "a.wav", "start_sec": 0.0, "end_sec": 5.0, "orca": 1}],
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

    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
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
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "a.wav",
                "start_sec": "0.0",
                "end_sec": "5.0",
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "1",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "",
            }
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    # Save orca=1, humpback=0
    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "filename": "a.wav",
                "start_sec": 0.0,
                "end_sec": 5.0,
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


# ---- Detection Embedding Endpoint ----


async def test_detection_embedding_404_for_nonexistent_job(client):
    """GET /classifier/detection-jobs/{id}/embedding returns 404 for missing job."""
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/embedding",
        params={"filename": "test.wav", "start_sec": 0.0, "end_sec": 5.0},
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

    # Create a completed job with a TSV but no embeddings parquet
    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "start_sec", "end_sec"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({"filename": "test.wav", "start_sec": "0.0", "end_sec": "5.0"})

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding",
        params={"filename": "test.wav", "start_sec": 0.0, "end_sec": 5.0},
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
                # no output_tsv_path
            )
        )
        await session.commit()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding",
        params={"filename": "test.wav", "start_sec": 0.0, "end_sec": 5.0},
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

    tsv_dir = Path(app_settings.storage_root) / "detections" / job_id
    tsv_dir.mkdir(parents=True)
    tsv_path = tsv_dir / "detections.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "start_sec", "end_sec"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({"filename": "song.wav", "start_sec": "0.0", "end_sec": "5.0"})

    # Write detection embeddings
    embedding_records = [
        {
            "filename": "song.wav",
            "start_sec": 0.0,
            "end_sec": 5.0,
            "embedding": [0.1, 0.2, 0.3, 0.4],
        }
    ]
    emb_path = tsv_dir / "detection_embeddings.parquet"
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding",
        params={"filename": "song.wav", "start_sec": 0.0, "end_sec": 5.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "perch_v1"
    assert data["vector_dim"] == 4
    assert len(data["vector"]) == 4
    import pytest as _pt

    assert data["vector"][0] == _pt.approx(0.1, abs=1e-5)

    await engine.dispose()
