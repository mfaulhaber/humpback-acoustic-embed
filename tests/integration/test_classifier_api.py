"""Integration tests for classifier API endpoints."""

import csv
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
    """GET /content returns partial results when job is running with TSV on disk."""
    from pathlib import Path
    from sqlalchemy import insert
    from humpback.database import create_engine, create_session_factory
    from humpback.models.classifier import DetectionJob

    # Create a running detection job directly in the DB
    job_id = str(uuid.uuid4())
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    # Write a partial TSV to disk
    storage_root = Path(app_settings.storage_root)
    ddir = storage_root / "detection" / job_id
    ddir.mkdir(parents=True)
    tsv_path = ddir / "detections.tsv"
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "test.wav",
                "start_sec": "1.0",
                "end_sec": "6.0",
                "avg_confidence": "0.85",
                "peak_confidence": "0.9",
                "n_windows": "2",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="running",
                classifier_model_id="fake-model-id",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
                output_tsv_path=str(tsv_path),
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

    # Job list should include progress fields
    resp = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert resp.status_code == 200
    job_data = resp.json()
    assert job_data["files_processed"] == 1
    assert job_data["files_total"] == 5
    assert job_data["status"] == "running"

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
    """PUT /labels preserves extra TSV columns such as extract_filename."""
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
        "detection_filename",
        "extract_filename",
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
                "detection_filename": "20250702T080155Z_20250702T080203Z.wav",
                "extract_filename": "20250702T080155Z_20250702T080205Z.wav",
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

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "detection_filename" in reader.fieldnames
    assert "extract_filename" in reader.fieldnames
    assert len(rows) == 1
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080203Z.wav"
    assert rows[0]["extract_filename"] == "20250702T080155Z_20250702T080205Z.wav"
    assert rows[0]["humpback"] == "1"

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


# ---- has_humpback_labels flag ----


async def test_save_labels_sets_has_humpback_labels_true(client, app_settings):
    """Saving labels with humpback=1 sets has_humpback_labels to True."""
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
    assert resp2.json()["has_humpback_labels"] is True

    await engine.dispose()


async def test_save_labels_clears_has_humpback_labels(client, app_settings):
    """Clearing all humpback labels sets has_humpback_labels to False."""
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
                output_tsv_path=str(tsv_path),
                has_humpback_labels=True,
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
    assert resp2.json()["has_humpback_labels"] is False

    await engine.dispose()


async def test_partial_save_preserves_humpback_flag_from_other_rows(
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
    assert resp2.json()["has_humpback_labels"] is True

    await engine.dispose()
