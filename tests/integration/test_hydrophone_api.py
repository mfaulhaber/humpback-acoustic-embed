"""Integration tests for hydrophone detection API endpoints."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
import uuid

import numpy as np


async def test_list_hydrophones(client):
    """GET /classifier/hydrophones returns configured archive sources."""
    resp = await client.get("/classifier/hydrophones")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 7
    ids = {h["id"] for h in data}
    assert "rpi_orcasound_lab" in ids
    assert "sanctsound_ci" in ids
    assert "sanctsound_oc" in ids
    assert "noaa_glacier_bay" in ids
    assert "sanctsound_fk01" not in ids
    assert all("name" in h and "location" in h for h in data)


async def test_list_hydrophone_detection_jobs_empty(client):
    """GET /classifier/hydrophone-detection-jobs returns empty list initially."""
    resp = await client.get("/classifier/hydrophone-detection-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_hydrophone_detection_job_bad_model(client):
    """POST with nonexistent model returns 400."""
    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": "nonexistent",
            "hydrophone_id": "rpi_orcasound_lab",
            "start_timestamp": 1700000000,
            "end_timestamp": 1700003600,
        },
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


async def test_create_hydrophone_detection_job_bad_hydrophone(client, app_settings):
    """POST with unknown hydrophone_id returns 400."""
    # First we need a valid model — insert one directly
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
                name="test-model",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": model_id,
            "hydrophone_id": "unknown_hydrophone",
            "start_timestamp": 1700000000,
            "end_timestamp": 1700003600,
        },
    )
    assert resp.status_code == 400
    assert "unknown hydrophone" in resp.json()["detail"].lower()

    await engine.dispose()


async def test_create_hydrophone_detection_job_invalid_time_range(client, app_settings):
    """POST with end <= start returns 422 (Pydantic validation)."""
    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": "some-model",
            "hydrophone_id": "rpi_orcasound_lab",
            "start_timestamp": 1700003600,
            "end_timestamp": 1700000000,
        },
    )
    assert resp.status_code == 422


async def test_create_hydrophone_detection_job_too_long_range(client):
    """POST with > 7 day range returns 422."""
    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": "some-model",
            "hydrophone_id": "rpi_orcasound_lab",
            "start_timestamp": 1700000000,
            "end_timestamp": 1700000000 + 8 * 86400,  # 8 days
        },
    )
    assert resp.status_code == 422


async def test_create_hydrophone_detection_job_rejects_detection_mode(
    client, app_settings
):
    """POST rejects the retired detection_mode field."""
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
                name="hydro-test-model-legacy",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": model_id,
            "hydrophone_id": "rpi_orcasound_lab",
            "start_timestamp": 1700000000,
            "end_timestamp": 1700003600,
            "detection_mode": "merged",
        },
    )
    assert resp.status_code == 422
    assert "detection_mode" in resp.text

    await engine.dispose()


async def test_create_hydrophone_detection_job_success(client, app_settings):
    """POST with valid model + hydrophone creates a queued job."""
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
                name="hydro-test-model",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": model_id,
            "hydrophone_id": "rpi_orcasound_lab",
            "start_timestamp": 1700000000,
            "end_timestamp": 1700003600,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["hydrophone_id"] == "rpi_orcasound_lab"
    assert data["hydrophone_name"] == "Orcasound Lab"
    assert data["start_timestamp"] == 1700000000
    assert data["end_timestamp"] == 1700003600
    assert data["detection_mode"] == "windowed"

    # Should appear in the list
    list_resp = await client.get("/classifier/hydrophone-detection-jobs")
    assert list_resp.status_code == 200
    jobs = list_resp.json()
    assert any(
        j["id"] == data["id"] and j["detection_mode"] == "windowed" for j in jobs
    )

    # Should NOT appear in the local detection jobs list
    local_resp = await client.get("/classifier/detection-jobs")
    local_jobs = local_resp.json()
    assert not any(j["id"] == data["id"] for j in local_jobs)

    await engine.dispose()


async def test_create_hydrophone_detection_job_noaa_success(client, app_settings):
    """POST with NOAA source creates a queued job."""
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
                name="hydro-test-model-noaa",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": model_id,
            "hydrophone_id": "noaa_glacier_bay",
            "start_timestamp": 1437782400,
            "end_timestamp": 1437786000,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["hydrophone_id"] == "noaa_glacier_bay"
    assert data["hydrophone_name"] == "NOAA Glacier Bay (Bartlett Cove)"

    await engine.dispose()


async def test_create_hydrophone_detection_job_sanctsound_success(client, app_settings):
    """POST with SanctSound NOAA source creates a queued job."""
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
                name="hydro-test-model-sanctsound",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": model_id,
            "hydrophone_id": "sanctsound_ci",
            "start_timestamp": 1541023200,
            "end_timestamp": 1541026800,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["hydrophone_id"] == "sanctsound_ci"
    assert data["hydrophone_name"] == "NOAA SanctSound (Channel Islands)"

    await engine.dispose()


async def test_create_hydrophone_detection_job_noaa_rejects_local_cache_path(
    client, app_settings, tmp_path
):
    """NOAA sources should reject local_cache_path inputs."""
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
                name="hydro-test-model-noaa-local",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": model_id,
            "hydrophone_id": "noaa_glacier_bay",
            "start_timestamp": 1437782400,
            "end_timestamp": 1437786000,
            "local_cache_path": str(tmp_path),
        },
    )
    assert resp.status_code == 400
    assert "local_cache_path" in resp.json()["detail"]

    await engine.dispose()


async def test_create_hydrophone_detection_job_hop_too_large(client, app_settings):
    """POST rejects hop_seconds larger than classifier window size."""
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
                name="hydro-test-model-hop",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.commit()

    resp = await client.post(
        "/classifier/hydrophone-detection-jobs",
        json={
            "classifier_model_id": model_id,
            "hydrophone_id": "rpi_orcasound_lab",
            "start_timestamp": 1700000000,
            "end_timestamp": 1700003600,
            "hop_seconds": 10.0,
        },
    )
    assert resp.status_code == 400
    assert "must be <=" in resp.json()["detail"]

    await engine.dispose()


async def test_cancel_hydrophone_job_not_found(client):
    """POST cancel for nonexistent job returns 404."""
    resp = await client.post("/classifier/hydrophone-detection-jobs/nonexistent/cancel")
    assert resp.status_code == 404


async def test_pause_resume_cancel_lifecycle(client, app_settings):
    """Pause/resume/cancel lifecycle works correctly."""
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
                status="running",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_north_sjc",
                hydrophone_name="North San Juan Channel",
                start_timestamp=1751644800.0,
                end_timestamp=1751648400.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    # Pause from running
    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"

    # Cannot pause again
    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/pause")
    assert resp.status_code == 400

    # Resume from paused
    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"

    # Cannot resume from running
    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/resume")
    assert resp.status_code == 400

    # Pause again then cancel from paused
    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/pause")
    assert resp.status_code == 200

    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "canceled"

    # Cannot pause/resume a canceled job
    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/pause")
    assert resp.status_code == 400
    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/resume")
    assert resp.status_code == 400

    await engine.dispose()


async def test_paused_job_content_endpoint_returns_rows(client, app_settings):
    """Paused hydrophone jobs with TSV output should support content endpoint."""
    import csv

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
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "20250706T002900Z.wav",
                "start_sec": "20.0",
                "end_sec": "25.0",
                "avg_confidence": "0.93",
                "peak_confidence": "0.94",
                "n_windows": "1",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="paused",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1751760000.0,
                end_timestamp=1751846400.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    # Legacy TSV row (filename=20250706T002900Z.wav, start_sec=20, end_sec=25)
    # is migrated to UTC: recording_ts(20250706T002900Z) + offsets
    assert rows[0]["start_utc"] == 1751761760.0
    assert rows[0]["end_utc"] == 1751761765.0

    await engine.dispose()


async def test_pause_resume_not_found(client):
    """Pause/resume for nonexistent job returns 404."""
    resp = await client.post("/classifier/hydrophone-detection-jobs/nonexistent/pause")
    assert resp.status_code == 404
    resp = await client.post("/classifier/hydrophone-detection-jobs/nonexistent/resume")
    assert resp.status_code == 404


async def test_canceled_job_content_and_download(client, app_settings):
    """Canceled jobs with output TSV support content and download endpoints."""
    import csv

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
        "detection_filename",
        "extract_filename",
        "hydrophone_name",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "20250704T160000Z.wav",
                "start_sec": "10.0",
                "end_sec": "16.0",
                "avg_confidence": "0.82",
                "peak_confidence": "0.86",
                "n_windows": "2",
                "detection_filename": "20250704T160010Z_20250704T160016Z.flac",
                "extract_filename": "20250704T160010Z_20250704T160020Z.flac",
                "hydrophone_name": "rpi_north_sjc",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="canceled",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_north_sjc",
                hydrophone_name="North San Juan Channel",
                start_timestamp=1751644800.0,
                end_timestamp=1751648400.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    # Content endpoint works for canceled jobs
    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["hydrophone_name"] == "rpi_north_sjc"
    # Legacy TSV migrated: filename=20250704T160000Z.wav + start_sec=10 -> start_utc
    assert rows[0]["start_utc"] == 1751644810.0
    assert rows[0]["end_utc"] == 1751644816.0

    # Download endpoint works for canceled jobs
    resp = await client.get(f"/classifier/detection-jobs/{job_id}/download")
    assert resp.status_code == 200

    await engine.dispose()


async def test_detection_job_out_has_hydrophone_fields(client, app_settings):
    """DetectionJobOut includes nullable hydrophone fields for local jobs."""
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
                classifier_model_id="fake-model",
                audio_folder="/tmp/fake",
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    # Hydrophone fields should be null for local jobs
    assert data["hydrophone_id"] is None
    assert data["hydrophone_name"] is None
    assert data["start_timestamp"] is None
    assert data["segments_processed"] is None
    assert data["alerts"] is None

    await engine.dispose()


async def test_hydrophone_content_returns_utc_identity(client, app_settings):
    """Hydrophone detection content returns start_utc/end_utc from migrated legacy TSV."""
    import csv

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
        "detection_filename",
        "extract_filename",
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
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1751439600.0,
                end_timestamp=1751461200.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    # detection_filename 20250702T080155Z_20250702T080203Z.flac is primary source
    assert rows[0]["start_utc"] == 1751443315.0
    assert rows[0]["end_utc"] == 1751443323.0
    assert rows[0]["avg_confidence"] == 0.951

    await engine.dispose()


async def test_hydrophone_content_migrates_legacy_rows_to_utc(client, app_settings):
    """Legacy rows without detection_filename are migrated using filename + offsets."""
    import csv

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
        "extract_filename",
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
                "extract_filename": "20250702T080155Z_20250702T080205Z.wav",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1751439600.0,
                end_timestamp=1751461200.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    # Legacy migration: base(20250702T080118Z) + 37/45 -> UTC epochs
    assert rows[0]["start_utc"] == 1751443315.0
    assert rows[0]["end_utc"] == 1751443323.0
    # raw_start_sec/raw_end_sec missing from TSV -> defaults to 0.0 -> base epoch
    base_epoch = datetime(2025, 7, 2, 8, 1, 18, tzinfo=timezone.utc).timestamp()
    assert rows[0]["raw_start_utc"] == base_epoch
    assert rows[0]["raw_end_utc"] == base_epoch
    assert rows[0]["merged_event_count"] == 1

    await engine.dispose()


async def test_hydrophone_download_includes_derived_detection_filename(
    client, app_settings
):
    """Download endpoint should derive detection_filename from UTC bounds in TSV export."""
    import csv
    import io

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
        "extract_filename",
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
                "extract_filename": "20250702T080155Z_20250702T080205Z.wav",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1751439600.0,
                end_timestamp=1751461200.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/download")
    assert resp.status_code == 200

    reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
    rows = list(reader)
    assert len(rows) == 1
    # TSV export: start_utc/end_utc columns + derived detection_filename
    assert reader.fieldnames is not None
    assert "start_utc" in reader.fieldnames
    assert "end_utc" in reader.fieldnames
    assert "detection_filename" in reader.fieldnames
    # Migrated from filename+offsets -> detection_filename derived from UTC bounds
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080203Z.flac"

    await engine.dispose()


async def test_hydrophone_download_returns_streaming_response(client, app_settings):
    """Download route should stream normalized TSV content."""
    import csv
    import io

    from fastapi.responses import StreamingResponse
    from sqlalchemy import insert

    from humpback.api.routers.classifier import download_detections
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
        "extract_filename",
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
                "extract_filename": "20250702T080155Z_20250702T080205Z.wav",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1751439600.0,
                end_timestamp=1751461200.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    async with sf() as session:
        response = await download_detections(job_id, session, app_settings)

    assert isinstance(response, StreamingResponse)

    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            chunks.append(chunk.encode())
        else:
            chunks.append(bytes(chunk))
    text = b"".join(chunks).decode()
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = list(reader)

    assert len(rows) == 1
    # TSV export derives detection_filename from UTC bounds
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080203Z.flac"
    assert reader.fieldnames is not None
    assert "start_utc" in reader.fieldnames
    assert "end_utc" in reader.fieldnames

    await engine.dispose()


async def test_hydrophone_audio_slice_late_row_uses_first_folder_anchor(
    client, app_settings, monkeypatch
):
    """Late synthetic rows should resolve via first-folder anchor mapping."""
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
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1000.0,
                end_timestamp=3000.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    def _list_hls_folders(_self, _hydrophone_id: str, start_ts: float, end_ts: float):
        return ["1500"] if start_ts <= 1500 <= end_ts else []

    def _list_segments(_self, _hydrophone_id: str, folder_ts: str):
        if folder_ts != "1500":
            return []
        return [f"rpi_orcasound_lab/hls/1500/seg{i:04d}.ts" for i in range(100)]

    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.list_hls_folders",
        _list_hls_folders,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.list_segments",
        _list_segments,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.fetch_segment",
        lambda _self, _key: b"fake-ts",
    )
    monkeypatch.setattr(
        "humpback.classifier.providers.orcasound_hls.decode_ts_bytes",
        lambda _ts_bytes, _sr: np.ones(32000 * 10, dtype=np.float32),
    )

    # start_utc=2350.0 is within job range [1000, 3000] and resolvable from folder 1500
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/audio-slice",
        params={"start_utc": 2350.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"

    await engine.dispose()


async def test_hydrophone_audio_slice_returns_404_for_uncovered_timestamp(
    client, app_settings, monkeypatch
):
    """Timestamps before any cached stream segments return 404."""
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
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1000.0,
                end_timestamp=3000.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    def _list_hls_folders(_self, _hydrophone_id: str, start_ts: float, end_ts: float):
        return ["1500"] if start_ts <= 1500 <= end_ts else []

    def _list_segments(_self, _hydrophone_id: str, folder_ts: str):
        if folder_ts != "1500":
            return []
        return [f"rpi_orcasound_lab/hls/1500/seg{i:04d}.ts" for i in range(100)]

    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.list_hls_folders",
        _list_hls_folders,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.list_segments",
        _list_segments,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.fetch_segment",
        lambda _self, _key: b"fake-ts",
    )
    monkeypatch.setattr(
        "humpback.classifier.providers.orcasound_hls.decode_ts_bytes",
        lambda _ts_bytes, _sr: np.ones(32000 * 10, dtype=np.float32),
    )

    # start_utc=1200.0 is before the only folder (1500) — no audio available
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/audio-slice",
        params={"start_utc": 1200.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 404

    await engine.dispose()


async def test_hydrophone_audio_slice_rejects_rows_beyond_job_end(
    client, app_settings, monkeypatch
):
    """Rows timestamped beyond end_timestamp should not resolve audio."""
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
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1500.0,
                end_timestamp=1525.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    def _list_hls_folders(_self, _hydrophone_id: str, start_ts: float, end_ts: float):
        return ["1500"] if start_ts <= 1500 <= end_ts else []

    def _list_segments(_self, _hydrophone_id: str, folder_ts: str):
        if folder_ts != "1500":
            return []
        return [f"rpi_orcasound_lab/hls/1500/live{i}.ts" for i in range(4)]

    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.list_hls_folders",
        _list_hls_folders,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.list_segments",
        _list_segments,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient.fetch_segment",
        lambda _self, _key: b"fake-ts",
    )
    monkeypatch.setattr(
        "humpback.classifier.providers.orcasound_hls.decode_ts_bytes",
        lambda _ts_bytes, _sr: np.ones(32000 * 10, dtype=np.float32),
    )

    # start_utc=1530 is beyond end_timestamp=1525 — should be rejected
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/audio-slice",
        params={"start_utc": 1530.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 404

    await engine.dispose()


async def test_hydrophone_audio_slice_supports_noaa_provider_without_cache(
    client, app_settings, monkeypatch
):
    """NOAA playback should use the provider builder without requiring cache config."""
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
                classifier_model_id="fake-model",
                hydrophone_id="noaa_glacier_bay",
                hydrophone_name="NOAA Glacier Bay (Bartlett Cove)",
                start_timestamp=1437782400.0,
                end_timestamp=1437786000.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    class DummyProvider:
        source_id = "noaa_glacier_bay"

    capture: dict[str, object] = {}

    def _fake_build_archive_playback_provider(
        source_id: str, *, cache_path: str | None, noaa_cache_path: str | None = None
    ):
        capture["source_id"] = source_id
        capture["cache_path"] = cache_path
        capture["noaa_cache_path"] = noaa_cache_path
        return DummyProvider()

    monkeypatch.setattr(
        "humpback.api.routers.classifier.build_archive_playback_provider",
        _fake_build_archive_playback_provider,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.resolve_audio_slice",
        lambda *args, **kwargs: np.ones(32000 * 5, dtype=np.float32),
    )

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/audio-slice",
        params={
            "start_utc": 1437782409.0,
            "duration_sec": 5.0,
        },
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"
    assert capture["source_id"] == "noaa_glacier_bay"
    assert capture["cache_path"] == app_settings.s3_cache_path
    assert capture["noaa_cache_path"] == app_settings.noaa_cache_path

    await engine.dispose()


def test_hydrophone_detection_respects_end_timestamp_with_local_hls_cache(
    tmp_path, monkeypatch
):
    """Hydrophone detector should not emit detections beyond job end time."""
    from humpback.classifier.hydrophone_detector import run_hydrophone_detection
    from humpback.classifier.providers import LocalHLSCacheProvider
    from humpback.config import ORCASOUND_S3_BUCKET

    hydrophone_id = "rpi_orcasound_lab"
    folder_ts = "1000"
    hls_dir = Path(tmp_path) / ORCASOUND_S3_BUCKET / hydrophone_id / "hls" / folder_ts
    hls_dir.mkdir(parents=True)

    # Four 10s segments exist, but requested range below should end at 25s.
    for i in range(4):
        (hls_dir / f"live{i}.ts").write_bytes(b"fake")
    (hls_dir / "live.m3u8").write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-VERSION:3",
                "#EXTINF:10.0,",
                "live0.ts",
                "#EXTINF:10.0,",
                "live1.ts",
                "#EXTINF:10.0,",
                "live2.ts",
                "#EXTINF:10.0,",
                "live3.ts",
            ]
        )
    )

    monkeypatch.setattr(
        "humpback.classifier.providers.orcasound_hls.decode_ts_bytes",
        lambda _ts_bytes, _sr: np.ones(32000 * 10, dtype=np.float32),
    )

    class FakeModel:
        @property
        def vector_dim(self) -> int:
            return 1

        def embed(self, batch):
            return np.zeros((len(batch), 1), dtype=np.float32)

    class FakePipeline:
        def predict_proba(self, emb):
            n = len(emb)
            return np.column_stack(
                [np.full(n, 0.1, dtype=np.float32), np.full(n, 0.9, dtype=np.float32)]
            )

    provider = LocalHLSCacheProvider(str(tmp_path), hydrophone_id, hydrophone_id)
    detections, _summary = run_hydrophone_detection(
        provider=provider,
        start_timestamp=1000.0,
        end_timestamp=1025.0,
        pipeline=cast(Any, FakePipeline()),
        model=FakeModel(),
        window_size_seconds=5.0,
        target_sample_rate=32000,
        confidence_threshold=0.5,
        input_format="waveform",
        hop_seconds=5.0,
        high_threshold=0.7,
        low_threshold=0.45,
    )

    assert detections
    for det in detections:
        # Detector now outputs absolute UTC epoch floats
        assert float(det["end_utc"]) <= 1025.01


# ---- Paused job label/download/extract gates ----


async def _create_paused_job_with_tsv(app_settings):
    """Helper: insert a paused hydrophone job with a small TSV file."""
    import csv

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
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "filename": "20250702T080118Z.wav",
                "start_sec": "0.0",
                "end_sec": "5.0",
                "avg_confidence": "0.85",
                "peak_confidence": "0.92",
                "n_windows": "3",
            }
        )

    async with sf() as session:
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="paused",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_orcasound_lab",
                hydrophone_name="Orcasound Lab",
                start_timestamp=1751439600.0,
                end_timestamp=1751461200.0,
                confidence_threshold=0.5,
                detection_mode="windowed",
            )
        )
        await session.commit()

    return job_id, engine


async def test_hydrophone_batch_add_populates_detection_range_metadata(
    client, app_settings
):
    """PATCH /labels should persist hydrophone timeline adds with UTC identity."""
    from humpback.classifier.detection_rows import read_detection_row_store

    job_id, engine = await _create_paused_job_with_tsv(app_settings)
    job_start = 1751439600.0

    add_start_utc = job_start + 123.0
    add_end_utc = job_start + 128.0

    resp = await client.patch(
        f"/classifier/detection-jobs/{job_id}/labels",
        json={
            "edits": [
                {
                    "action": "add",
                    "start_utc": add_start_utc,
                    "end_utc": add_end_utc,
                    "label": "humpback",
                }
            ]
        },
    )
    assert resp.status_code == 200

    content_resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert content_resp.status_code == 200
    content_rows = content_resp.json()
    added = next(row for row in content_rows if row["humpback"] == 1)
    assert added["start_utc"] == add_start_utc
    assert added["end_utc"] == add_end_utc

    row_store_path = (
        Path(app_settings.storage_root)
        / "detections"
        / job_id
        / "detection_rows.parquet"
    )
    _fieldnames, stored_rows = read_detection_row_store(row_store_path)
    stored = next(row for row in stored_rows if row.get("humpback") == "1")
    assert stored["start_utc"] == str(add_start_utc)
    assert stored["end_utc"] == str(add_end_utc)

    await engine.dispose()


async def test_hydrophone_content_returns_utc_row_store_rows(client, app_settings):
    """GET /content should return rows with start_utc/end_utc from new-schema row store."""
    from humpback.classifier.detection_rows import (
        ROW_STORE_FIELDNAMES,
        read_detection_row_store,
        write_detection_row_store,
    )

    job_id, engine = await _create_paused_job_with_tsv(app_settings)
    job_start = 1751439600.0
    row_store_path = (
        Path(app_settings.storage_root)
        / "detections"
        / job_id
        / "detection_rows.parquet"
    )

    utc_row = {field: "" for field in ROW_STORE_FIELDNAMES}
    utc_row["start_utc"] = str(job_start + 180.0)
    utc_row["end_utc"] = str(job_start + 185.0)
    utc_row["humpback"] = "1"
    utc_row["positive_selection_origin"] = "clip_bounds_fallback"
    utc_row["positive_selection_score_source"] = "clip_bounds_fallback"
    utc_row["positive_selection_decision"] = "positive"
    utc_row["positive_selection_start_utc"] = str(job_start + 180.0)
    utc_row["positive_selection_end_utc"] = str(job_start + 185.0)
    write_detection_row_store(row_store_path, [utc_row])

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["start_utc"] == job_start + 180.0
    assert rows[0]["end_utc"] == job_start + 185.0
    assert rows[0]["positive_selection_start_utc"] == job_start + 180.0
    assert rows[0]["positive_selection_end_utc"] == job_start + 185.0

    _fieldnames, stored_rows = read_detection_row_store(row_store_path)
    assert stored_rows[0]["start_utc"] == str(job_start + 180.0)
    assert stored_rows[0]["end_utc"] == str(job_start + 185.0)

    await engine.dispose()


async def test_hydrophone_content_returns_positive_selection_utc_fields(
    client, app_settings
):
    """GET /content should return positive_selection_start_utc/end_utc from row store."""
    from humpback.classifier.detection_rows import (
        ROW_STORE_FIELDNAMES,
        read_detection_row_store,
        write_detection_row_store,
    )

    job_id, engine = await _create_paused_job_with_tsv(app_settings)
    job_start = 1751439600.0
    row_store_path = (
        Path(app_settings.storage_root)
        / "detections"
        / job_id
        / "detection_rows.parquet"
    )

    utc_row = {field: "" for field in ROW_STORE_FIELDNAMES}
    utc_row["start_utc"] = str(job_start + 18323.0)
    utc_row["end_utc"] = str(job_start + 18328.0)
    utc_row["humpback"] = "1"
    utc_row["positive_selection_origin"] = "clip_bounds_fallback"
    utc_row["positive_selection_score_source"] = "clip_bounds_fallback"
    utc_row["positive_selection_decision"] = "positive"
    utc_row["positive_selection_start_utc"] = str(job_start + 18323.0)
    utc_row["positive_selection_end_utc"] = str(job_start + 18328.0)
    write_detection_row_store(row_store_path, [utc_row])

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["start_utc"] == job_start + 18323.0
    assert rows[0]["end_utc"] == job_start + 18328.0
    assert rows[0]["positive_selection_start_utc"] == job_start + 18323.0
    assert rows[0]["positive_selection_end_utc"] == job_start + 18328.0

    _fieldnames, stored_rows = read_detection_row_store(row_store_path)
    assert stored_rows[0]["positive_selection_start_utc"] == str(job_start + 18323.0)
    assert stored_rows[0]["positive_selection_end_utc"] == str(job_start + 18328.0)

    await engine.dispose()


async def test_save_labels_paused_job(client, app_settings):
    """PUT /labels on a paused job with TSV should succeed."""
    job_id, engine = await _create_paused_job_with_tsv(app_settings)

    # Legacy TSV row: filename=20250702T080118Z.wav, start_sec=0, end_sec=5
    # After migration: start_utc = recording_ts(20250702T080118Z) + 0.0
    start_utc = datetime(2025, 7, 2, 8, 1, 18, tzinfo=timezone.utc).timestamp()
    end_utc = start_utc + 5.0

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": start_utc,
                "end_utc": end_utc,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 200

    await engine.dispose()


async def test_save_labels_utc_identity_round_trip(client, app_settings):
    """Label saves accept the UTC identity returned by GET /content (round-trip)."""
    job_id, engine = await _create_paused_job_with_tsv(app_settings)

    content_resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert content_resp.status_code == 200
    rows = content_resp.json()
    assert len(rows) == 1

    row = rows[0]
    # Legacy TSV: filename=20250702T080118Z.wav + start_sec=0 -> start_utc
    expected_utc = datetime(2025, 7, 2, 8, 1, 18, tzinfo=timezone.utc).timestamp()
    assert row["start_utc"] == expected_utc
    assert row["end_utc"] == expected_utc + 5.0
    assert row["humpback"] is None

    save_resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": row["start_utc"],
                "end_utc": row["end_utc"],
                "humpback": 1,
            }
        ],
    )
    assert save_resp.status_code == 200

    verify_resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert verify_resp.status_code == 200
    updated = verify_resp.json()[0]
    assert updated["start_utc"] == expected_utc
    assert updated["end_utc"] == expected_utc + 5.0
    assert updated["humpback"] == 1

    job_resp = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["has_positive_labels"] is True

    await engine.dispose()


async def test_download_paused_job(client, app_settings):
    """GET /download on a paused job with TSV should return 200."""
    job_id, engine = await _create_paused_job_with_tsv(app_settings)

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/download")
    assert resp.status_code == 200

    await engine.dispose()


async def test_extract_paused_job(client, app_settings):
    """POST /extract on a paused job should be accepted."""
    job_id, engine = await _create_paused_job_with_tsv(app_settings)

    # First save a label so extraction has something to do
    # Legacy TSV: filename=20250702T080118Z.wav, start_sec=0.0 -> start_utc
    start_utc = datetime(2025, 7, 2, 8, 1, 18, tzinfo=timezone.utc).timestamp()
    await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": start_utc,
                "end_utc": start_utc + 5.0,
                "humpback": 1,
            }
        ],
    )

    resp = await client.post(
        "/classifier/detection-jobs/extract",
        json={"job_ids": [job_id]},
    )
    assert resp.status_code == 200

    await engine.dispose()


async def test_cancel_queued_hydrophone_job(client, app_settings):
    """POST cancel on a queued hydrophone job should succeed."""
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
                status="queued",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_north_sjc",
                hydrophone_name="North San Juan Channel",
                start_timestamp=1751644800.0,
                end_timestamp=1751648400.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.post(f"/classifier/hydrophone-detection-jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "canceled"

    # Verify in DB
    async with sf() as session:
        result = await session.execute(
            select(DetectionJob).where(DetectionJob.id == job_id)
        )
        job = result.scalar_one()
        assert job.status == "canceled"

    await engine.dispose()


async def test_save_labels_running_job_rejected(client, app_settings):
    """PUT /labels on a running job (no stable TSV) should return 400."""
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
                status="running",
                classifier_model_id="fake-model",
                hydrophone_id="rpi_north_sjc",
                hydrophone_name="North San Juan Channel",
                start_timestamp=1751644800.0,
                end_timestamp=1751648400.0,
                confidence_threshold=0.5,
            )
        )
        await session.commit()

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "start_utc": 1751644800.0,
                "end_utc": 1751644805.0,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 400

    await engine.dispose()
