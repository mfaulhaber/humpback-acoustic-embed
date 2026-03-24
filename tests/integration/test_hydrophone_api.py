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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["filename"] == "20250706T002900Z.wav"

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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    # Content endpoint works for canceled jobs
    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["hydrophone_name"] == "rpi_north_sjc"
    assert rows[0]["detection_filename"] == "20250704T160010Z_20250704T160016Z.flac"
    assert rows[0]["extract_filename"] == "20250704T160010Z_20250704T160020Z.flac"

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


async def test_hydrophone_content_includes_extract_filename(client, app_settings):
    """Hydrophone detection content surfaces extract_filename when present in TSV."""
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080203Z.flac"
    assert rows[0]["extract_filename"] == "20250702T080155Z_20250702T080205Z.flac"

    await engine.dispose()


async def test_hydrophone_content_derives_detection_filename_for_legacy_rows(
    client, app_settings
):
    """Legacy rows without detection_filename should reuse extract_filename when available."""
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/content")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080205Z.wav"
    assert rows[0]["extract_filename"] == "20250702T080155Z_20250702T080205Z.wav"
    assert rows[0]["raw_start_sec"] == 37.0
    assert rows[0]["raw_end_sec"] == 45.0
    assert rows[0]["merged_event_count"] == 1

    await engine.dispose()


async def test_hydrophone_download_normalizes_legacy_detection_filename(
    client, app_settings
):
    """Download endpoint should normalize legacy rows without rewriting source TSV."""
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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/download")
    assert resp.status_code == 200

    reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080205Z.wav"
    assert rows[0]["extract_filename"] == "20250702T080155Z_20250702T080205Z.wav"
    assert rows[0]["raw_start_sec"] == "37.000000"
    assert rows[0]["raw_end_sec"] == "45.000000"
    assert rows[0]["merged_event_count"] == "1"

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
                output_tsv_path=str(tsv_path),
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
    assert rows[0]["detection_filename"] == "20250702T080155Z_20250702T080205Z.wav"
    assert rows[0]["extract_filename"] == "20250702T080155Z_20250702T080205Z.wav"
    assert rows[0]["raw_start_sec"] == "37.000000"
    assert rows[0]["raw_end_sec"] == "45.000000"
    assert rows[0]["merged_event_count"] == "1"

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

    # 2350 is too far from job.start (legacy anchor), but valid from first folder (1500)
    filename = (
        datetime.fromtimestamp(2350, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".wav"
    )

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/audio-slice",
        params={"filename": filename, "start_sec": 0.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"

    await engine.dispose()


async def test_hydrophone_audio_slice_legacy_anchor_fallback_still_works(
    client, app_settings, monkeypatch
):
    """Older jobs still resolve rows anchored to job.start_timestamp."""
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

    # 1200 is before first folder (1500) so first-anchor offset is negative.
    # Legacy job.start anchor should resolve this row.
    filename = (
        datetime.fromtimestamp(1200, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".wav"
    )

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/audio-slice",
        params={"filename": filename, "start_sec": 0.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"

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

    # Synthetic row starts after end_timestamp=1525.
    filename = (
        datetime.fromtimestamp(1530, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".wav"
    )
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/audio-slice",
        params={"filename": filename, "start_sec": 0.0, "duration_sec": 5.0},
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
            "filename": "20150725T000009Z.wav",
            "start_sec": 0.0,
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
        chunk_ts = (
            datetime.strptime(det["filename"][:-4], "%Y%m%dT%H%M%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        assert chunk_ts + float(det["end_sec"]) <= 1025.01


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
                output_tsv_path=str(tsv_path),
            )
        )
        await session.commit()

    return job_id, engine


async def test_save_labels_paused_job(client, app_settings):
    """PUT /labels on a paused job with TSV should succeed."""
    job_id, engine = await _create_paused_job_with_tsv(app_settings)

    resp = await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "filename": "20250702T080118Z.wav",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 200

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
    await client.put(
        f"/classifier/detection-jobs/{job_id}/labels",
        json=[
            {
                "filename": "20250702T080118Z.wav",
                "start_sec": 0.0,
                "end_sec": 5.0,
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
                "filename": "test.wav",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "humpback": 1,
            }
        ],
    )
    assert resp.status_code == 400

    await engine.dispose()
