"""Integration tests for label processing API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_list_jobs_empty(client):
    resp = await client.get("/label-processing/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_create_job_missing_model(client, tmp_path):
    ann_dir = tmp_path / "ann"
    aud_dir = tmp_path / "audio"
    ann_dir.mkdir()
    aud_dir.mkdir()
    resp = await client.post(
        "/label-processing/jobs",
        json={
            "classifier_model_id": "nonexistent",
            "annotation_folder": str(ann_dir),
            "audio_folder": str(aud_dir),
            "output_root": str(tmp_path / "output"),
        },
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_job_missing_folders(client):
    resp = await client.post(
        "/label-processing/jobs",
        json={
            "classifier_model_id": "some-id",
            "annotation_folder": "/nonexistent/path/ann",
            "audio_folder": "/nonexistent/path/audio",
            "output_root": "/tmp/output",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_job_not_found(client):
    resp = await client.get("/label-processing/jobs/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_job_not_found(client):
    resp = await client.delete("/label-processing/jobs/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_preview_missing_folder(client):
    resp = await client.get(
        "/label-processing/preview",
        params={
            "annotation_folder": "/nonexistent/ann",
            "audio_folder": "/nonexistent/audio",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_preview_valid_folders(client, tmp_path):
    ann_dir = tmp_path / "ann"
    aud_dir = tmp_path / "audio"
    ann_dir.mkdir()
    aud_dir.mkdir()

    content = "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n"
    content += "1\t1.0\t2.0\tMoan\n"
    content += "2\t5.0\t6.5\tGrunt\n"
    content += "3\t10.0\t11.0\tMoan\n"
    (ann_dir / "rec1.Table.1.selections.txt").write_text(content)
    (aud_dir / "rec1.flac").write_bytes(b"fake")

    resp = await client.get(
        "/label-processing/preview",
        params={
            "annotation_folder": str(ann_dir),
            "audio_folder": str(aud_dir),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_annotations"] == 3
    assert len(data["paired_files"]) == 1
    assert data["paired_files"][0]["annotation_count"] == 3
    assert data["call_type_distribution"]["Moan"] == 2
    assert data["call_type_distribution"]["Grunt"] == 1
