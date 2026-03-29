async def test_create_processing_job(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]
    resp = await client.post(
        "/processing/jobs",
        json={"audio_file_id": audio_id},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["encoding_signature"]
    assert data["skipped"] is False


async def test_create_processing_job_audio_not_found(client):
    resp = await client.post(
        "/processing/jobs",
        json={"audio_file_id": "does-not-exist"},
    )
    assert resp.status_code == 404
    assert "audio file not found" in resp.json()["detail"].lower()


async def test_list_processing_jobs(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]
    await client.post("/processing/jobs", json={"audio_file_id": audio_id})
    resp = await client.get("/processing/jobs")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


async def test_get_processing_job(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]
    create = await client.post("/processing/jobs", json={"audio_file_id": audio_id})
    job_id = create.json()["id"]
    resp = await client.get(f"/processing/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job_id


async def test_cancel_processing_job(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]
    create = await client.post("/processing/jobs", json={"audio_file_id": audio_id})
    job_id = create.json()["id"]
    resp = await client.post(f"/processing/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "canceled"


async def test_list_embedding_sets_empty(client):
    resp = await client.get("/processing/embedding-sets")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_folder_embedding_set_invalid_path(client):
    resp = await client.post(
        "/processing/folder-embedding-set",
        json={"folder_path": "/nonexistent/path"},
    )
    assert resp.status_code == 400


async def test_folder_embedding_set_with_audio(client, tmp_path, wav_bytes):
    # Create a folder with audio
    audio_dir = tmp_path / "test_audio"
    audio_dir.mkdir()
    (audio_dir / "sample.wav").write_bytes(wav_bytes)

    resp = await client.post(
        "/processing/folder-embedding-set",
        json={"folder_path": str(audio_dir)},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["folder_path"] == str(audio_dir)
    assert data["total_files"] >= 1
    assert data["status"] in ("ready", "processing", "queued")
