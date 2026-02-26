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
