async def test_upload_audio(client, wav_bytes):
    resp = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "test.wav"
    assert data["checksum_sha256"]
    assert data["id"]


async def test_upload_dedup(client, wav_bytes):
    resp1 = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    resp2 = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert resp1.json()["id"] == resp2.json()["id"]


async def test_list_audio(client, wav_bytes):
    await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    resp = await client.get("/audio/")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


async def test_get_audio(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]
    resp = await client.get(f"/audio/{audio_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == audio_id


async def test_get_audio_not_found(client):
    resp = await client.get("/audio/nonexistent")
    assert resp.status_code == 404


async def test_update_metadata(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]
    resp = await client.put(
        f"/audio/{audio_id}/metadata",
        json={"tag_data": {"species": "humpback"}, "visual_observations": {"breaching": True}},
    )
    assert resp.status_code == 200
    assert resp.json()["tag_data"] == {"species": "humpback"}


async def test_upload_with_folder_path(client, wav_bytes):
    resp = await client.post(
        "/audio/upload",
        files={"file": ("song.wav", wav_bytes, "audio/wav")},
        data={"folder_path": "field_recordings/hawaii/2024"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["folder_path"] == "field_recordings/hawaii/2024"


async def test_upload_folder_path_normalization(client, wav_bytes):
    resp = await client.post(
        "/audio/upload",
        files={"file": ("norm.wav", wav_bytes, "audio/wav")},
        data={"folder_path": "//a///b//c//"},
    )
    assert resp.status_code == 201
    assert resp.json()["folder_path"] == "a/b/c"


async def test_upload_default_empty_folder_path(client, wav_bytes):
    resp = await client.post(
        "/audio/upload",
        files={"file": ("default.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 201
    assert resp.json()["folder_path"] == ""


async def test_folder_path_in_list(client, wav_bytes):
    await client.post(
        "/audio/upload",
        files={"file": ("listed.wav", wav_bytes, "audio/wav")},
        data={"folder_path": "recordings/pacific"},
    )
    resp = await client.get("/audio/")
    assert resp.status_code == 200
    files = resp.json()
    match = [f for f in files if f["filename"] == "listed.wav"]
    assert len(match) == 1
    assert match[0]["folder_path"] == "recordings/pacific"
