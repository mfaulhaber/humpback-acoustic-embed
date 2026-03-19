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
        json={
            "tag_data": {"species": "humpback"},
            "visual_observations": {"breaching": True},
        },
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


async def test_download_audio_invalid_range_header(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("range.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    resp = await client.get(
        f"/audio/{audio_id}/download",
        headers={"Range": "bytes=abc-def"},
    )
    assert resp.status_code == 416


# ---------------------------------------------------------------------------
# GET /audio/{id}/spectrogram-png
# ---------------------------------------------------------------------------


async def test_spectrogram_png_returns_png(client, wav_bytes):
    """GET /audio/{id}/spectrogram-png returns PNG image bytes."""
    upload = await client.post(
        "/audio/upload",
        files={"file": ("spec.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    resp = await client.get(
        f"/audio/{audio_id}/spectrogram-png",
        params={"start_seconds": 0.0, "duration_seconds": 1.0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # PNG files start with the magic bytes \x89PNG
    assert resp.content[:4] == b"\x89PNG"
    assert len(resp.content) > 100  # non-trivial PNG


async def test_spectrogram_png_not_found(client):
    """GET /audio/{id}/spectrogram-png returns 404 for nonexistent audio."""
    resp = await client.get(
        "/audio/nonexistent/spectrogram-png",
        params={"start_seconds": 0.0, "duration_seconds": 1.0},
    )
    assert resp.status_code == 404


async def test_spectrogram_png_out_of_bounds(client, wav_bytes):
    """GET /audio/{id}/spectrogram-png returns 400 when range exceeds audio."""
    upload = await client.post(
        "/audio/upload",
        files={"file": ("oob.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    # WAV is 2 seconds long; request starting at 999 seconds
    resp = await client.get(
        f"/audio/{audio_id}/spectrogram-png",
        params={"start_seconds": 999.0, "duration_seconds": 1.0},
    )
    assert resp.status_code == 400
