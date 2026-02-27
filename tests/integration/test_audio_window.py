import struct
import wave


async def test_audio_window_returns_wav_segment(client, wav_bytes):
    """Window endpoint returns a valid WAV covering the requested time range."""
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert upload.status_code == 201
    audio_id = upload.json()["id"]

    resp = await client.get(
        f"/audio/{audio_id}/window",
        params={"start_seconds": 0.0, "duration_seconds": 1.0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"

    # Parse returned WAV to verify duration
    import io

    buf = io.BytesIO(resp.content)
    with wave.open(buf, "r") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sr
        assert 0.9 <= duration <= 1.1  # ~1 second


async def test_audio_window_offset(client, wav_bytes):
    """Window with start offset returns correct number of samples."""
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    resp = await client.get(
        f"/audio/{audio_id}/window",
        params={"start_seconds": 0.5, "duration_seconds": 0.5},
    )
    assert resp.status_code == 200

    import io

    buf = io.BytesIO(resp.content)
    with wave.open(buf, "r") as wf:
        duration = wf.getnframes() / wf.getframerate()
        assert 0.4 <= duration <= 0.6


async def test_audio_window_clamps_to_file_end(client, wav_bytes):
    """Requesting past file end returns whatever audio remains."""
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    # wav_bytes is 2 seconds; request starting at 1.5s for 5s
    resp = await client.get(
        f"/audio/{audio_id}/window",
        params={"start_seconds": 1.5, "duration_seconds": 5.0},
    )
    assert resp.status_code == 200

    import io

    buf = io.BytesIO(resp.content)
    with wave.open(buf, "r") as wf:
        duration = wf.getnframes() / wf.getframerate()
        assert duration <= 0.6  # at most ~0.5s remaining


async def test_audio_window_not_found(client):
    resp = await client.get(
        "/audio/nonexistent/window",
        params={"start_seconds": 0, "duration_seconds": 1},
    )
    assert resp.status_code == 404


async def test_audio_window_missing_params(client, wav_bytes):
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    # Missing duration_seconds
    resp = await client.get(
        f"/audio/{audio_id}/window",
        params={"start_seconds": 0},
    )
    assert resp.status_code == 422
