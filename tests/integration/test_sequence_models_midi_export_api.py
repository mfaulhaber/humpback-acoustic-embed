"""Integration tests for Piano Roll MIDI export API endpoints."""

from __future__ import annotations

import io
import wave
from datetime import datetime, timezone

import mido
import soundfile as sf

from humpback.database import create_engine, create_session_factory
from humpback.models.piano_roll_midi_export import PianoRollMidiExport
from humpback.models.processing import JobStatus
from humpback.storage import (
    event_encoder_audio_export_path,
    event_encoder_midi_export_path,
)

from .test_sequence_models_api import (
    _seed_event_encoder_timeline_job,
    _seed_notes_job,
    _write_notes_sidecar,
)


# The event encoder timeline fixture has start=2000.0, end=2600.0.
_WIN_START = 2100.0
_WIN_END = 2160.0
_DEFAULT_BODY = {"window_start_utc": _WIN_START, "window_end_utc": _WIN_END}


async def _set_midi_export_row(
    app_settings,
    *,
    event_encoder_job_id: str,
    status: str,
    midi_path: str | None = None,
    extractor_version: str = "v2",
    n_bytes: int | None = None,
    audio_path: str = "",
    audio_size_bytes: int = 0,
    audio_sample_rate: int = 0,
    audio_duration_s: float = 0.0,
    window_start_utc: float = _WIN_START,
    window_end_utc: float = _WIN_END,
) -> str:
    engine = create_engine(app_settings.database_url)
    sf_factory = create_session_factory(engine)
    now = datetime.now(timezone.utc)
    async with sf_factory() as session:
        row = PianoRollMidiExport(
            event_encoder_job_id=event_encoder_job_id,
            extractor_version=extractor_version,
            status=status,
            midi_path=midi_path,
            n_notes=2 if status == JobStatus.complete.value else None,
            n_bytes=n_bytes,
            compute_seconds=0.1 if status == JobStatus.complete.value else None,
            finished_at=now if status == JobStatus.complete.value else None,
            params_json="{}",
            window_start_utc=window_start_utc,
            window_end_utc=window_end_utc,
            audio_path=audio_path,
            audio_size_bytes=audio_size_bytes,
            audio_sample_rate=audio_sample_rate,
            audio_duration_s=audio_duration_s,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row.id


# ---------- status endpoint ----------


async def test_get_midi_export_status_absent(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _write_notes_sidecar(app_settings, job_id, rows=[])
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/midi-export-status"
    )
    assert response.status_code == 200
    assert response.json() == {"status": "absent"}


async def test_get_midi_export_status_returns_existing(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.running.value,
    )

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/midi-export-status"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["event_encoder_job_id"] == job_id
    assert body["window_start_utc"] == _WIN_START
    assert body["window_end_utc"] == _WIN_END


async def test_get_midi_export_status_missing_encoder_returns_404(client) -> None:
    response = await client.get(
        "/sequence-models/event-encoders/missing/midi-export-status"
    )
    assert response.status_code == 404


# ---------- POST (enqueue) endpoint ----------


async def test_post_midi_export_creates_201(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _write_notes_sidecar(app_settings, job_id, rows=[])
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json=_DEFAULT_BODY,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["event_encoder_job_id"] == job_id
    assert body["status"] == "queued"
    assert body["extractor_version"] == "v2"
    assert body["window_start_utc"] == _WIN_START
    assert body["window_end_utc"] == _WIN_END


async def test_post_midi_export_rejects_zero_duration_window(
    client, app_settings
) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json={"window_start_utc": _WIN_START, "window_end_utc": _WIN_START},
    )
    assert response.status_code == 422


async def test_post_midi_export_rejects_over_cap_window(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json={
            "window_start_utc": _WIN_START,
            "window_end_utc": _WIN_START + 1801.0,
        },
    )
    assert response.status_code == 422


async def test_post_midi_export_rejects_window_outside_range(
    client, app_settings
) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json={"window_start_utc": 5000.0, "window_end_utc": 5060.0},
    )
    assert response.status_code == 400


async def test_post_midi_export_conflicts_with_running(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.running.value,
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json=_DEFAULT_BODY,
    )
    assert response.status_code == 409


async def test_post_midi_export_resets_failed_row_200(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.failed.value,
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json=_DEFAULT_BODY,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "queued"


async def test_post_midi_export_force_resets_complete_200(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        midi_path="exports/event_encoders/foo/notes_v2.mid",
        audio_path="exports/event_encoders/foo/audio_v2.flac",
        audio_size_bytes=10,
        audio_sample_rate=32_000,
        audio_duration_s=60.0,
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json={**_DEFAULT_BODY, "force": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"


async def test_post_midi_export_window_mismatch_resets_complete_200(
    client, app_settings
) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        midi_path="exports/event_encoders/foo/notes_v2.mid",
        audio_path="exports/event_encoders/foo/audio_v2.flac",
        audio_size_bytes=10,
        audio_sample_rate=32_000,
        audio_duration_s=60.0,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json={"window_start_utc": 2200.0, "window_end_utc": 2260.0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert body["window_start_utc"] == 2200.0
    assert body["window_end_utc"] == 2260.0


async def test_post_midi_export_window_match_returns_existing_200(
    client, app_settings
) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        midi_path="exports/event_encoders/foo/notes_v2.mid",
        audio_path="exports/event_encoders/foo/audio_v2.flac",
        audio_size_bytes=10,
        audio_sample_rate=32_000,
        audio_duration_s=60.0,
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json=_DEFAULT_BODY,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "complete"


async def test_post_midi_export_without_notes_returns_422(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json=_DEFAULT_BODY,
    )
    assert response.status_code == 422


async def test_post_midi_export_missing_encoder_returns_404(client) -> None:
    response = await client.post(
        "/sequence-models/event-encoders/missing/midi-exports",
        json=_DEFAULT_BODY,
    )
    assert response.status_code == 404


# ---------- GET MIDI download endpoint ----------


async def test_download_midi_export_returns_file(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    midi_path = event_encoder_midi_export_path(app_settings.storage_root, job_id, "v2")
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    mf = mido.MidiFile(type=1, ticks_per_beat=480)
    mf.tracks.append(mido.MidiTrack([mido.MetaMessage("end_of_track", time=0)]))
    mf.tracks.append(mido.MidiTrack([mido.MetaMessage("end_of_track", time=0)]))
    mf.save(str(midi_path))

    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        midi_path=str(midi_path),
        n_bytes=midi_path.stat().st_size,
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/midi-export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/midi")
    assert "attachment" in response.headers["content-disposition"]
    assert (
        f"event_encoder_{job_id}_notes_v2.mid"
        in response.headers["content-disposition"]
    )
    parsed = mido.MidiFile(file=io.BytesIO(response.content))
    assert parsed.type == 1


async def test_download_midi_export_404_when_no_complete_row(
    client, app_settings
) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.queued.value,
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/midi-export")
    assert response.status_code == 404


async def test_download_midi_export_404_when_file_missing(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        midi_path="exports/event_encoders/missing/notes_v2.mid",
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/midi-export")
    assert response.status_code == 404


# ---------- GET audio (FLAC) download endpoint ----------


async def test_download_audio_export_returns_file(client, app_settings) -> None:
    import numpy as np

    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    audio_path = event_encoder_audio_export_path(
        app_settings.storage_root, job_id, "v2"
    )
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(32_000, 0.5, dtype=np.float32)
    sf.write(
        str(audio_path),
        samples,
        32_000,
        format="FLAC",
        subtype="PCM_16",
    )

    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        audio_path=str(audio_path),
        audio_size_bytes=audio_path.stat().st_size,
        audio_sample_rate=32_000,
        audio_duration_s=1.0,
    )

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/audio-export"
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/flac")
    assert "attachment" in response.headers["content-disposition"]
    assert f"event_encoder_{job_id}_v2_" in response.headers["content-disposition"]
    # Body is parseable as FLAC.
    decoded, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    assert sr == 32_000
    assert decoded.ndim == 1


async def test_download_audio_export_404_when_no_complete_row(
    client, app_settings
) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.queued.value,
    )

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/audio-export"
    )
    assert response.status_code == 404


async def test_download_audio_export_404_when_file_missing(
    client, app_settings
) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        audio_path="exports/event_encoders/missing/audio_v2.flac",
        audio_size_bytes=42,
        audio_sample_rate=32_000,
        audio_duration_s=1.0,
    )

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/audio-export"
    )
    assert response.status_code == 404


# Reference the wave import so static analyzers don't complain when it's unused.
assert wave is not None
