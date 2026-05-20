"""Integration tests for Piano Roll MIDI export API endpoints."""

from __future__ import annotations

import io
from datetime import datetime, timezone

import mido

from humpback.database import create_engine, create_session_factory
from humpback.models.piano_roll_midi_export import PianoRollMidiExport
from humpback.models.processing import JobStatus
from humpback.storage import event_encoder_midi_export_path

from .test_sequence_models_api import (
    _seed_event_encoder_timeline_job,
    _seed_notes_job,
    _write_notes_sidecar,
)


async def _set_midi_export_row(
    app_settings,
    *,
    event_encoder_job_id: str,
    status: str,
    midi_path: str | None = None,
    extractor_version: str = "v1",
    n_bytes: int | None = None,
) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    now = datetime.now(timezone.utc)
    async with sf() as session:
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
        f"/sequence-models/event-encoders/{job_id}/midi-exports", json={}
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["event_encoder_job_id"] == job_id
    assert body["status"] == "queued"
    assert body["extractor_version"] == "v1"


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
        f"/sequence-models/event-encoders/{job_id}/midi-exports", json={}
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
        f"/sequence-models/event-encoders/{job_id}/midi-exports", json={}
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
        midi_path="exports/event_encoders/foo/notes_v1.mid",
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports",
        json={"force": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"


async def test_post_midi_export_without_notes_returns_422(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    # no notes job at all

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/midi-exports", json={}
    )
    assert response.status_code == 422


async def test_post_midi_export_missing_encoder_returns_404(client) -> None:
    response = await client.post(
        "/sequence-models/event-encoders/missing/midi-exports", json={}
    )
    assert response.status_code == 404


# ---------- GET (download) endpoint ----------


async def test_download_midi_export_returns_file(client, app_settings) -> None:
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(
        app_settings, job_id, status=JobStatus.complete.value, n_notes=0
    )
    # Write a valid MIDI file to the expected export path.
    midi_path = event_encoder_midi_export_path(app_settings.storage_root, job_id, "v1")
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
        f"event_encoder_{job_id}_notes_v1.mid"
        in response.headers["content-disposition"]
    )
    # Body is a parseable MIDI file.
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
    # Complete row but no file on disk.
    await _set_midi_export_row(
        app_settings,
        event_encoder_job_id=job_id,
        status=JobStatus.complete.value,
        midi_path="exports/event_encoders/missing/notes_v1.mid",
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/midi-export")
    assert response.status_code == 404
