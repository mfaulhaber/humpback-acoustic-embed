"""Smoke tests for the PianoRollMidiExport SQLAlchemy model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from humpback.models.piano_roll_midi_export import PianoRollMidiExport
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob


async def _make_encoder_job(session) -> str:
    job = EventEncoderJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id="seg-1",
        event_source_mode="raw",
        continuous_embedding_job_id="cej-1",
        continuous_embedding_signature="cej-sig",
        tokenizer_version="crnn-event-encoder-v2",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature=f"tok-sig-{datetime.now(timezone.utc).timestamp()}",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job.id


@pytest.mark.asyncio
async def test_persist_and_read_back_defaults(session) -> None:
    encoder_id = await _make_encoder_job(session)
    row = PianoRollMidiExport(event_encoder_job_id=encoder_id)
    session.add(row)
    await session.commit()
    await session.refresh(row)

    assert row.id is not None
    assert row.event_encoder_job_id == encoder_id
    assert row.extractor_version == "v2"
    assert row.status == JobStatus.queued.value
    assert row.started_at is None
    assert row.finished_at is None
    assert row.error_message is None
    assert row.midi_path is None
    assert row.n_notes is None
    assert row.n_bytes is None
    assert row.compute_seconds is None
    assert row.params_json == "{}"


@pytest.mark.asyncio
async def test_unique_constraint_on_encoder_and_version(session) -> None:
    encoder_id = await _make_encoder_job(session)
    session.add(
        PianoRollMidiExport(event_encoder_job_id=encoder_id, extractor_version="v2")
    )
    await session.commit()

    session.add(
        PianoRollMidiExport(event_encoder_job_id=encoder_id, extractor_version="v2")
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
