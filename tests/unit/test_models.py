import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from humpback.models import AudioFile, EmbeddingSet, ProcessingJob


async def test_create_audio_file(session):
    af = AudioFile(filename="test.wav", checksum_sha256="abc123")
    session.add(af)
    await session.commit()
    result = await session.execute(select(AudioFile))
    row = result.scalar_one()
    assert row.filename == "test.wav"
    assert row.id is not None


async def test_unique_checksum(session):
    af1 = AudioFile(filename="a.wav", checksum_sha256="same")
    af2 = AudioFile(filename="b.wav", checksum_sha256="same")
    session.add(af1)
    await session.commit()
    session.add(af2)
    try:
        await session.commit()
        assert False, "Should have raised IntegrityError"
    except IntegrityError:
        await session.rollback()


async def test_unique_encoding_signature(session):
    af = AudioFile(filename="a.wav", checksum_sha256="c1")
    session.add(af)
    await session.commit()
    es1 = EmbeddingSet(
        audio_file_id=af.id,
        encoding_signature="sig1",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        vector_dim=512,
        parquet_path="/tmp/test.parquet",
    )
    session.add(es1)
    await session.commit()
    es2 = EmbeddingSet(
        audio_file_id=af.id,
        encoding_signature="sig1",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        vector_dim=512,
        parquet_path="/tmp/test2.parquet",
    )
    session.add(es2)
    try:
        await session.commit()
        assert False, "Should have raised IntegrityError"
    except IntegrityError:
        await session.rollback()


async def test_processing_job_defaults(session):
    af = AudioFile(filename="a.wav", checksum_sha256="c2")
    session.add(af)
    await session.commit()
    job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig_test",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(job)
    await session.commit()
    assert job.status == "queued"
    assert job.error_message is None
