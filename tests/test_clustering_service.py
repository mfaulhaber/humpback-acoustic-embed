"""Tests for clustering service validation logic."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from humpback.database import Base
from humpback.models.audio import AudioFile
from humpback.models.processing import EmbeddingSet
from humpback.services.clustering_service import create_clustering_job


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _make_embedding_set(
    session: AsyncSession,
    model_version: str = "perch_v1",
    vector_dim: int = 1280,
) -> EmbeddingSet:
    unique = uuid.uuid4().hex[:8]
    af = AudioFile(filename=f"test-{unique}.wav", checksum_sha256=unique)
    session.add(af)
    await session.flush()
    es = EmbeddingSet(
        audio_file_id=af.id,
        encoding_signature=f"sig-{af.id}-{model_version}-{unique}",
        model_version=model_version,
        window_size_seconds=5.0,
        target_sample_rate=32000,
        vector_dim=vector_dim,
        parquet_path="/tmp/test.parquet",
    )
    session.add(es)
    await session.flush()
    return es


async def test_clustering_rejects_mixed_model_versions(session: AsyncSession):
    es1 = await _make_embedding_set(session, model_version="perch_v1")
    es2 = await _make_embedding_set(session, model_version="perch_v2")

    with pytest.raises(
        ValueError, match="Cannot cluster embedding sets from different models"
    ):
        await create_clustering_job(session, [es1.id, es2.id])


async def test_clustering_accepts_same_model_version(session: AsyncSession):
    es1 = await _make_embedding_set(session, model_version="perch_v1")
    es2 = await _make_embedding_set(session, model_version="perch_v1")

    job = await create_clustering_job(session, [es1.id, es2.id])
    assert job.id is not None


async def test_clustering_rejects_mixed_vector_dims(session: AsyncSession):
    es1 = await _make_embedding_set(session, vector_dim=1280)
    es2 = await _make_embedding_set(session, vector_dim=512)

    with pytest.raises(ValueError, match="different vector dimensions"):
        await create_clustering_job(session, [es1.id, es2.id])
