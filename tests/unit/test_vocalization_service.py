"""Unit tests for vocalization vocabulary service."""

import json

import pytest
from sqlalchemy import select

from humpback.database import Base, create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.processing import EmbeddingSet
from humpback.models.vocalization import VocalizationClassifierModel
from humpback.services.vocalization_service import (
    activate_model,
    create_type,
    delete_type,
    import_types_from_embedding_sets,
    list_types,
    update_type,
)


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.mark.asyncio
async def test_crud_lifecycle(session_factory):
    """Create, list, update, delete vocalization types."""
    async with session_factory() as session:
        vt = await create_type(session, "  Whup  ", description="A low-freq call")
        assert vt.name == "Whup"
        assert vt.description == "A low-freq call"

        types = await list_types(session)
        assert len(types) == 1
        assert types[0].name == "Whup"

        updated = await update_type(session, vt.id, name="MOAN")
        assert updated is not None
        assert updated.name == "Moan"

        deleted = await delete_type(session, vt.id)
        assert deleted is True

        types = await list_types(session)
        assert len(types) == 0


@pytest.mark.asyncio
async def test_update_nonexistent(session_factory):
    async with session_factory() as session:
        result = await update_type(session, "nonexistent", name="x")
        assert result is None


@pytest.mark.asyncio
async def test_delete_nonexistent(session_factory):
    async with session_factory() as session:
        result = await delete_type(session, "nonexistent")
        assert result is False


@pytest.mark.asyncio
async def test_delete_blocked_by_active_model(session_factory):
    """Cannot delete a type used in an active model's vocabulary."""
    async with session_factory() as session:
        vt = await create_type(session, "Whup")

        model = VocalizationClassifierModel(
            name="test-model",
            model_dir_path="/fake",
            vocabulary_snapshot=json.dumps(["Whup", "Moan"]),
            per_class_thresholds=json.dumps({"Whup": 0.5, "Moan": 0.5}),
            is_active=True,
        )
        session.add(model)
        await session.commit()

        with pytest.raises(ValueError, match="active model"):
            await delete_type(session, vt.id)


@pytest.mark.asyncio
async def test_delete_allowed_when_model_inactive(session_factory):
    """Can delete a type used by an inactive model."""
    async with session_factory() as session:
        vt = await create_type(session, "Whup")

        model = VocalizationClassifierModel(
            name="test-model",
            model_dir_path="/fake",
            vocabulary_snapshot=json.dumps(["Whup"]),
            per_class_thresholds=json.dumps({"Whup": 0.5}),
            is_active=False,
        )
        session.add(model)
        await session.commit()

        deleted = await delete_type(session, vt.id)
        assert deleted is True


@pytest.mark.asyncio
async def test_import_from_embedding_sets(session_factory):
    """Import vocalization types from embedding set folder structure."""
    async with session_factory() as session:
        # Create audio files with call-type folder paths
        af1 = AudioFile(
            filename="call1.wav",
            folder_path="accepted/whup",
            checksum_sha256="aaa",
        )
        af2 = AudioFile(
            filename="call2.wav",
            folder_path="accepted/moan",
            checksum_sha256="bbb",
        )
        af3 = AudioFile(
            filename="call3.wav",
            folder_path="accepted/Shriek",
            checksum_sha256="ccc",
        )
        session.add_all([af1, af2, af3])
        await session.flush()

        # Create embedding sets for each audio file
        es1 = EmbeddingSet(
            audio_file_id=af1.id,
            encoding_signature="sig1",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=128,
            parquet_path="/fake/1.parquet",
        )
        es2 = EmbeddingSet(
            audio_file_id=af2.id,
            encoding_signature="sig2",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=128,
            parquet_path="/fake/2.parquet",
        )
        es3 = EmbeddingSet(
            audio_file_id=af3.id,
            encoding_signature="sig3",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=128,
            parquet_path="/fake/3.parquet",
        )
        session.add_all([es1, es2, es3])
        await session.commit()

        added, skipped = await import_types_from_embedding_sets(
            session, [es1.id, es2.id, es3.id]
        )

        assert sorted(added) == ["Moan", "Shriek", "Whup"]
        assert skipped == []

        # Verify types exist in DB
        types = await list_types(session)
        assert len(types) == 3


@pytest.mark.asyncio
async def test_import_deduplication(session_factory):
    """Re-importing same embedding sets doesn't create duplicates."""
    async with session_factory() as session:
        af = AudioFile(
            filename="call.wav",
            folder_path="accepted/whup",
            checksum_sha256="aaa",
        )
        session.add(af)
        await session.flush()

        es = EmbeddingSet(
            audio_file_id=af.id,
            encoding_signature="sig1",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=128,
            parquet_path="/fake/1.parquet",
        )
        session.add(es)
        await session.commit()

        added1, skipped1 = await import_types_from_embedding_sets(session, [es.id])
        assert added1 == ["Whup"]
        assert skipped1 == []

        added2, skipped2 = await import_types_from_embedding_sets(session, [es.id])
        assert added2 == []
        assert skipped2 == ["Whup"]

        types = await list_types(session)
        assert len(types) == 1


@pytest.mark.asyncio
async def test_import_nonexistent_embedding_set(session_factory):
    """Import with nonexistent embedding set ID is skipped gracefully."""
    async with session_factory() as session:
        added, skipped = await import_types_from_embedding_sets(
            session, ["nonexistent-id"]
        )
        assert added == []
        assert skipped == []


@pytest.mark.asyncio
async def test_activate_model(session_factory):
    """Activating a model deactivates the previously active one."""
    async with session_factory() as session:
        m1 = VocalizationClassifierModel(
            name="model-1",
            model_dir_path="/fake/1",
            vocabulary_snapshot=json.dumps(["Whup"]),
            per_class_thresholds=json.dumps({"Whup": 0.5}),
            is_active=True,
        )
        m2 = VocalizationClassifierModel(
            name="model-2",
            model_dir_path="/fake/2",
            vocabulary_snapshot=json.dumps(["Whup"]),
            per_class_thresholds=json.dumps({"Whup": 0.5}),
            is_active=False,
        )
        session.add_all([m1, m2])
        await session.commit()

        activated = await activate_model(session, m2.id)
        assert activated is not None
        assert activated.is_active is True

        # m1 should now be inactive
        result = await session.execute(
            select(VocalizationClassifierModel).where(
                VocalizationClassifierModel.id == m1.id
            )
        )
        m1_refreshed = result.scalar_one()
        assert m1_refreshed.is_active is False


@pytest.mark.asyncio
async def test_activate_nonexistent_model(session_factory):
    async with session_factory() as session:
        result = await activate_model(session, "nonexistent")
        assert result is None
