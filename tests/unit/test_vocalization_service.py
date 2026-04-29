"""Unit tests for vocalization vocabulary service."""

import json

import pytest
from sqlalchemy import select

from humpback.database import Base, create_engine, create_session_factory
from humpback.models.vocalization import VocalizationClassifierModel
from humpback.services.vocalization_service import (
    activate_model,
    create_type,
    delete_type,
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
