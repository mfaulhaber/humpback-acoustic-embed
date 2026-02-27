import pytest
from sqlalchemy import select

from humpback.models.model_registry import TFLiteModelConfig
from humpback.models.processing import EmbeddingSet
from humpback.services import model_registry_service


@pytest.fixture
async def sample_model(session):
    """Create a sample model for testing."""
    return await model_registry_service.create_model(
        session,
        name="test_model",
        display_name="Test Model",
        path="models/test.tflite",
        vector_dim=1280,
        is_default=True,
    )


async def test_create_model(session):
    model = await model_registry_service.create_model(
        session,
        name="my_model",
        display_name="My Model",
        path="models/my.tflite",
        vector_dim=512,
    )
    assert model.name == "my_model"
    assert model.vector_dim == 512
    assert model.is_default is False


async def test_create_duplicate_name_rejected(session, sample_model):
    """Duplicate names should raise IntegrityError."""
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        await model_registry_service.create_model(
            session,
            name="test_model",
            display_name="Duplicate",
            path="models/dup.tflite",
        )


async def test_default_toggling(session):
    m1 = await model_registry_service.create_model(
        session,
        name="model_a",
        display_name="A",
        path="a.tflite",
        is_default=True,
    )
    assert m1.is_default is True

    m2 = await model_registry_service.create_model(
        session,
        name="model_b",
        display_name="B",
        path="b.tflite",
        is_default=True,
    )
    assert m2.is_default is True

    # Refresh m1 from DB
    await session.refresh(m1)
    assert m1.is_default is False


async def test_set_default_model(session):
    m1 = await model_registry_service.create_model(
        session, name="x", display_name="X", path="x.tflite", is_default=True,
    )
    m2 = await model_registry_service.create_model(
        session, name="y", display_name="Y", path="y.tflite",
    )
    result = await model_registry_service.set_default_model(session, m2.id)
    assert result.is_default is True
    await session.refresh(m1)
    assert m1.is_default is False


async def test_get_by_name(session, sample_model):
    found = await model_registry_service.get_model_by_name(session, "test_model")
    assert found is not None
    assert found.id == sample_model.id


async def test_get_by_name_not_found(session):
    found = await model_registry_service.get_model_by_name(session, "nonexistent")
    assert found is None


async def test_get_default_model(session, sample_model):
    default = await model_registry_service.get_default_model(session)
    assert default is not None
    assert default.name == "test_model"


async def test_list_models(session, sample_model):
    models = await model_registry_service.list_models(session)
    assert len(models) >= 1
    names = [m.name for m in models]
    assert "test_model" in names


async def test_update_model(session, sample_model):
    updated = await model_registry_service.update_model(
        session, sample_model.id, display_name="Updated Name", vector_dim=256,
    )
    assert updated.display_name == "Updated Name"
    assert updated.vector_dim == 256


async def test_delete_model(session, sample_model):
    await model_registry_service.delete_model(session, sample_model.id)
    found = await model_registry_service.get_model_by_id(session, sample_model.id)
    assert found is None


async def test_delete_model_rejected_when_embeddings_exist(session, sample_model):
    """Cannot delete a model when embedding sets reference its model_version."""
    from humpback.models.audio import AudioFile

    af = AudioFile(filename="test.wav", checksum_sha256="abc123")
    session.add(af)
    await session.flush()

    es = EmbeddingSet(
        audio_file_id=af.id,
        encoding_signature="sig1",
        model_version="test_model",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        vector_dim=1280,
        parquet_path="/tmp/test.parquet",
    )
    session.add(es)
    await session.flush()

    with pytest.raises(ValueError, match="embedding sets reference it"):
        await model_registry_service.delete_model(session, sample_model.id)


async def test_seed_default_model(session):
    await model_registry_service.seed_default_model(session)
    models = await model_registry_service.list_models(session)
    assert len(models) == 1
    assert models[0].name == "multispecies_whale_fp16"
    assert models[0].is_default is True

    # Calling again should not create a duplicate
    await model_registry_service.seed_default_model(session)
    models = await model_registry_service.list_models(session)
    assert len(models) == 1
