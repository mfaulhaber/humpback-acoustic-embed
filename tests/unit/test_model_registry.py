import os
import tempfile

import pytest
from sqlalchemy import select

from humpback.models.model_registry import ModelConfig, TFLiteModelConfig
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


async def test_create_model_with_model_type(session):
    """Creating a model with model_type and input_format should persist them."""
    model = await model_registry_service.create_model(
        session,
        name="surfperch_tf2",
        display_name="SurfPerch TF2",
        path="models/surfperch-tensorflow2",
        vector_dim=1280,
        model_type="tf2_saved_model",
        input_format="waveform",
    )
    assert model.model_type == "tf2_saved_model"
    assert model.input_format == "waveform"

    # Fetch and verify
    fetched = await model_registry_service.get_model_by_name(session, "surfperch_tf2")
    assert fetched.model_type == "tf2_saved_model"
    assert fetched.input_format == "waveform"


async def test_default_model_type_values(session):
    """Models created without explicit model_type should default to tflite/spectrogram."""
    model = await model_registry_service.create_model(
        session,
        name="default_type_test",
        display_name="Default Type Test",
        path="models/test.tflite",
    )
    assert model.model_type == "tflite"
    assert model.input_format == "spectrogram"


def test_scan_detects_saved_model_dirs():
    """scan_model_files should detect TF2 SavedModel directories."""
    from humpback.config import Settings

    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir)

        # Create a .tflite file
        tflite_path = os.path.join(models_dir, "test_model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(b"\x00" * 100)

        # Create a SavedModel directory
        saved_model_dir = os.path.join(models_dir, "surfperch-tensorflow2")
        os.makedirs(saved_model_dir)
        with open(os.path.join(saved_model_dir, "saved_model.pb"), "wb") as f:
            f.write(b"\x00" * 200)

        settings = Settings(models_dir=models_dir)
        files = model_registry_service.scan_model_files(settings)

        assert len(files) == 2

        tflite_files = [f for f in files if f["model_type"] == "tflite"]
        tf2_files = [f for f in files if f["model_type"] == "tf2_saved_model"]

        assert len(tflite_files) == 1
        assert tflite_files[0]["filename"] == "test_model.tflite"
        assert tflite_files[0]["input_format"] == "spectrogram"

        assert len(tf2_files) == 1
        assert tf2_files[0]["filename"] == "surfperch-tensorflow2"
        assert tf2_files[0]["input_format"] == "waveform"
        assert tf2_files[0]["size_bytes"] == 200


def test_scan_ignores_dirs_without_saved_model_pb():
    """Directories without saved_model.pb should not be detected."""
    from humpback.config import Settings

    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir)

        # Create a regular directory (not a SavedModel)
        regular_dir = os.path.join(models_dir, "some_dir")
        os.makedirs(regular_dir)
        with open(os.path.join(regular_dir, "random_file.txt"), "w") as f:
            f.write("not a model")

        settings = Settings(models_dir=models_dir)
        files = model_registry_service.scan_model_files(settings)

        assert len(files) == 0
