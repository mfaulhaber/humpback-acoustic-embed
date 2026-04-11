"""Service-layer tests for ``src/humpback/services/call_parsing.py``.

Covers:
- ``RegionDetectionConfig`` default values.
- ``CreateRegionJobRequest`` exactly-one-of validator cases.
- ``create_region_job`` happy paths (audio-file and hydrophone sources).
- ``create_region_job`` FK validation raising ``CallParsingFKError`` for
  each of the four FK-like fields.
- ``create_parent_run`` atomically creating parent + child with the child
  carrying ``parent_run_id``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from humpback.database import Base, create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import CallParsingRun, RegionDetectionJob
from humpback.models.classifier import ClassifierModel
from humpback.models.model_registry import ModelConfig
from humpback.schemas.call_parsing import (
    CreateRegionJobRequest,
    RegionDetectionConfig,
)
from humpback.services.call_parsing import (
    CallParsingFKError,
    create_parent_run,
    create_region_job,
)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture
async def session_factory(tmp_path: Path):
    db_path = tmp_path / "service.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


async def _seed_models_and_audio(session_factory) -> dict[str, str]:
    """Insert one AudioFile + ModelConfig + ClassifierModel, return their ids."""
    async with session_factory() as session:
        audio = AudioFile(
            filename="fixture.wav",
            folder_path="",
            checksum_sha256="svc-fixture-sha",
        )
        model_config = ModelConfig(
            name="perch_v2_test",
            display_name="Perch v2 (test)",
            path="/tmp/perch.tflite",
            vector_dim=1536,
        )
        classifier = ClassifierModel(
            name="binary-test",
            model_path="/tmp/binary.joblib",
            model_version="perch_v2_test",
            vector_dim=1536,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add_all([audio, model_config, classifier])
        await session.commit()
        return {
            "audio_file_id": audio.id,
            "model_config_id": model_config.id,
            "classifier_model_id": classifier.id,
        }


# --- RegionDetectionConfig ------------------------------------------------


def test_region_detection_config_defaults():
    """ADR-049 defaults: 5 s / 1 s hop, 0.70/0.45 hysteresis, 1 s sym padding."""
    cfg = RegionDetectionConfig()
    assert cfg.window_size_seconds == 5.0
    assert cfg.hop_seconds == 1.0
    assert cfg.high_threshold == 0.70
    assert cfg.low_threshold == 0.45
    assert cfg.padding_sec == 1.0
    assert cfg.min_region_duration_sec == 0.0
    assert cfg.stream_chunk_sec == 1800.0


def test_region_detection_config_round_trips_json():
    cfg = RegionDetectionConfig(padding_sec=2.5, high_threshold=0.9)
    payload = cfg.model_dump_json()
    restored = RegionDetectionConfig.model_validate_json(payload)
    assert restored.padding_sec == 2.5
    assert restored.high_threshold == 0.9
    # Untouched fields keep their defaults.
    assert restored.hop_seconds == 1.0


# --- CreateRegionJobRequest validator ------------------------------------
#
# These cases go through ``model_validate`` on a dict rather than ``**body``
# unpacking because Pyright narrows the dict value type and refuses to
# match it against the model's heterogeneous ``__init__`` signature — the
# ``model_validate`` path is the idiomatic Pydantic v2 entry point for
# dict-shaped payloads anyway.


def _valid_file_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "audio_file_id": "af-1",
        "model_config_id": "mc-1",
        "classifier_model_id": "cm-1",
    }
    body.update(overrides)
    return body


def _valid_hydro_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "hydrophone_id": "rpi_north_sjc",
        "start_timestamp": 1_700_000_000.0,
        "end_timestamp": 1_700_003_600.0,
        "model_config_id": "mc-1",
        "classifier_model_id": "cm-1",
    }
    body.update(overrides)
    return body


def test_create_region_job_request_accepts_file_source():
    req = CreateRegionJobRequest.model_validate(_valid_file_body())
    assert req.audio_file_id == "af-1"
    assert req.hydrophone_id is None
    # Config defaults are applied when the caller omits it.
    assert req.config.padding_sec == 1.0


def test_create_region_job_request_accepts_hydrophone_source():
    req = CreateRegionJobRequest.model_validate(_valid_hydro_body())
    assert req.audio_file_id is None
    assert req.hydrophone_id == "rpi_north_sjc"
    assert req.start_timestamp == 1_700_000_000.0
    assert req.end_timestamp == 1_700_003_600.0


def test_create_region_job_request_rejects_both_sources():
    body = _valid_file_body(
        hydrophone_id="rpi_north_sjc",
        start_timestamp=1.0,
        end_timestamp=2.0,
    )
    with pytest.raises(ValidationError) as excinfo:
        CreateRegionJobRequest.model_validate(body)
    assert "not both" in str(excinfo.value)


def test_create_region_job_request_rejects_neither_source():
    body: dict[str, object] = {
        "model_config_id": "mc-1",
        "classifier_model_id": "cm-1",
    }
    with pytest.raises(ValidationError) as excinfo:
        CreateRegionJobRequest.model_validate(body)
    assert "exactly one" in str(excinfo.value)


def test_create_region_job_request_rejects_partial_hydrophone_triple():
    # Only hydrophone_id, no timestamps.
    body: dict[str, object] = {
        "hydrophone_id": "rpi_north_sjc",
        "model_config_id": "mc-1",
        "classifier_model_id": "cm-1",
    }
    with pytest.raises(ValidationError):
        CreateRegionJobRequest.model_validate(body)


def test_create_region_job_request_rejects_inverted_timestamps():
    body = _valid_hydro_body(start_timestamp=2.0, end_timestamp=1.0)
    with pytest.raises(ValidationError) as excinfo:
        CreateRegionJobRequest.model_validate(body)
    assert "strictly after" in str(excinfo.value)


def test_create_region_job_request_rejects_equal_timestamps():
    body = _valid_hydro_body(start_timestamp=1.0, end_timestamp=1.0)
    with pytest.raises(ValidationError):
        CreateRegionJobRequest.model_validate(body)


# --- create_region_job service tests -------------------------------------


async def test_create_region_job_happy_path_file_source(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        audio_file_id=ids["audio_file_id"],
        model_config_id=ids["model_config_id"],
        classifier_model_id=ids["classifier_model_id"],
    )

    async with session_factory() as session:
        job = await create_region_job(session, request)
        await session.commit()
        job_id = job.id

    async with session_factory() as session:
        refreshed = await session.get(RegionDetectionJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "queued"
        assert refreshed.audio_file_id == ids["audio_file_id"]
        assert refreshed.hydrophone_id is None
        assert refreshed.model_config_id == ids["model_config_id"]
        assert refreshed.classifier_model_id == ids["classifier_model_id"]
        # config_json round-trips and includes the defaults.
        config_dict = json.loads(refreshed.config_json or "{}")
        assert config_dict["padding_sec"] == 1.0
        assert config_dict["high_threshold"] == 0.70


async def test_create_region_job_happy_path_hydrophone_source(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        hydrophone_id="rpi_north_sjc",
        start_timestamp=1_700_000_000.0,
        end_timestamp=1_700_003_600.0,
        model_config_id=ids["model_config_id"],
        classifier_model_id=ids["classifier_model_id"],
    )

    async with session_factory() as session:
        job = await create_region_job(session, request)
        await session.commit()
        job_id = job.id

    async with session_factory() as session:
        refreshed = await session.get(RegionDetectionJob, job_id)
        assert refreshed is not None
        assert refreshed.audio_file_id is None
        assert refreshed.hydrophone_id == "rpi_north_sjc"
        assert refreshed.start_timestamp == 1_700_000_000.0
        assert refreshed.end_timestamp == 1_700_003_600.0


async def test_create_region_job_persists_parent_run_id(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    async with session_factory() as session:
        parent = CallParsingRun(
            status="queued",
            audio_file_id=ids["audio_file_id"],
        )
        session.add(parent)
        await session.commit()
        parent_id = parent.id

    request = CreateRegionJobRequest(
        audio_file_id=ids["audio_file_id"],
        model_config_id=ids["model_config_id"],
        classifier_model_id=ids["classifier_model_id"],
        parent_run_id=parent_id,
    )
    async with session_factory() as session:
        job = await create_region_job(session, request)
        await session.commit()
        assert job.parent_run_id == parent_id


async def test_create_region_job_missing_audio_file(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        audio_file_id="does-not-exist",
        model_config_id=ids["model_config_id"],
        classifier_model_id=ids["classifier_model_id"],
    )
    async with session_factory() as session:
        with pytest.raises(CallParsingFKError) as excinfo:
            await create_region_job(session, request)
    assert excinfo.value.field == "audio_file_id"
    assert excinfo.value.value == "does-not-exist"


async def test_create_region_job_missing_hydrophone(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        hydrophone_id="not-a-real-hydrophone",
        start_timestamp=1.0,
        end_timestamp=2.0,
        model_config_id=ids["model_config_id"],
        classifier_model_id=ids["classifier_model_id"],
    )
    async with session_factory() as session:
        with pytest.raises(CallParsingFKError) as excinfo:
            await create_region_job(session, request)
    assert excinfo.value.field == "hydrophone_id"


async def test_create_region_job_missing_model_config(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        audio_file_id=ids["audio_file_id"],
        model_config_id="mc-nope",
        classifier_model_id=ids["classifier_model_id"],
    )
    async with session_factory() as session:
        with pytest.raises(CallParsingFKError) as excinfo:
            await create_region_job(session, request)
    assert excinfo.value.field == "model_config_id"


async def test_create_region_job_missing_classifier_model(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        audio_file_id=ids["audio_file_id"],
        model_config_id=ids["model_config_id"],
        classifier_model_id="cm-nope",
    )
    async with session_factory() as session:
        with pytest.raises(CallParsingFKError) as excinfo:
            await create_region_job(session, request)
    assert excinfo.value.field == "classifier_model_id"


# --- create_parent_run service tests -------------------------------------


async def test_create_parent_run_creates_atomic_parent_and_child(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        audio_file_id=ids["audio_file_id"],
        model_config_id=ids["model_config_id"],
        classifier_model_id=ids["classifier_model_id"],
    )

    async with session_factory() as session:
        run = await create_parent_run(session, request)
        run_id = run.id
        child_id = run.region_detection_job_id

    async with session_factory() as session:
        refreshed_run = await session.get(CallParsingRun, run_id)
        assert refreshed_run is not None
        assert refreshed_run.audio_file_id == ids["audio_file_id"]
        assert refreshed_run.region_detection_job_id == child_id
        assert refreshed_run.config_snapshot is not None
        snapshot = json.loads(refreshed_run.config_snapshot)
        assert snapshot["padding_sec"] == 1.0

        assert child_id is not None
        refreshed_child = await session.get(RegionDetectionJob, child_id)
        assert refreshed_child is not None
        assert refreshed_child.parent_run_id == run_id
        assert refreshed_child.audio_file_id == ids["audio_file_id"]
        assert refreshed_child.model_config_id == ids["model_config_id"]


async def test_create_parent_run_hydrophone_mirrors_source(session_factory):
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        hydrophone_id="rpi_north_sjc",
        start_timestamp=1_700_000_000.0,
        end_timestamp=1_700_086_400.0,
        model_config_id=ids["model_config_id"],
        classifier_model_id=ids["classifier_model_id"],
    )

    async with session_factory() as session:
        run = await create_parent_run(session, request)
        run_id = run.id

    async with session_factory() as session:
        refreshed = await session.get(CallParsingRun, run_id)
        assert refreshed is not None
        assert refreshed.hydrophone_id == "rpi_north_sjc"
        assert refreshed.start_timestamp == 1_700_000_000.0
        assert refreshed.end_timestamp == 1_700_086_400.0
        assert refreshed.audio_file_id is None

        assert refreshed.region_detection_job_id is not None
        child = await session.get(RegionDetectionJob, refreshed.region_detection_job_id)
        assert child is not None
        assert child.hydrophone_id == "rpi_north_sjc"
        assert child.start_timestamp == 1_700_000_000.0
        assert child.end_timestamp == 1_700_086_400.0


async def test_create_parent_run_missing_fk_raises(session_factory):
    """Parent creation rolls back if the Pass 1 child FK resolution fails."""
    ids = await _seed_models_and_audio(session_factory)
    request = CreateRegionJobRequest(
        audio_file_id=ids["audio_file_id"],
        model_config_id="mc-nope",
        classifier_model_id=ids["classifier_model_id"],
    )

    async with session_factory() as session:
        with pytest.raises(CallParsingFKError):
            await create_parent_run(session, request)

    # The rollback should have removed the in-progress parent row that
    # was flushed before FK validation ran on the child.
    async with session_factory() as session:
        from sqlalchemy import func, select

        count = (
            await session.execute(select(func.count(CallParsingRun.id)))
        ).scalar_one()
        assert count == 0
