"""Integration smoke test for the Pass 3 event classifier pipeline.

Exercises the full end-to-end path:
1. Train a tiny event classifier CNN on synthetic event crops.
2. Run inference on fixture events via the inference module.
3. Verify ``typed_events.parquet`` schema and per-type score ranges.
4. Exercise the full worker path: create job -> run worker -> verify
   complete status + typed_event_count.
"""

from __future__ import annotations

import json
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from humpback.call_parsing.event_classifier.inference import (
    classify_events,
    load_event_classifier,
)
from humpback.call_parsing.event_classifier.trainer import (
    EventClassifierTrainingConfig,
    train_event_classifier,
)
from humpback.call_parsing.storage import (
    read_typed_events,
    segmentation_job_dir,
    write_events,
)
from humpback.call_parsing.types import Event, TYPED_EVENT_SCHEMA
from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.vocalization import VocalizationClassifierModel
from humpback.schemas.call_parsing import SegmentationFeatureConfig
from humpback.workers.event_classification_worker import run_event_classification_job


@dataclass
class FakeSample:
    start_sec: float
    end_sec: float
    type_index: int
    audio_file_id: str = "af-smoke"


FEATURE_CONFIG = SegmentationFeatureConfig()
VOCABULARY = ["upcall", "moan"]


def _make_audio(duration_sec: float = 5.0, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(duration_sec * sr)).astype(np.float32)


def _const_loader(audio: np.ndarray):  # noqa: ANN202
    def _load(_obj: object) -> np.ndarray:
        return audio

    return _load


def _write_wav(path: Path, duration_sec: float = 5.0) -> None:
    sr = 16000
    n = int(duration_sec * sr)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


@pytest.fixture
def trained_model_dir(tmp_path: Path) -> Path:
    """Train a tiny event classifier and return the model directory."""
    audio = _make_audio(5.0)
    samples = [
        FakeSample(0.0, 1.5, type_index=0, audio_file_id="f1"),
        FakeSample(1.0, 2.5, type_index=0, audio_file_id="f1"),
        FakeSample(0.5, 2.0, type_index=1, audio_file_id="f2"),
        FakeSample(1.5, 3.0, type_index=1, audio_file_id="f2"),
        FakeSample(0.0, 1.0, type_index=0, audio_file_id="f3"),
        FakeSample(2.0, 3.5, type_index=1, audio_file_id="f3"),
    ]
    config = EventClassifierTrainingConfig(
        epochs=2, batch_size=4, min_examples_per_type=2, val_fraction=0.34
    )
    model_dir = tmp_path / "model"
    train_event_classifier(
        samples=samples,
        vocabulary=VOCABULARY,
        feature_config=FEATURE_CONFIG,
        audio_loader=_const_loader(audio),
        config=config,
        model_dir=model_dir,
    )
    return model_dir


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


class TestTrainAndInfer:
    def test_train_and_classify_events(self, trained_model_dir: Path, tmp_path: Path):
        """Train → infer round-trip: typed events have correct schema and sane scores."""
        model, vocabulary, thresholds, feature_config = load_event_classifier(
            trained_model_dir
        )
        audio = _make_audio(10.0)
        events = [
            Event("ev-1", "r1", 0.5, 2.0, 1.25, 0.9),
            Event("ev-2", "r1", 3.0, 5.0, 4.0, 0.85),
            Event("ev-3", "r1", 6.0, 8.0, 7.0, 0.7),
        ]
        typed_events = classify_events(
            model=model,
            events=events,
            audio_loader=_const_loader(audio),
            feature_config=feature_config,
            vocabulary=vocabulary,
            thresholds=thresholds,
        )

        assert len(typed_events) >= len(events)

        for te in typed_events:
            assert te.type_name in VOCABULARY
            assert 0.0 <= te.score <= 1.0
            assert te.event_id in {"ev-1", "ev-2", "ev-3"}

    def test_typed_events_parquet_schema(self, trained_model_dir: Path, tmp_path: Path):
        """typed_events.parquet has the expected schema and value ranges."""
        from humpback.call_parsing.storage import write_typed_events

        model, vocabulary, thresholds, feature_config = load_event_classifier(
            trained_model_dir
        )
        audio = _make_audio(5.0)
        events = [Event("ev-A", "r1", 0.0, 1.5, 0.75, 0.95)]
        typed_events = classify_events(
            model=model,
            events=events,
            audio_loader=_const_loader(audio),
            feature_config=feature_config,
            vocabulary=vocabulary,
            thresholds=thresholds,
        )

        out_path = tmp_path / "typed_events.parquet"
        write_typed_events(out_path, typed_events)

        table = pq.read_table(out_path)
        assert table.schema.names == TYPED_EVENT_SCHEMA.names
        assert table.num_rows == len(typed_events)
        for col_name in ("score",):
            values = table.column(col_name).to_pylist()
            assert all(0.0 <= v <= 1.0 for v in values)


class TestWorkerEndToEnd:
    async def test_worker_completes_classification_job(
        self, session_factory, trained_model_dir: Path, tmp_path: Path
    ):
        """Full worker path: queued job -> run_event_classification_job -> complete."""
        storage_root = tmp_path / "storage"
        storage_root.mkdir()
        settings = Settings(
            storage_root=storage_root,
            database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        )

        # Write a test audio file
        audio_dir = tmp_path / "audio_src"
        audio_dir.mkdir()
        _write_wav(audio_dir / "test.wav", duration_sec=5.0)

        # Write upstream events.parquet
        seg_job_id = "seg-job-smoke"
        seg_dir = segmentation_job_dir(storage_root, seg_job_id)
        seg_dir.mkdir(parents=True)
        upstream_events = [
            Event("ev-w1", "r1", 0.5, 2.0, 1.25, 0.9),
            Event("ev-w2", "r1", 2.5, 4.0, 3.25, 0.8),
        ]
        write_events(seg_dir / "events.parquet", upstream_events)

        async with session_factory() as session:
            # Create required DB rows
            af = AudioFile(
                id="af-smoke",
                filename="test.wav",
                source_folder=str(audio_dir),
                checksum_sha256="fake-sha256-smoke",
                duration_seconds=5.0,
            )
            session.add(af)

            region_job = RegionDetectionJob(
                id="rj-smoke",
                status="complete",
                audio_file_id="af-smoke",
                classifier_model_id="fake-model",
            )
            session.add(region_job)

            seg_job = EventSegmentationJob(
                id=seg_job_id,
                status="complete",
                region_detection_job_id="rj-smoke",
                segmentation_model_id="fake-seg-model",
                event_count=2,
            )
            session.add(seg_job)

            voc_model = VocalizationClassifierModel(
                id="vm-smoke",
                name="smoke-test-model",
                model_dir_path=str(trained_model_dir),
                vocabulary_snapshot=json.dumps(VOCABULARY),
                per_class_thresholds=json.dumps({"upcall": 0.5, "moan": 0.5}),
                model_family="pytorch_event_cnn",
                input_mode="segmented_event",
            )
            session.add(voc_model)

            ec_job = EventClassificationJob(
                id="ec-smoke",
                status="running",
                event_segmentation_job_id=seg_job_id,
                vocalization_model_id="vm-smoke",
            )
            session.add(ec_job)
            await session.commit()

            await run_event_classification_job(session, ec_job, settings)

            refreshed = await session.get(EventClassificationJob, "ec-smoke")
            assert refreshed is not None
            assert refreshed.status == "complete"
            assert refreshed.typed_event_count is not None
            assert refreshed.typed_event_count >= 2

            out_parquet = (
                storage_root
                / "call_parsing"
                / "classification"
                / "ec-smoke"
                / "typed_events.parquet"
            )
            assert out_parquet.exists()
            typed_events = read_typed_events(out_parquet)
            assert len(typed_events) >= 2
            for te in typed_events:
                assert te.type_name in VOCABULARY
                assert 0.0 <= te.score <= 1.0
