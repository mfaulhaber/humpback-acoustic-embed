"""Tests for the Pass 3 event classifier bootstrap script.

Verifies:
- Single-label windows produce correctly labeled events.
- Multi-label windows are excluded.
- ``(Negative)`` windows are excluded.
- Events outside the window time range are not included.

All fixtures use hydrophone-sourced detection jobs with mocked
``resolve_timeline_audio`` for audio fetching.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from humpback.call_parsing.types import Event
from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    write_detection_row_store,
)
from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.call_parsing import SegmentationModel
from humpback.models.classifier import DetectionJob
from humpback.models.labeling import VocalizationLabel
from scripts.bootstrap_event_classifier_dataset import run_bootstrap

JOB_START_TS = 0.0
JOB_END_TS = 100.0
HYDROPHONE_ID = "test_hydrophone"


def _make_row(row_id: str, start_utc: float, end_utc: float) -> dict[str, str]:
    row = {f: "" for f in ROW_STORE_FIELDNAMES}
    row["row_id"] = row_id
    row["start_utc"] = str(start_utc)
    row["end_utc"] = str(end_utc)
    return row


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


def _settings(tmp_path: Path) -> Settings:
    storage = tmp_path / "storage"
    storage.mkdir(exist_ok=True)
    return Settings(
        storage_root=storage,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
    )


def _setup_fake_segmentation_model(tmp_path: Path) -> tuple[str, str]:
    """Create a fake segmentation model checkpoint and return (model_path, config_json)."""

    from humpback.call_parsing.segmentation.model import SegmentationCRNN
    from humpback.ml.checkpointing import save_checkpoint

    model_dir = tmp_path / "seg_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "model.pt"

    model = SegmentationCRNN(
        n_mels=64, conv_channels=[8, 16], gru_hidden=8, gru_layers=1
    )
    save_checkpoint(path=checkpoint_path, model=model, optimizer=None, config={})

    config_json = json.dumps(
        {
            "n_mels": 64,
            "conv_channels": [8, 16],
            "gru_hidden": 8,
            "gru_layers": 1,
            "feature_config": {"sample_rate": 16000, "n_mels": 64},
        }
    )
    return str(checkpoint_path), config_json


def _mock_segmentation_events(
    events: list[Event],
):
    """Return a patcher that replaces segmentation inference with fixed events."""

    def _fake_run_segmentation_on_window(
        *, model, audio, start_sec, end_sec, target_sr, feature_config, decoder_config
    ):
        return events

    return patch(
        "scripts.bootstrap_event_classifier_dataset._run_segmentation_on_window",
        side_effect=_fake_run_segmentation_on_window,
    )


def _mock_timeline_audio():
    """Return a patcher that replaces resolve_timeline_audio with synthetic audio."""

    def _fake_resolve(**kwargs):
        duration_sec = kwargs["duration_sec"]
        target_sr = kwargs["target_sr"]
        n = int(target_sr * duration_sec)
        return np.zeros(n, dtype=np.float32)

    return patch(
        "scripts.bootstrap_event_classifier_dataset.resolve_timeline_audio",
        side_effect=_fake_resolve,
    )


async def _setup_detection_job(
    session,
    *,
    job_id: str,
    row_store_path: Path,
    rows: list[dict[str, str]],
    labels: list[tuple[str, str]],
    hydrophone_id: str = HYDROPHONE_ID,
    start_timestamp: float = JOB_START_TS,
    end_timestamp: float = JOB_END_TS,
):
    """Create a hydrophone detection job, row store, and vocalization labels."""
    dj = DetectionJob(
        id=job_id,
        classifier_model_id="fake-model",
        hydrophone_id=hydrophone_id,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        output_row_store_path=str(row_store_path),
        status="complete",
    )
    session.add(dj)

    for row_id, label_name in labels:
        lb = VocalizationLabel(
            detection_job_id=job_id,
            row_id=row_id,
            label=label_name,
        )
        session.add(lb)

    write_detection_row_store(row_store_path, rows)
    await session.commit()


class TestSingleLabelWindow:
    async def test_produces_labeled_events(self, session_factory, tmp_path):
        """A single-label window produces correct labeled events."""
        settings = _settings(tmp_path)

        row_store_path = tmp_path / "rows.parquet"
        rows = [_make_row("row-1", 1.0, 6.0)]

        model_path, config_json = _setup_fake_segmentation_model(tmp_path)

        async with session_factory() as session:
            await _setup_detection_job(
                session,
                job_id="dj-1",
                row_store_path=row_store_path,
                rows=rows,
                labels=[("row-1", "upcall")],
            )

            seg_model = SegmentationModel(
                id="seg-1",
                name="test-seg",
                model_family="segmentation_crnn",
                model_path=model_path,
                config_json=config_json,
            )
            session.add(seg_model)
            await session.commit()

            # Events within the window duration [0, 5.0] (buffer-relative)
            mock_events = [
                Event(
                    "e1",
                    "r1",
                    start_sec=0.5,
                    end_sec=2.0,
                    center_sec=1.25,
                    segmentation_confidence=0.9,
                ),
                Event(
                    "e2",
                    "r1",
                    start_sec=3.0,
                    end_sec=4.5,
                    center_sec=3.75,
                    segmentation_confidence=0.8,
                ),
            ]

            output_path = tmp_path / "output.json"
            with _mock_segmentation_events(mock_events), _mock_timeline_audio():
                result = await run_bootstrap(
                    session,
                    detection_job_ids=["dj-1"],
                    segmentation_model_id="seg-1",
                    dry_run=False,
                    output_path=output_path,
                    storage_root=settings.storage_root,
                    settings=settings,
                )

            assert result.inserted == 2
            samples = json.loads(output_path.read_text())
            assert len(samples) == 2
            assert all(s["type_name"] == "upcall" for s in samples)
            assert all(s["hydrophone_id"] == HYDROPHONE_ID for s in samples)
            assert "audio_file_id" not in samples[0]


class TestMultiLabelExcluded:
    async def test_multi_label_window_skipped(self, session_factory, tmp_path):
        """Windows with more than one distinct type label are excluded."""
        settings = _settings(tmp_path)

        row_store_path = tmp_path / "rows.parquet"
        rows = [_make_row("row-ml", 1.0, 6.0)]

        model_path, config_json = _setup_fake_segmentation_model(tmp_path)

        async with session_factory() as session:
            await _setup_detection_job(
                session,
                job_id="dj-2",
                row_store_path=row_store_path,
                rows=rows,
                labels=[("row-ml", "upcall"), ("row-ml", "downcall")],
            )

            seg_model = SegmentationModel(
                id="seg-2",
                name="test-seg",
                model_family="segmentation_crnn",
                model_path=model_path,
                config_json=config_json,
            )
            session.add(seg_model)
            await session.commit()

            output_path = tmp_path / "output.json"
            with _mock_segmentation_events([]), _mock_timeline_audio():
                result = await run_bootstrap(
                    session,
                    detection_job_ids=["dj-2"],
                    segmentation_model_id="seg-2",
                    dry_run=False,
                    output_path=output_path,
                    storage_root=settings.storage_root,
                    settings=settings,
                )

            assert result.inserted == 0
            assert result.skipped.get("multi-label", 0) == 1


class TestNegativeExcluded:
    async def test_negative_only_window_skipped(self, session_factory, tmp_path):
        """Windows with only ``(Negative)`` labels are excluded."""
        settings = _settings(tmp_path)

        row_store_path = tmp_path / "rows.parquet"
        rows = [_make_row("row-neg", 1.0, 6.0)]

        model_path, config_json = _setup_fake_segmentation_model(tmp_path)

        async with session_factory() as session:
            await _setup_detection_job(
                session,
                job_id="dj-3",
                row_store_path=row_store_path,
                rows=rows,
                labels=[("row-neg", "(Negative)")],
            )

            seg_model = SegmentationModel(
                id="seg-3",
                name="test-seg",
                model_family="segmentation_crnn",
                model_path=model_path,
                config_json=config_json,
            )
            session.add(seg_model)
            await session.commit()

            output_path = tmp_path / "output.json"
            with _mock_segmentation_events([]), _mock_timeline_audio():
                result = await run_bootstrap(
                    session,
                    detection_job_ids=["dj-3"],
                    segmentation_model_id="seg-3",
                    dry_run=False,
                    output_path=output_path,
                    storage_root=settings.storage_root,
                    settings=settings,
                )

            assert result.inserted == 0
            assert result.skipped.get("(Negative) only", 0) == 1


class TestEventsOutsideWindowExcluded:
    async def test_events_outside_window_not_labeled(self, session_factory, tmp_path):
        """Events whose bounds extend beyond the window are not included."""
        settings = _settings(tmp_path)

        row_store_path = tmp_path / "rows.parquet"
        rows = [_make_row("row-out", 1.0, 6.0)]

        model_path, config_json = _setup_fake_segmentation_model(tmp_path)

        async with session_factory() as session:
            await _setup_detection_job(
                session,
                job_id="dj-4",
                row_store_path=row_store_path,
                rows=rows,
                labels=[("row-out", "upcall")],
            )

            seg_model = SegmentationModel(
                id="seg-4",
                name="test-seg",
                model_family="segmentation_crnn",
                model_path=model_path,
                config_json=config_json,
            )
            session.add(seg_model)
            await session.commit()

            # Window duration is 5.0 s (6.0 - 1.0). One event inside, one past.
            mock_events = [
                Event(
                    "e-in",
                    "r1",
                    start_sec=1.0,
                    end_sec=2.0,
                    center_sec=1.5,
                    segmentation_confidence=0.9,
                ),
                Event(
                    "e-out",
                    "r1",
                    start_sec=4.0,
                    end_sec=6.0,
                    center_sec=5.0,
                    segmentation_confidence=0.8,
                ),
            ]

            output_path = tmp_path / "output.json"
            with _mock_segmentation_events(mock_events), _mock_timeline_audio():
                result = await run_bootstrap(
                    session,
                    detection_job_ids=["dj-4"],
                    segmentation_model_id="seg-4",
                    dry_run=False,
                    output_path=output_path,
                    storage_root=settings.storage_root,
                    settings=settings,
                )

            # Only the event fully inside the window should be included
            assert result.inserted == 1
            samples = json.loads(output_path.read_text())
            assert len(samples) == 1
            # Output has absolute UTC timestamps
            assert samples[0]["start_sec"] == pytest.approx(1.0 + 1.0, abs=0.01)
            assert samples[0]["end_sec"] == pytest.approx(1.0 + 2.0, abs=0.01)
