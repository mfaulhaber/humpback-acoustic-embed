"""Integration tests for the Pass 2 event segmentation worker.

Exercises the happy path (claim -> load checkpoint -> decode events ->
``events.parquet``), the failure path (inference raises -> cleanup +
failed row), and the upstream-not-complete guard. Audio, regions, and
the checkpoint are all synthesized per-test so the suite is
self-contained and CPU-only.

Hydrophone-source worker coverage is deferred — see
``docs/plans/backlog.md``.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.call_parsing.storage import (
    read_events,
    region_job_dir,
    segmentation_job_dir,
    write_regions,
)
from humpback.call_parsing.types import Region
from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.ml.checkpointing import save_checkpoint
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
)
from humpback.storage import ensure_dir
from humpback.workers.event_segmentation_worker import run_one_iteration
from humpback.workers.queue import recover_stale_jobs


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


def _make_settings(tmp_path: Path) -> Settings:
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    return Settings(
        storage_root=storage,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
    )


def _write_tone_wav(
    path: Path,
    duration_sec: float,
    freq: float = 440.0,
    sample_rate: int = 16000,
) -> None:
    n = int(sample_rate * duration_sec)
    samples = [
        int(32767 * 0.7 * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _build_tiny_model_config() -> dict[str, object]:
    """Minimal CRNN architecture parameters the checkpoint round-trips.

    Paired with a checkpoint saved via ``save_checkpoint``, this is
    everything the worker needs to reconstruct the model on load.
    """
    return {
        "model_type": "SegmentationCRNN",
        "n_mels": 16,
        "conv_channels": [4],
        "gru_hidden": 4,
        "gru_layers": 1,
        "feature_config": SegmentationFeatureConfig(n_mels=16).model_dump(),
    }


def _save_tiny_checkpoint(checkpoint_path: Path, *, bias: float) -> dict[str, object]:
    """Build a tiny ``SegmentationCRNN`` and persist it at ``checkpoint_path``.

    ``bias`` is added to the frame-head's output bias so the test harness
    can control whether sigmoid(frame_logits) lands above or below the
    decoder's default ``high_threshold=0.5``. A positive bias pushes
    every frame above threshold so the decoder is guaranteed to emit at
    least one event across the synthetic regions.
    """
    config = _build_tiny_model_config()
    model = SegmentationCRNN(
        n_mels=int(config["n_mels"]),  # type: ignore[arg-type]
        conv_channels=list(config["conv_channels"]),  # type: ignore[arg-type]
        gru_hidden=int(config["gru_hidden"]),  # type: ignore[arg-type]
        gru_layers=int(config["gru_layers"]),  # type: ignore[arg-type]
    )
    with torch.no_grad():
        model.frame_head.bias.fill_(bias)
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        config=config,
    )
    return config


async def _seed_fixture(
    session_factory,
    tmp_path: Path,
    *,
    pass1_status: str = "complete",
    checkpoint_bias: float = 10.0,
) -> tuple[str, str, str, Path]:
    """Seed DB + disk for one event segmentation worker iteration.

    Returns ``(upstream_job_id, seg_model_id, pass2_job_id, storage_root)``.
    """
    settings = _make_settings(tmp_path)
    storage_root = settings.storage_root

    audio_dir = tmp_path / "audio_src"
    audio_path = audio_dir / "sample.wav"
    _write_tone_wav(audio_path, duration_sec=4.0)

    async with session_factory() as session:
        audio_file = AudioFile(
            filename="sample.wav",
            folder_path="",
            source_folder=str(audio_dir),
            checksum_sha256=f"sum-{tmp_path.name}",
            duration_seconds=4.0,
        )
        session.add(audio_file)
        await session.flush()
        audio_file_id = audio_file.id

        upstream_job = RegionDetectionJob(
            audio_file_id=audio_file_id,
            status=pass1_status,
            config_json="{}",
        )
        session.add(upstream_job)
        await session.commit()
        upstream_job_id = upstream_job.id

    regions_dir = ensure_dir(region_job_dir(storage_root, upstream_job_id))
    regions = [
        Region(
            region_id="region-0",
            start_sec=0.5,
            end_sec=2.0,
            padded_start_sec=0.0,
            padded_end_sec=2.5,
            max_score=0.9,
            mean_score=0.7,
            n_windows=3,
        ),
        Region(
            region_id="region-1",
            start_sec=2.5,
            end_sec=3.5,
            padded_start_sec=2.25,
            padded_end_sec=4.0,
            max_score=0.85,
            mean_score=0.6,
            n_windows=2,
        ),
    ]
    write_regions(regions_dir / "regions.parquet", regions)

    checkpoint_path = storage_root / "segmentation_models" / "tiny" / "checkpoint.pt"
    model_config = _save_tiny_checkpoint(checkpoint_path, bias=checkpoint_bias)

    async with session_factory() as session:
        seg_model = SegmentationModel(
            name="tiny-seg",
            model_family="pytorch_crnn",
            model_path=str(checkpoint_path),
            config_json=json.dumps(model_config),
        )
        session.add(seg_model)
        await session.commit()
        seg_model_id = seg_model.id

        decoder_config = SegmentationDecoderConfig()
        pass2_job = EventSegmentationJob(
            region_detection_job_id=upstream_job_id,
            segmentation_model_id=seg_model_id,
            config_json=decoder_config.model_dump_json(),
            status="queued",
        )
        session.add(pass2_job)
        await session.commit()
        pass2_job_id = pass2_job.id

    return upstream_job_id, seg_model_id, pass2_job_id, storage_root


async def test_worker_happy_path_writes_events_and_completes_row(
    session_factory, tmp_path
):
    _, _, pass2_job_id, storage_root = await _seed_fixture(session_factory, tmp_path)
    settings = _make_settings(tmp_path)

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None
    assert claimed.id == pass2_job_id

    async with session_factory() as session:
        refreshed = await session.get(EventSegmentationJob, pass2_job_id)
        assert refreshed is not None
        assert refreshed.status == "complete"
        assert refreshed.error_message is None
        assert refreshed.event_count is not None
        assert refreshed.event_count >= 1
        assert refreshed.started_at is not None
        assert refreshed.completed_at is not None

    events_path = segmentation_job_dir(storage_root, pass2_job_id) / "events.parquet"
    assert events_path.exists()

    events = read_events(events_path)
    assert len(events) == refreshed.event_count
    assert len(events) >= 1

    # Every decoded event must fall within one of the two source regions'
    # padded bounds. The decoder quantizes end_sec to the nearest hop
    # boundary (hop_length / sample_rate = 0.032s), so bounds checks
    # allow one hop of slack on each side.
    hop_tolerance = 512.0 / 16000.0
    for event in events:
        assert event.region_id in {"region-0", "region-1"}
        if event.region_id == "region-0":
            assert -hop_tolerance <= event.start_sec
            assert event.start_sec <= event.end_sec
            assert event.end_sec <= 2.5 + hop_tolerance
        else:
            assert 2.25 - hop_tolerance <= event.start_sec
            assert event.start_sec <= event.end_sec
            assert event.end_sec <= 4.0 + hop_tolerance
        assert 0.0 <= event.segmentation_confidence <= 1.0


async def test_worker_failure_path_cleans_up_and_marks_failed(
    session_factory, tmp_path, monkeypatch
):
    _, _, pass2_job_id, storage_root = await _seed_fixture(session_factory, tmp_path)
    settings = _make_settings(tmp_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("inference exploded")

    monkeypatch.setattr(
        "humpback.workers.event_segmentation_worker._run_inference_pipeline",
        _boom,
    )

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None

    async with session_factory() as session:
        refreshed = await session.get(EventSegmentationJob, pass2_job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None
        assert "inference exploded" in refreshed.error_message
        assert refreshed.event_count is None

    job_dir = segmentation_job_dir(storage_root, pass2_job_id)
    assert not (job_dir / "events.parquet").exists()
    assert list(job_dir.glob("*.tmp")) == []


async def test_worker_fails_when_upstream_pass1_not_complete(session_factory, tmp_path):
    _, _, pass2_job_id, _ = await _seed_fixture(
        session_factory, tmp_path, pass1_status="queued"
    )
    settings = _make_settings(tmp_path)

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None

    async with session_factory() as session:
        refreshed = await session.get(EventSegmentationJob, pass2_job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None
        assert "not complete" in refreshed.error_message


async def test_stale_event_segmentation_job_is_recovered(session_factory):
    stale_ts = datetime.now(timezone.utc) - timedelta(minutes=20)
    async with session_factory() as session:
        job = EventSegmentationJob(
            region_detection_job_id="upstream-x",
            config_json="{}",
            status="running",
            updated_at=stale_ts,
        )
        session.add(job)
        await session.commit()
        job_id = job.id

    async with session_factory() as session:
        recovered_count = await recover_stale_jobs(session)
    assert recovered_count >= 1

    async with session_factory() as session:
        refreshed = await session.get(EventSegmentationJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "queued"
