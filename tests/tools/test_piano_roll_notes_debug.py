"""Smoke tests for ``tools/piano_roll_notes_debug.py``.

Exercises the CLI's full resolution chain against an in-memory encoder
job fixture (no real audio files) and asserts a non-empty PNG is
written, plus a couple of negative argument-validation cases.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.call_parsing.storage import segmentation_job_dir, write_events
from humpback.call_parsing.types import Event
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.storage import event_encoder_dir, event_encoder_tokens_path

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from piano_roll_notes_debug import _async_main, main  # noqa: E402  # pyright: ignore[reportMissingImports]
from piano_roll_notes_registry import EXTRACTORS  # noqa: E402  # pyright: ignore[reportMissingImports]

SAMPLE_RATE = 22050


def _harmonic_stack(
    *, fundamental_hz: float, duration_s: float, harmonic_amplitudes: list[float]
) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.zeros_like(t)
    for n, amp in enumerate(harmonic_amplitudes, start=1):
        audio += amp * np.sin(2 * np.pi * fundamental_hz * n * t)
    return audio.astype(np.float32)


def _write_event_tokens_parquet(
    path: Path,
    *,
    event_ids: Iterable[str],
    source_sequence_key: str,
    k: int,
    token_id: int,
) -> None:
    schema = pa.schema(
        [
            pa.field("k", pa.int32(), nullable=False),
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("source_sequence_key", pa.string(), nullable=False),
            pa.field("sequence_index", pa.int32(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
        ]
    )
    rows = [
        {
            "k": k,
            "event_id": eid,
            "source_sequence_key": source_sequence_key,
            "sequence_index": idx,
            "token_id": token_id,
        }
        for idx, eid in enumerate(event_ids)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


async def _seed_encoder_job(
    session, settings, tmp_path: Path
) -> tuple[str, np.ndarray]:
    audio_file = AudioFile(
        filename="synthetic.wav",
        folder_path=str(tmp_path),
        checksum_sha256="debug-cli-fixture",
        duration_seconds=1.0,
        sample_rate_original=SAMPLE_RATE,
    )
    session.add(audio_file)
    await session.commit()
    await session.refresh(audio_file)

    region = RegionDetectionJob(
        status=JobStatus.complete.value,
        audio_file_id=audio_file.id,
        start_timestamp=1000.0,
        end_timestamp=1001.0,
    )
    session.add(region)
    await session.commit()
    await session.refresh(region)

    seg = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region.id,
    )
    session.add(seg)
    await session.commit()
    await session.refresh(seg)

    encoder = EventEncoderJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg.id,
        event_source_mode="raw",
        continuous_embedding_job_id="cej-stub",
        continuous_embedding_signature="cej-sig",
        tokenizer_version="crnn-event-encoder-v3",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature="tok-sig-debug",
    )
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    events = [
        Event(
            event_id="ev-debug-0",
            region_id=region.id,
            start_sec=0.10,
            end_sec=0.50,
            center_sec=0.30,
            segmentation_confidence=0.9,
        ),
    ]
    seg_path = segmentation_job_dir(settings.storage_root, seg.id) / "events.parquet"
    write_events(seg_path, events)

    encoder_dir_path = event_encoder_dir(settings.storage_root, encoder.id)
    encoder_dir_path.mkdir(parents=True, exist_ok=True)
    _write_event_tokens_parquet(
        event_encoder_tokens_path(settings.storage_root, encoder.id),
        event_ids=[e.event_id for e in events],
        source_sequence_key="test:seq:0:1",
        k=50,
        token_id=42,
    )

    buffer = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    stack = _harmonic_stack(
        fundamental_hz=200.0,
        duration_s=0.40,
        harmonic_amplitudes=[0.40, 0.30, 0.20, 0.15],
    )
    start_idx = int(0.10 * SAMPLE_RATE)
    buffer[start_idx : start_idx + stack.size] += stack
    return encoder.id, buffer


def _patch_for_cli(monkeypatch, buffer: np.ndarray, settings) -> None:
    """Wire the CLI to the test session / settings / synthetic audio.

    Patches ``create_engine`` to construct a fresh AsyncEngine against
    the same SQLite test database URL; the CLI's own ``await
    engine.dispose()`` then cleans up its connection without affecting
    the test fixture's engine.
    """

    import piano_roll_notes_debug as debug_mod  # pyright: ignore[reportMissingImports]
    from humpback.database import create_engine as real_create_engine
    from humpback.workers import piano_roll_notes_worker

    async def _fake_build_audio_provider(*_args, **_kwargs):
        def _provider(_event):
            return buffer, 0.0

        return _provider

    monkeypatch.setattr(
        piano_roll_notes_worker,
        "_build_audio_provider",
        _fake_build_audio_provider,
    )
    monkeypatch.setattr(debug_mod, "load_dotenv", lambda: None)
    monkeypatch.setattr(debug_mod, "Settings", lambda: settings)
    monkeypatch.setattr(
        debug_mod,
        "create_engine",
        lambda *_a, **_k: real_create_engine(settings.database_url),
    )


def test_registry_has_expected_phase1_keys() -> None:
    assert set(EXTRACTORS) == {"v3", "v4", "v5", "v6"}
    assert all(callable(fn) for fn in EXTRACTORS.values())


@pytest.mark.asyncio
async def test_cli_renders_png_for_token(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder_id, buffer = await _seed_encoder_job(session, settings, tmp_path)
    _patch_for_cli(monkeypatch, buffer, settings)

    out_path = tmp_path / "out.png"
    rc = await _async_main(
        [
            "--job",
            encoder_id,
            "--token",
            "0",
            "--variants",
            "v4,v5",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.mark.asyncio
async def test_cli_renders_png_for_event_id(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder_id, buffer = await _seed_encoder_job(session, settings, tmp_path)
    _patch_for_cli(monkeypatch, buffer, settings)

    out_path = tmp_path / "out_eid.png"
    rc = await _async_main(
        [
            "--job",
            encoder_id,
            "--event-id",
            "ev-debug-0",
            "--variants",
            "v4",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_cli_token_and_event_id_mutually_exclusive(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--job",
                "x",
                "--token",
                "0",
                "--event-id",
                "y",
                "--out",
                str(tmp_path / "no.png"),
            ]
        )
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "not allowed with" in err or "mutually exclusive" in err


def test_cli_requires_token_or_event_id(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--job", "x", "--out", str(tmp_path / "no.png")])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "--token" in err or "--event-id" in err


def test_cli_rejects_unknown_variant(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--job",
                "x",
                "--token",
                "0",
                "--variants",
                "v4,not-a-real-variant",
                "--out",
                str(tmp_path / "no.png"),
            ]
        )
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "not-a-real-variant" in err
