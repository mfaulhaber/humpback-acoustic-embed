"""Test-bed CLI for the Piano Roll Notes pipeline.

Renders a side-by-side spectrogram + piano-roll PNG for one event of
one encoder job, using one or more registered algorithm variants. Used
during Phase 2 iteration on the v5 candidate; stays in the repo as a
permanent debug surface for any future Piano Roll Notes investigation.

Example::

    uv run python tools/piano_roll_notes_debug.py \\
        --job 690580c5-7804-43c9-bd8d-690691b5d6d4 \\
        --token 47 \\
        --variants v4,v5-candidate \\
        --out /tmp/token47.png

See ``docs/specs/2026-05-24-piano-roll-notes-v5-test-bed-design.md``
§4.1 for the design rationale.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.axes  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from humpback.call_parsing.storage import read_events, segmentation_job_dir  # noqa: E402
from humpback.call_parsing.types import Event  # noqa: E402
from humpback.config import Settings  # noqa: E402
from humpback.database import create_engine, create_session_factory  # noqa: E402
from humpback.models.call_parsing import (  # noqa: E402
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.sequence_models import EventEncoderJob  # noqa: E402
from humpback.processing.note_extractor_v3 import (  # noqa: E402
    ContourFrame,
    NotesV3Result,
    NoteV3,
)
from humpback.processing.piano_roll_cqt import CQTParams, compute_event_cqt  # noqa: E402
from humpback.processing.ridge_path import compute_ridge_path  # noqa: E402
from humpback.storage import (  # noqa: E402
    event_encoder_ridges_path,
    event_encoder_tokens_path,
)
from humpback.workers.piano_roll_notes_worker import (  # noqa: E402
    _load_ridge_sidecar,
    _slice_event_audio,
)

# The tools/ directory is on sys.path (Python adds the script's parent
# dir at startup); the registry lives next to this script as a sibling
# module rather than under a package because tools/ is not a package.
from piano_roll_notes_registry import EXTRACTORS  # noqa: E402


logger = logging.getLogger("piano_roll_notes_debug")


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CliArgs:
    job: str
    token: Optional[int]
    event_id: Optional[str]
    variants: tuple[str, ...]
    out: Path
    pad_seconds: float
    width: int
    height: int


def _parse_args(argv: Sequence[str]) -> _CliArgs:
    parser = argparse.ArgumentParser(
        description="Render spectrogram + piano-roll PNG for one encoder-job event."
    )
    parser.add_argument("--job", required=True, help="Event Encoder job UUID.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--token", type=int, help="sequence_index of the target token."
    )
    selector.add_argument("--event-id", type=str, help="Event UUID (alternative).")
    parser.add_argument(
        "--variants",
        default="v4",
        help="Comma-separated registry keys to render (default: 'v4').",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--pad-seconds", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    ns = parser.parse_args(list(argv))
    variants = tuple(v.strip() for v in str(ns.variants).split(",") if v.strip())
    if not variants:
        parser.error("--variants must list at least one registry key")
    unknown = [v for v in variants if v not in EXTRACTORS]
    if unknown:
        parser.error(
            f"unknown variants: {', '.join(unknown)}; available: "
            + ", ".join(sorted(EXTRACTORS))
        )
    return _CliArgs(
        job=str(ns.job),
        token=int(ns.token) if ns.token is not None else None,
        event_id=str(ns.event_id) if ns.event_id is not None else None,
        variants=variants,
        out=Path(ns.out),
        pad_seconds=float(ns.pad_seconds),
        width=int(ns.width),
        height=int(ns.height),
    )


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedEvent:
    encoder: EventEncoderJob
    event: Event
    region_offset_s: float
    sequence_index: Optional[int]
    audio: np.ndarray
    sample_rate: int
    pad_seconds: float
    ridge_rows: Optional[list[dict[str, Any]]]


def _resolve_event_id_from_token(
    settings: Settings, encoder_id: str, token: int
) -> str:
    tokens_path = event_encoder_tokens_path(settings.storage_root, encoder_id)
    if not tokens_path.exists():
        raise FileNotFoundError(f"event_tokens.parquet not found: {tokens_path}")
    table = pq.read_table(
        tokens_path, columns=["sequence_index", "source_sequence_key", "event_id"]
    )
    matches = table.filter(
        pc.equal(table["sequence_index"], int(token))  # type: ignore[attr-defined]
    )
    if matches.num_rows == 0:
        raise ValueError(f"no token row with sequence_index={token} in {tokens_path}")
    sequences = pc.unique(  # type: ignore[attr-defined]
        matches["source_sequence_key"]
    ).to_pylist()
    if len(sequences) > 1:
        raise ValueError(
            f"sequence_index={token} is ambiguous across "
            f"{len(sequences)} sequences in {tokens_path}; "
            "pass --event-id explicitly"
        )
    return str(matches["event_id"][0].as_py())


async def _resolve(
    session: AsyncSession,
    settings: Settings,
    args: _CliArgs,
) -> _ResolvedEvent:
    # Import inline to avoid circular load order with the worker module.
    from humpback.workers.piano_roll_notes_worker import _build_audio_provider

    encoder = await session.get(EventEncoderJob, args.job)
    if encoder is None:
        raise ValueError(f"event_encoder_job not found: {args.job}")

    seg_job = await session.get(EventSegmentationJob, encoder.event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(
            f"event_segmentation_job not found: {encoder.event_segmentation_job_id}"
        )

    region_job = await session.get(RegionDetectionJob, seg_job.region_detection_job_id)
    if region_job is None:
        raise ValueError(
            f"region_detection_job not found: {seg_job.region_detection_job_id}"
        )

    events_path = (
        segmentation_job_dir(settings.storage_root, seg_job.id) / "events.parquet"
    )
    if not events_path.exists():
        raise FileNotFoundError(f"events.parquet not found: {events_path}")
    events = read_events(events_path)
    if not events:
        raise ValueError(f"events.parquet has no rows: {events_path}")

    target_event_id: str
    sequence_index: Optional[int] = None
    if args.token is not None:
        target_event_id = _resolve_event_id_from_token(settings, encoder.id, args.token)
        sequence_index = args.token
    else:
        assert args.event_id is not None
        target_event_id = args.event_id

    target_event: Optional[Event] = None
    for ev in events:
        if ev.event_id == target_event_id:
            target_event = ev
            break
    if target_event is None:
        raise ValueError(f"event_id={target_event_id} not present in {events_path}")

    # CQT target_sample_rate is the canonical SR for note extractors.
    target_sr = CQTParams().target_sample_rate
    audio_provider = await _build_audio_provider(
        session, settings, region_job, events, target_sr
    )
    audio = _slice_event_audio(
        target_event,
        audio_provider,
        target_sr=target_sr,
        pad_seconds=args.pad_seconds,
    )
    if audio.size == 0:
        raise ValueError(f"audio slice is empty for event {target_event_id}")

    ridges_path = event_encoder_ridges_path(
        settings.storage_root, encoder.id, encoder.tokenizer_version
    )
    sidecar_by_event = _load_ridge_sidecar(ridges_path)
    ridge_rows = (
        sidecar_by_event.get(target_event_id) if sidecar_by_event is not None else None
    )

    region_offset = float(region_job.start_timestamp or 0.0)

    logger.info(
        "resolved | encoder=%s | seg=%s | region=%s | event=%s | seq_idx=%s | "
        "audio_samples=%d | ridge_rows=%s",
        encoder.id,
        seg_job.id,
        region_job.id,
        target_event_id,
        "?" if sequence_index is None else str(sequence_index),
        int(audio.size),
        "0" if ridge_rows is None else str(len(ridge_rows)),
    )

    return _ResolvedEvent(
        encoder=encoder,
        event=target_event,
        region_offset_s=region_offset,
        sequence_index=sequence_index,
        audio=audio,
        sample_rate=target_sr,
        pad_seconds=args.pad_seconds,
        ridge_rows=ridge_rows,
    )


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VariantResult:
    name: str
    result: NotesV3Result
    wall_seconds: float


def _run_variants(
    resolved: _ResolvedEvent,
    variants: Sequence[str],
) -> list[_VariantResult]:
    out: list[_VariantResult] = []
    event_start_utc = resolved.region_offset_s + float(resolved.event.start_sec)
    for name in variants:
        extractor = EXTRACTORS[name]
        started = time.monotonic()
        result = extractor(
            resolved.audio,
            resolved.sample_rate,
            job_id=resolved.encoder.id,
            event_id=resolved.event.event_id,
            event_start_utc=event_start_utc,
            ridge_sidecar_rows=resolved.ridge_rows,
        )
        elapsed = time.monotonic() - started
        logger.info(
            "variant | %s | notes=%d contours=%d secs=%.3f",
            name,
            len(result.notes),
            len(result.contours),
            elapsed,
        )
        out.append(_VariantResult(name=name, result=result, wall_seconds=elapsed))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_HZ_TICKS = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0)
_PIANO_MIDI_MIN = 12
_PIANO_MIDI_MAX = 120
_BLACK_KEYS = {1, 3, 6, 8, 10}  # offsets within an octave
_F0_COLOR = "#1f77b4"
_HARMONIC_COLOR = "#7e9bbd"


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(hz, 1e-9) / 440.0)


def _ridge_xy_from_rows(
    rows: Sequence[Mapping[str, Any]] | None,
    audio: np.ndarray,
    sample_rate: int,
    pad_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, freqs_hz) for the ridge overlay.

    Uses the persisted sidecar when present; otherwise recomputes the
    ridge in-process with the same defaults the v3/v4 extractors use.
    """
    if rows:
        ts = np.asarray(
            [float(r["frame_time_offset_s"]) for r in rows], dtype=np.float64
        )
        fs = np.asarray(
            [2.0 ** float(r["log_frequency"]) for r in rows], dtype=np.float64
        )
        return ts, fs

    if audio.size == 0:
        return np.zeros(0), np.zeros(0)
    n_fft = 1024
    hop = 512
    if audio.size < n_fft:
        return np.zeros(0), np.zeros(0)
    n_frames = (audio.size - n_fft) // hop + 1
    window = np.hanning(n_fft).astype(np.float64)
    spectra = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_frames):
        chunk = audio[i * hop : i * hop + n_fft].astype(np.float64)
        spectra[i, :] = np.abs(np.fft.rfft(chunk * window))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    result = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=sample_rate,
        hop_length=hop,
        min_frequency_hz=30.0,
        max_frequency_hz=6000.0,
        candidate_count=5,
        smoothness_penalty=8.0,
        peak_prominence_ratio=0.0,
    )
    if result.log_frequencies.size == 0:
        return np.zeros(0), np.zeros(0)
    return np.asarray(result.frame_times, dtype=np.float64), np.power(
        2.0, np.asarray(result.log_frequencies, dtype=np.float64)
    )


def _draw_spectrogram(
    ax: matplotlib.axes.Axes,
    audio: np.ndarray,
    sample_rate: int,
    duration_s: float,
    ridge_rows: Sequence[Mapping[str, Any]] | None,
    pad_seconds: float,
) -> None:
    cqt_params = CQTParams()
    cqt_log = compute_event_cqt(audio, sample_rate, params=cqt_params)
    if cqt_log.size == 0:
        ax.text(
            0.5, 0.5, "(empty CQT)", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_xlim(0, duration_s)
        ax.set_ylim(_PIANO_MIDI_MIN, _PIANO_MIDI_MAX)
        return
    n_bins = cqt_log.shape[0]
    bin_log_freqs = np.log2(
        np.asarray(
            [
                cqt_params.fmin * (2.0 ** (i / cqt_params.bins_per_octave))
                for i in range(n_bins)
            ],
            dtype=np.float64,
        )
    )
    hop = cqt_params.hop_length
    seconds_per_frame = float(hop) / float(sample_rate)
    n_frames = cqt_log.shape[1]
    extent = (
        0.0,
        n_frames * seconds_per_frame,
        float(bin_log_freqs[0]),
        float(bin_log_freqs[-1]),
    )
    ax.imshow(
        cqt_log,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=extent,
        interpolation="nearest",
    )
    ridge_t, ridge_hz = _ridge_xy_from_rows(ridge_rows, audio, sample_rate, pad_seconds)
    if ridge_hz.size:
        ax.plot(
            ridge_t,
            np.log2(np.maximum(ridge_hz, 1e-9)),
            color="#00ffd0",
            linewidth=1.0,
            alpha=0.85,
            label="STFT ridge",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, duration_s)
    ax.set_ylim(extent[2], extent[3])

    # Frequency ticks (log Hz axis -> tick label in Hz).
    tick_log = [
        math.log2(f) for f in _HZ_TICKS if extent[2] <= math.log2(f) <= extent[3]
    ]
    tick_lbl = [
        f"{int(2.0**lf) if 2.0**lf >= 1.0 else f'{2.0**lf:.1f}'} Hz" for lf in tick_log
    ]
    ax.set_yticks(tick_log)
    ax.set_yticklabels(tick_lbl)
    ax.set_ylabel("Frequency (log Hz)")
    ax.set_title("CQT spectrogram + STFT ridge", fontsize=10)


def _draw_piano_roll(
    ax: matplotlib.axes.Axes,
    variant: _VariantResult,
    duration_s: float,
) -> None:
    # Background: octave gridlines + black-key shading.
    ax.set_xlim(0, duration_s)
    ax.set_ylim(_PIANO_MIDI_MIN, _PIANO_MIDI_MAX)
    for midi in range(_PIANO_MIDI_MIN, _PIANO_MIDI_MAX + 1):
        if (midi % 12) in _BLACK_KEYS:
            ax.axhspan(midi - 0.5, midi + 0.5, color="#f1f1f1", zorder=0)
        if midi % 12 == 0:
            ax.axhline(midi - 0.5, color="#dadada", linewidth=0.5, zorder=0)
            ax.text(
                duration_s * 1.005,
                midi,
                f"C{midi // 12 - 1}",
                fontsize=7,
                va="center",
                color="#888",
            )

    # Group contour rows by note_uid for ribbon rendering.
    by_uid: dict[str, list[ContourFrame]] = defaultdict(list)
    for c in variant.result.contours:
        by_uid[c.note_uid].append(c)
    for rows in by_uid.values():
        rows.sort(key=lambda r: r.frame_index)

    for note in variant.result.notes:
        rows = by_uid.get(note.note_uid, [])
        if rows:
            _draw_ribbon(ax, note, rows)
        else:
            _draw_flat_bar(ax, note)

    ax.set_ylabel("MIDI pitch")
    label = (
        f"{variant.name}  ({variant.wall_seconds * 1000:.0f} ms, "
        f"{len(variant.result.notes)} notes, {len(variant.result.contours)} contour rows)"
    )
    ax.set_title(label, fontsize=10, loc="left")


def _draw_flat_bar(ax: matplotlib.axes.Axes, note: NoteV3) -> None:
    color = _F0_COLOR if note.partial_index == 0 else _HARMONIC_COLOR
    alpha = 0.85 if note.partial_index == 0 else 0.45
    rect = mpatches.Rectangle(
        (max(0.0, note.start_offset_s), note.midi_pitch - 0.4),
        max(note.duration_s, 1e-3),
        0.8,
        color=color,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(rect)


def _draw_ribbon(
    ax: matplotlib.axes.Axes,
    note: NoteV3,
    rows: Sequence[ContourFrame],
) -> None:
    color = _F0_COLOR if note.partial_index == 0 else _HARMONIC_COLOR
    line_alpha = 0.95 if note.partial_index == 0 else 0.55
    body_alpha = 0.25 if note.partial_index == 0 else 0.15
    width = 2.4 if note.partial_index == 0 else 1.4
    xs = np.asarray(
        [max(0.0, note.start_offset_s + r.time_offset_s) for r in rows],
        dtype=np.float64,
    )
    pitch = np.asarray(
        [note.midi_pitch + r.cents_from_pitch / 100.0 for r in rows],
        dtype=np.float64,
    )
    ax.plot(xs, pitch, color=color, linewidth=width, alpha=line_alpha, zorder=3)
    ax.fill_between(
        xs,
        pitch - 0.4,
        pitch + 0.4,
        color=color,
        alpha=body_alpha,
        zorder=2,
    )


def _render_png(
    resolved: _ResolvedEvent,
    variants: Sequence[_VariantResult],
    args: _CliArgs,
) -> Path:
    duration_s = float(resolved.audio.size) / float(resolved.sample_rate)
    n_panels = 1 + len(variants)
    height_per_panel = max(args.height / max(n_panels, 1), 200)
    fig_height = (height_per_panel * n_panels) / 100.0
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(args.width / 100.0, fig_height),
        sharex=True,
        gridspec_kw={"hspace": 0.25},
    )
    if n_panels == 1:
        axes = [axes]
    _draw_spectrogram(
        axes[0],
        resolved.audio,
        resolved.sample_rate,
        duration_s,
        resolved.ridge_rows,
        resolved.pad_seconds,
    )
    for ax, variant in zip(axes[1:], variants):
        _draw_piano_roll(ax, variant, duration_s)
    axes[-1].set_xlabel("Event time (s, since padded start)")
    token_label = (
        f"token={resolved.sequence_index}"
        if resolved.sequence_index is not None
        else "token=?"
    )
    fig.suptitle(
        f"job={resolved.encoder.id}  {token_label}  "
        f"event={resolved.event.event_id}  duration={duration_s * 1000:.0f} ms",
        fontsize=11,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    plt.close(fig)
    return args.out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _async_main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    # load_dotenv runs only after a successful parse — arg-validation
    # errors raise SystemExit out of _parse_args without polluting
    # ``os.environ`` (matters when this CLI is exercised from pytest).
    load_dotenv()
    settings = Settings()
    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            resolved = await _resolve(session, settings, args)
        variants = _run_variants(resolved, args.variants)
        out_path = _render_png(resolved, variants, args)
        print(f"wrote {out_path}", file=sys.stdout)
        return 0
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return asyncio.run(_async_main(list(argv) if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
