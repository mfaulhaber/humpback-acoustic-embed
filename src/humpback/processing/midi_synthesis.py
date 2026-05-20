"""Deterministic Standard MIDI File synthesis from Piano Roll Notes.

The Piano Roll Notes worker writes a parquet sidecar with one row per detected
MIDI note (pitch, velocity, start, duration). This module converts that
parquet (loaded as a ``pyarrow.Table``) into the bytes of a Standard MIDI File
(SMF Type 1) so callers can persist it as a ``.mid`` artifact or stream it to
a client.

The output is deterministic: identical input rows produce byte-identical
output. The encoded file uses 480 ticks per quarter, a constant 120 BPM tempo
written once at tick 0, and a single notes track on MIDI channel 1.

Pitch-bend support is out of scope for this version. See
docs/specs/2026-05-20-piano-roll-midi-export-design.md §15 for the planned
MPE-based future extension.
"""

from __future__ import annotations

import io
import math
from typing import Iterable

import mido
import pyarrow as pa

__all__ = [
    "TICKS_PER_QUARTER",
    "TEMPO_BPM",
    "MIDI_CHANNEL",
    "notes_table_to_midi_bytes",
]


TICKS_PER_QUARTER = 480
TEMPO_BPM = 120.0
MIDI_CHANNEL = 0  # mido is 0-indexed; "channel 1" in MIDI parlance


_REQUIRED_COLUMNS = (
    "event_id",
    "partial_index",
    "midi_pitch",
    "start_utc",
    "duration_s",
    "velocity",
)


def notes_table_to_midi_bytes(notes_table: pa.Table) -> bytes:
    """Synthesize a Standard MIDI File from a Piano Roll Notes pyarrow Table.

    The Table is expected to follow the columns of
    ``humpback.workers.piano_roll_notes_worker.NOTES_SCHEMA``. Only the columns
    listed in ``_REQUIRED_COLUMNS`` are read; extras are ignored.

    Returns the bytes of an SMF Type 1 file. Empty input yields a valid SMF
    with header + tempo track only (no notes).
    """

    for column in _REQUIRED_COLUMNS:
        if column not in notes_table.column_names:
            raise ValueError(f"notes table missing required column: {column}")

    midi_file = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_QUARTER)
    midi_file.tracks.append(_build_tempo_track())
    midi_file.tracks.append(_build_notes_track(notes_table))

    buf = io.BytesIO()
    midi_file.save(file=buf)
    return buf.getvalue()


def _build_tempo_track() -> mido.MidiTrack:
    track = mido.MidiTrack()
    microseconds_per_quarter = int(round(60_000_000.0 / TEMPO_BPM))
    track.append(mido.MetaMessage("set_tempo", tempo=microseconds_per_quarter, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _build_notes_track(notes_table: pa.Table) -> mido.MidiTrack:
    track = mido.MidiTrack()

    valid_rows = list(_iter_valid_rows(notes_table))
    if not valid_rows:
        track.append(mido.MetaMessage("end_of_track", time=0))
        return track

    valid_rows.sort(key=lambda r: (r["start_utc"], r["event_id"], r["partial_index"]))
    time_origin = valid_rows[0]["start_utc"]

    events: list[tuple[int, tuple[int, int], mido.Message]] = []
    # Tie-breakers (the middle tuple element) make ordering deterministic when
    # absolute ticks collide: note_off events fire before note_on at the same
    # tick to avoid hanging notes when one note ends exactly as another begins.
    for index, row in enumerate(valid_rows):
        pitch = _clamp_pitch(row["midi_pitch"])
        velocity = _clamp_velocity(row["velocity"])
        on_tick = _seconds_to_ticks(row["start_utc"] - time_origin)
        off_tick = on_tick + _seconds_to_ticks(row["duration_s"])
        events.append(
            (
                off_tick,
                (0, index),
                mido.Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    channel=MIDI_CHANNEL,
                    time=0,
                ),
            )
        )
        events.append(
            (
                on_tick,
                (1, index),
                mido.Message(
                    "note_on",
                    note=pitch,
                    velocity=velocity,
                    channel=MIDI_CHANNEL,
                    time=0,
                ),
            )
        )

    events.sort(key=lambda triple: (triple[0], triple[1]))

    previous_tick = 0
    for absolute_tick, _, message in events:
        delta = absolute_tick - previous_tick
        message.time = delta
        track.append(message)
        previous_tick = absolute_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _iter_valid_rows(notes_table: pa.Table) -> Iterable[dict]:
    event_ids = notes_table.column("event_id").to_pylist()
    partials = notes_table.column("partial_index").to_pylist()
    pitches = notes_table.column("midi_pitch").to_pylist()
    starts = notes_table.column("start_utc").to_pylist()
    durations = notes_table.column("duration_s").to_pylist()
    velocities = notes_table.column("velocity").to_pylist()

    for event_id, partial, pitch, start, duration, velocity in zip(
        event_ids, partials, pitches, starts, durations, velocities
    ):
        try:
            duration_f = float(duration)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(duration_f) or duration_f <= 0.0:
            continue
        try:
            start_f = float(start)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start_f):
            continue
        yield {
            "event_id": str(event_id),
            "partial_index": int(partial),
            "midi_pitch": int(pitch),
            "start_utc": start_f,
            "duration_s": duration_f,
            "velocity": int(velocity),
        }


def _seconds_to_ticks(seconds: float) -> int:
    # 120 BPM → 2 beats per second → ticks_per_second = TICKS_PER_QUARTER * 2.
    ticks_per_second = TICKS_PER_QUARTER * (TEMPO_BPM / 60.0)
    return int(round(max(seconds, 0.0) * ticks_per_second))


def _clamp_pitch(pitch: int) -> int:
    if pitch < 0:
        return 0
    if pitch > 127:
        return 127
    return pitch


def _clamp_velocity(velocity: int) -> int:
    if velocity < 1:
        return 1
    if velocity > 127:
        return 127
    return velocity
