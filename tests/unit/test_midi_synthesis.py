"""Tests for the Piano Roll MIDI synthesis module."""

from __future__ import annotations

import io

import mido
import pyarrow as pa
import pytest

from humpback.processing.midi_synthesis import (
    MIDI_CHANNEL,
    TEMPO_BPM,
    TICKS_PER_QUARTER,
    notes_table_to_midi_bytes,
)
from humpback.workers.piano_roll_notes_worker import NOTES_SCHEMA


def _ticks_per_second() -> float:
    return TICKS_PER_QUARTER * (TEMPO_BPM / 60.0)


def _seconds_to_ticks(seconds: float) -> int:
    return int(round(seconds * _ticks_per_second()))


def _make_table(rows: list[dict]) -> pa.Table:
    defaults = {
        "event_token": 0,
        "start_offset_s": 0.0,
        "peak_magnitude": 0.0,
        "track_id": 0,
    }
    expanded = [
        {
            field.name: row.get(field.name, defaults.get(field.name))
            for field in NOTES_SCHEMA
        }
        for row in rows
    ]
    return pa.Table.from_pylist(expanded, schema=NOTES_SCHEMA)


def _parse_midi(data: bytes) -> mido.MidiFile:
    return mido.MidiFile(file=io.BytesIO(data))


def _collect_notes(midi_file: mido.MidiFile) -> list[dict]:
    """Walk the notes track and pair note_on/note_off into note dicts."""
    notes_track = midi_file.tracks[1]
    pending: dict[int, list[dict]] = {}
    completed: list[dict] = []
    absolute_tick = 0
    for message in notes_track:
        absolute_tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            pending.setdefault(message.note, []).append(
                {
                    "note": message.note,
                    "velocity": message.velocity,
                    "channel": message.channel,
                    "on_tick": absolute_tick,
                }
            )
        elif message.type in {"note_off", "note_on"} and (
            message.type == "note_off" or message.velocity == 0
        ):
            queue = pending.get(message.note, [])
            if queue:
                started = queue.pop(0)
                started["off_tick"] = absolute_tick
                started["duration_ticks"] = absolute_tick - started["on_tick"]
                completed.append(started)
    return completed


def test_three_note_round_trip() -> None:
    table = _make_table(
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 1_000.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 64,
                "start_utc": 1_000.5,
                "duration_s": 0.25,
                "velocity": 90,
            },
            {
                "event_id": "e1",
                "partial_index": 0,
                "midi_pitch": 67,
                "start_utc": 1_001.0,
                "duration_s": 1.0,
                "velocity": 100,
            },
        ]
    )

    data = notes_table_to_midi_bytes(table)
    parsed = _parse_midi(data)

    assert parsed.type == 1
    assert parsed.ticks_per_beat == TICKS_PER_QUARTER
    notes = _collect_notes(parsed)
    assert len(notes) == 3

    pitches = sorted(n["note"] for n in notes)
    assert pitches == [60, 64, 67]
    by_pitch = {n["note"]: n for n in notes}
    assert by_pitch[60]["velocity"] == 80
    assert by_pitch[64]["velocity"] == 90
    assert by_pitch[67]["velocity"] == 100

    assert by_pitch[60]["on_tick"] == 0
    assert abs(by_pitch[64]["on_tick"] - _seconds_to_ticks(0.5)) <= 1
    assert abs(by_pitch[67]["on_tick"] - _seconds_to_ticks(1.0)) <= 1

    assert abs(by_pitch[60]["duration_ticks"] - _seconds_to_ticks(0.5)) <= 1
    assert abs(by_pitch[64]["duration_ticks"] - _seconds_to_ticks(0.25)) <= 1
    assert abs(by_pitch[67]["duration_ticks"] - _seconds_to_ticks(1.0)) <= 1

    assert all(n["channel"] == MIDI_CHANNEL for n in notes)


def test_all_partials_stacked_on_same_channel() -> None:
    table = _make_table(
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 0.0,
                "duration_s": 1.0,
                "velocity": 80,
            },
            {
                "event_id": "e0",
                "partial_index": 1,
                "midi_pitch": 72,
                "start_utc": 0.0,
                "duration_s": 1.0,
                "velocity": 40,
            },
            {
                "event_id": "e0",
                "partial_index": 2,
                "midi_pitch": 79,
                "start_utc": 0.0,
                "duration_s": 1.0,
                "velocity": 20,
            },
        ]
    )

    notes = _collect_notes(_parse_midi(notes_table_to_midi_bytes(table)))
    assert len(notes) == 3
    assert {n["note"] for n in notes} == {60, 72, 79}
    assert {n["channel"] for n in notes} == {MIDI_CHANNEL}
    assert {n["on_tick"] for n in notes} == {0}


def test_deterministic_output() -> None:
    table = _make_table(
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 1.0,
                "duration_s": 0.5,
                "velocity": 64,
            },
            {
                "event_id": "e0",
                "partial_index": 1,
                "midi_pitch": 72,
                "start_utc": 1.0,
                "duration_s": 0.5,
                "velocity": 32,
            },
        ]
    )
    first = notes_table_to_midi_bytes(table)
    second = notes_table_to_midi_bytes(table)
    assert first == second


def test_empty_table_produces_valid_midi() -> None:
    table = _make_table([])
    data = notes_table_to_midi_bytes(table)
    parsed = _parse_midi(data)
    assert parsed.type == 1
    assert parsed.ticks_per_beat == TICKS_PER_QUARTER
    assert len(parsed.tracks) == 2
    notes_track = parsed.tracks[1]
    non_meta = [m for m in notes_track if not isinstance(m, mido.MetaMessage)]
    assert non_meta == []


def test_pitch_clamping() -> None:
    # NOTES_SCHEMA uses uint8 so we can't load 200 or -10 directly. Build a
    # synthesized table that omits the schema constraint for this case.
    table = pa.table(
        {
            "event_id": pa.array(["e0", "e1"], pa.string()),
            "event_token": pa.array([0, 0], pa.int32()),
            "partial_index": pa.array([0, 0], pa.int32()),
            "midi_pitch": pa.array([200, -10], pa.int32()),
            "start_utc": pa.array([0.0, 0.0], pa.float64()),
            "start_offset_s": pa.array([0.0, 0.0], pa.float64()),
            "duration_s": pa.array([0.5, 0.5], pa.float64()),
            "velocity": pa.array([80, 80], pa.int32()),
            "peak_magnitude": pa.array([0.0, 0.0], pa.float32()),
            "track_id": pa.array([0, 0], pa.uint32()),
        }
    )
    notes = _collect_notes(_parse_midi(notes_table_to_midi_bytes(table)))
    pitches = sorted(n["note"] for n in notes)
    assert pitches == [0, 127]


def test_zero_duration_notes_dropped() -> None:
    table = _make_table(
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 0.0,
                "duration_s": 0.0,
                "velocity": 80,
            },
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 64,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
        ]
    )
    notes = _collect_notes(_parse_midi(notes_table_to_midi_bytes(table)))
    assert len(notes) == 1
    assert notes[0]["note"] == 64


def test_time_origin_shift() -> None:
    table = _make_table(
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 1_700_000_000.0,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ]
    )
    notes = _collect_notes(_parse_midi(notes_table_to_midi_bytes(table)))
    assert len(notes) == 1
    assert notes[0]["on_tick"] == 0


def test_missing_required_column_raises() -> None:
    table = pa.table({"event_id": pa.array(["e0"], pa.string())})
    with pytest.raises(ValueError):
        notes_table_to_midi_bytes(table)
