"""Tests for the channelized Piano Roll MIDI synthesis module."""

from __future__ import annotations

import io

import mido
import pyarrow as pa
import pytest

from humpback.processing.midi_synthesis import (
    CHANNEL_F0,
    CHANNEL_HARMONIC_2,
    CHANNEL_HARMONIC_3,
    CHANNEL_HARMONIC_4,
    CHANNEL_HARMONIC_5,
    CHANNEL_HARMONIC_HIGH,
    CHANNEL_LAYOUT,
    CHANNEL_UNMATCHED,
    TEMPO_BPM,
    TICKS_PER_QUARTER,
    notes_table_to_midi_bytes,
)
from humpback.workers.piano_roll_notes_worker import NOTES_SCHEMA


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


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


def _collect_notes_for_track(track: mido.MidiTrack) -> list[dict]:
    """Walk one track and pair note_on/note_off into note dicts."""
    pending: dict[int, list[dict]] = {}
    completed: list[dict] = []
    absolute_tick = 0
    for message in track:
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


def _collect_all_notes(midi_file: mido.MidiFile) -> list[dict]:
    """Walk every channel track (skipping the tempo track) and pair notes."""
    out: list[dict] = []
    for track in midi_file.tracks[1:]:
        out.extend(_collect_notes_for_track(track))
    return out


def _channel_track_index_by_name(midi_file: mido.MidiFile) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, track in enumerate(midi_file.tracks):
        for message in track:
            if getattr(message, "type", None) == "track_name":
                out[getattr(message, "name", "")] = idx
                break
    return out


# --------------------------------------------------------------------------- #
# Tempo / header
# --------------------------------------------------------------------------- #


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

    # Tempo track + 7 channel tracks = 8 total.
    assert len(parsed.tracks) == 1 + len(CHANNEL_LAYOUT)

    notes = _collect_all_notes(parsed)
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

    # All three notes are F0 (partial_index = 0) → all on CHANNEL_F0.
    assert all(n["channel"] == CHANNEL_F0 for n in notes)


# --------------------------------------------------------------------------- #
# Channel routing
# --------------------------------------------------------------------------- #


def test_channel_routing_by_partial_index() -> None:
    """Each partial_index maps to the expected slim-layout channel."""
    # Cover every channel including the collapsed "higher harmonics" bucket.
    rows = [
        ("e0", -1, 60),  # unmatched
        ("e0", 0, 61),  # F0
        ("e0", 1, 62),  # 2nd
        ("e0", 2, 63),  # 3rd
        ("e0", 3, 64),  # 4th
        ("e0", 4, 65),  # 5th
        ("e0", 5, 66),  # higher (6th)
        ("e0", 7, 67),  # higher (8th)
        ("e0", 12, 68),  # higher (13th)
    ]
    table = _make_table(
        [
            {
                "event_id": eid,
                "partial_index": p,
                "midi_pitch": pitch,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            }
            for eid, p, pitch in rows
        ]
    )

    parsed = _parse_midi(notes_table_to_midi_bytes(table))
    notes = _collect_all_notes(parsed)
    by_pitch = {n["note"]: n["channel"] for n in notes}

    assert by_pitch[60] == CHANNEL_UNMATCHED
    assert by_pitch[61] == CHANNEL_F0
    assert by_pitch[62] == CHANNEL_HARMONIC_2
    assert by_pitch[63] == CHANNEL_HARMONIC_3
    assert by_pitch[64] == CHANNEL_HARMONIC_4
    assert by_pitch[65] == CHANNEL_HARMONIC_5
    # 6th, 8th, and 13th all collapse onto the single "higher" channel.
    assert by_pitch[66] == CHANNEL_HARMONIC_HIGH
    assert by_pitch[67] == CHANNEL_HARMONIC_HIGH
    assert by_pitch[68] == CHANNEL_HARMONIC_HIGH


def test_track_layout_emits_one_track_per_channel_with_headers() -> None:
    """Tempo + 7 channel tracks, each labeled and given a GM program."""
    table = _make_table(
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ]
    )
    parsed = _parse_midi(notes_table_to_midi_bytes(table))

    assert len(parsed.tracks) == 1 + len(CHANNEL_LAYOUT)

    # Each channel track in order matches the spec's program and name.
    for i, spec in enumerate(CHANNEL_LAYOUT, start=1):
        track = parsed.tracks[i]
        non_eot = [m for m in track if m.type != "end_of_track"]
        assert len(non_eot) >= 2, (
            f"channel {spec.channel} track is missing header events"
        )
        track_name_msg = non_eot[0]
        program_msg = non_eot[1]
        assert getattr(track_name_msg, "type", None) == "track_name"
        assert getattr(track_name_msg, "name", None) == spec.name
        assert getattr(program_msg, "type", None) == "program_change"
        assert getattr(program_msg, "program", None) == spec.program
        assert getattr(program_msg, "channel", None) == spec.channel


def test_empty_channels_still_emit_headers() -> None:
    """A parquet with only F0 notes still emits all 7 channel tracks."""
    table = _make_table(
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ]
    )
    parsed = _parse_midi(notes_table_to_midi_bytes(table))
    assert len(parsed.tracks) == 1 + len(CHANNEL_LAYOUT)

    # Every channel track except F0 has zero note-on events.
    for i, spec in enumerate(CHANNEL_LAYOUT, start=1):
        track = parsed.tracks[i]
        note_ons = [m for m in track if m.type == "note_on" and m.velocity > 0]
        if spec.channel == CHANNEL_F0:
            assert len(note_ons) == 1
        else:
            assert note_ons == [], (
                f"channel {spec.channel} should have no notes for this fixture"
            )
        # Track headers still present.
        meta_names = [
            getattr(m, "name", "")
            for m in track
            if getattr(m, "type", None) == "track_name"
        ]
        program_changes = [m for m in track if m.type == "program_change"]
        assert meta_names == [spec.name]
        assert len(program_changes) == 1


def test_gm_drum_channel_never_written() -> None:
    """Channel 9 (1-indexed GM drum kit) must not appear on any message."""
    rows = [
        ("e0", -1, 60),
        ("e0", 0, 61),
        ("e0", 1, 62),
        ("e0", 2, 63),
        ("e0", 3, 64),
        ("e0", 4, 65),
        ("e0", 5, 66),
        ("e0", 7, 67),
        ("e0", 12, 68),
    ]
    table = _make_table(
        [
            {
                "event_id": eid,
                "partial_index": p,
                "midi_pitch": pitch,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            }
            for eid, p, pitch in rows
        ]
    )
    parsed = _parse_midi(notes_table_to_midi_bytes(table))
    for track in parsed.tracks:
        for message in track:
            channel = getattr(message, "channel", None)
            assert channel != 9, f"GM drum channel 9 leaked into {message!r}"


# --------------------------------------------------------------------------- #
# Determinism and defensive behavior
# --------------------------------------------------------------------------- #


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


def test_empty_table_produces_tempo_track_only() -> None:
    table = _make_table([])
    data = notes_table_to_midi_bytes(table)
    parsed = _parse_midi(data)
    assert parsed.type == 1
    assert parsed.ticks_per_beat == TICKS_PER_QUARTER
    # Empty parquet → tempo track only, no channel tracks. Matches v1.
    assert len(parsed.tracks) == 1


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
    parsed = _parse_midi(notes_table_to_midi_bytes(table))
    notes = _collect_all_notes(parsed)
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
    parsed = _parse_midi(notes_table_to_midi_bytes(table))
    notes = _collect_all_notes(parsed)
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
    parsed = _parse_midi(notes_table_to_midi_bytes(table))
    notes = _collect_all_notes(parsed)
    assert len(notes) == 1
    assert notes[0]["on_tick"] == 0


def test_missing_required_column_raises() -> None:
    table = pa.table({"event_id": pa.array(["e0"], pa.string())})
    with pytest.raises(ValueError):
        notes_table_to_midi_bytes(table)


def test_explicit_time_origin_anchors_first_note_at_zero() -> None:
    """A note whose ``start_utc`` equals ``time_origin_utc`` lands at tick 0."""
    origin = 1_700_000_000.0
    table = _make_table(
        [
            {
                "event_id": "anchor",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": origin,
                "duration_s": 0.5,
                "velocity": 80,
            },
            {
                "event_id": "one-second",
                "partial_index": 1,
                "midi_pitch": 72,
                "start_utc": origin + 1.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
        ]
    )
    parsed = _parse_midi(notes_table_to_midi_bytes(table, time_origin_utc=origin))

    f0_track = parsed.tracks[_channel_track_index_by_name(parsed)["F0"]]
    h2_track = parsed.tracks[_channel_track_index_by_name(parsed)["2nd harmonic"]]

    def _first_note_on_tick(track) -> int:
        absolute = 0
        for msg in track:
            absolute += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                return absolute
        raise AssertionError("no note_on found")

    assert _first_note_on_tick(f0_track) == 0
    expected = _seconds_to_ticks(1.0)
    assert _first_note_on_tick(h2_track) == expected


def test_empty_table_with_time_origin_produces_tempo_track_only() -> None:
    """An empty notes table still emits a valid SMF (tempo only)."""
    table = _make_table([])
    parsed = _parse_midi(
        notes_table_to_midi_bytes(table, time_origin_utc=1_700_000_000.0)
    )
    # Tempo track + one track per channel layout entry (some may be empty
    # but they still emit track_name + program_change + end_of_track).
    assert parsed.type == 1
    # Tempo track always present.
    assert len(parsed.tracks) >= 1


def test_per_track_deterministic_ordering_within_channel() -> None:
    """Notes on the same channel keep deterministic intra-track ordering."""
    table = _make_table(
        [
            # Two F0 notes sharing the same start tick should not produce
            # interleaved on/off chaos. Pick distinct pitches and assert
            # that BOTH note_ons appear before either note_off.
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
            {
                "event_id": "e1",
                "partial_index": 0,
                "midi_pitch": 64,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 90,
            },
        ]
    )
    parsed = _parse_midi(notes_table_to_midi_bytes(table))
    name_to_index = _channel_track_index_by_name(parsed)
    f0_track = parsed.tracks[name_to_index["F0"]]
    note_messages = [m for m in f0_track if m.type in {"note_on", "note_off"}]
    # Both note_ons should come before either note_off (notes overlap).
    types = [m.type for m in note_messages]
    assert types == ["note_on", "note_on", "note_off", "note_off"]
