"""Deterministic Standard MIDI File synthesis from Piano Roll Notes.

The Piano Roll Notes worker writes a parquet sidecar with one row per detected
MIDI note (pitch, velocity, start, duration, plus a ``partial_index`` carrying
the v2 harmonic label). This module converts that parquet (loaded as a
``pyarrow.Table``) into the bytes of a Standard MIDI File (SMF Type 1) so
callers can persist it as a ``.mid`` artifact or stream it to a client.

The output is deterministic: identical input rows produce byte-identical
output. The encoded file uses 480 ticks per quarter, a constant 120 BPM
tempo written once at tick 0, and a seven-channel slim layout — one SMF
track per channel — that routes the fundamental, the 2nd–5th harmonics,
all higher harmonics combined, and the unmatched bucket onto separate
General MIDI channels. The GM drum channel (MIDI channel 10, 0-indexed 9)
is intentionally left empty so no pitched humpback content is re-mapped
to drums by GM-compliant playback engines.

Pitch-bend support is out of scope for this version. See
docs/specs/2026-05-20-event-encoder-midi-channelized-design.md §6 and
docs/specs/2026-05-20-piano-roll-midi-export-design.md §15 for the planned
MPE-based future extension.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable

import mido
import pyarrow as pa

__all__ = [
    "TICKS_PER_QUARTER",
    "TEMPO_BPM",
    "ChannelSpec",
    "CHANNEL_LAYOUT",
    "CHANNEL_F0",
    "CHANNEL_HARMONIC_2",
    "CHANNEL_HARMONIC_3",
    "CHANNEL_HARMONIC_4",
    "CHANNEL_HARMONIC_5",
    "CHANNEL_HARMONIC_HIGH",
    "CHANNEL_UNMATCHED",
    "notes_table_to_midi_bytes",
]


TICKS_PER_QUARTER = 480
TEMPO_BPM = 120.0

# 0-indexed MIDI channels. Channel 9 (1-indexed channel 10) is the GM
# percussion channel and is intentionally absent from the layout below.
CHANNEL_F0 = 0
CHANNEL_HARMONIC_2 = 1
CHANNEL_HARMONIC_3 = 2
CHANNEL_HARMONIC_4 = 3
CHANNEL_HARMONIC_5 = 4
CHANNEL_HARMONIC_HIGH = 5
CHANNEL_UNMATCHED = 6


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """One row of the slim channel layout: channel id, GM patch, display name."""

    channel: int
    program: int
    name: str


# The slim 7-channel layout. Order is preserved as track order in the SMF
# output (after the tempo track), so DAWs render the partials top-to-bottom
# the way they're listed here.
CHANNEL_LAYOUT: tuple[ChannelSpec, ...] = (
    ChannelSpec(channel=CHANNEL_F0, program=0, name="F0"),
    ChannelSpec(channel=CHANNEL_HARMONIC_2, program=11, name="2nd harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_3, program=12, name="3rd harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_4, program=10, name="4th harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_5, program=8, name="5th harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_HIGH, program=88, name="higher harmonics"),
    ChannelSpec(channel=CHANNEL_UNMATCHED, program=90, name="unmatched"),
)


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
    with header + tempo track only (no channel tracks).
    """

    for column in _REQUIRED_COLUMNS:
        if column not in notes_table.column_names:
            raise ValueError(f"notes table missing required column: {column}")

    midi_file = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_QUARTER)
    midi_file.tracks.append(_build_tempo_track())

    valid_rows = list(_iter_valid_rows(notes_table))
    if valid_rows:
        for track in _build_channel_tracks(valid_rows):
            midi_file.tracks.append(track)

    buf = io.BytesIO()
    midi_file.save(file=buf)
    return buf.getvalue()


def _build_tempo_track() -> mido.MidiTrack:
    track = mido.MidiTrack()
    microseconds_per_quarter = int(round(60_000_000.0 / TEMPO_BPM))
    track.append(mido.MetaMessage("set_tempo", tempo=microseconds_per_quarter, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _build_channel_tracks(valid_rows: list[dict]) -> list[mido.MidiTrack]:
    """Build one ``MidiTrack`` per ``CHANNEL_LAYOUT`` entry.

    Each track starts at tick 0 with a ``track_name`` meta-event followed
    by a ``program_change`` selecting that channel's GM patch. Notes then
    follow with delta-time encoded note-on / note-off pairs. Tracks with
    no notes still emit ``track_name`` + ``program_change`` + ``end_of_track``
    so the SMF's track layout is structurally identical across jobs —
    DAW project templates and routing configs port between exports.
    """
    valid_rows.sort(key=lambda r: (r["start_utc"], r["event_id"], r["partial_index"]))
    time_origin = valid_rows[0]["start_utc"]

    # Bucket rows by channel via the partial-index mapping.
    rows_by_channel: dict[int, list[tuple[int, dict]]] = {
        spec.channel: [] for spec in CHANNEL_LAYOUT
    }
    for index, row in enumerate(valid_rows):
        channel = _channel_for_partial(row["partial_index"])
        rows_by_channel[channel].append((index, row))

    tracks: list[mido.MidiTrack] = []
    for spec in CHANNEL_LAYOUT:
        tracks.append(
            _build_channel_track(
                spec=spec,
                rows=rows_by_channel[spec.channel],
                time_origin=time_origin,
            )
        )
    return tracks


def _build_channel_track(
    *,
    spec: ChannelSpec,
    rows: list[tuple[int, dict]],
    time_origin: float,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=spec.name, time=0))
    track.append(
        mido.Message(
            "program_change", program=spec.program, channel=spec.channel, time=0
        )
    )

    # Tie-breakers (the middle tuple element) make ordering deterministic when
    # absolute ticks collide: note_off events fire before note_on at the same
    # tick to avoid hanging notes when one note ends exactly as another begins.
    events: list[tuple[int, tuple[int, int], mido.Message]] = []
    for index, row in rows:
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
                    channel=spec.channel,
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
                    channel=spec.channel,
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


def _channel_for_partial(partial_index: int) -> int:
    """Map a parquet ``partial_index`` to one of the slim layout channels.

    ``-1`` (unmatched) and the F0 (``0``) each have their own dedicated
    channels. Harmonics 2..5 (``partial_index`` 1..4) each get their own
    channel. Harmonics 6 and above (``partial_index >= 5``) collapse onto
    the single ``CHANNEL_HARMONIC_HIGH``. The GM drum channel is never
    returned.

    Defensive: any negative ``partial_index`` (``-1`` or otherwise
    unexpected) routes to ``CHANNEL_UNMATCHED`` rather than silently
    falling through to the high-harmonics bucket.
    """
    if partial_index < 0:
        return CHANNEL_UNMATCHED
    if partial_index == 0:
        return CHANNEL_F0
    if 1 <= partial_index <= 4:
        return CHANNEL_HARMONIC_2 + (partial_index - 1)
    return CHANNEL_HARMONIC_HIGH


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
