"""Deterministic Standard MIDI File synthesis from Piano Roll Notes.

The Piano Roll Notes worker writes a parquet sidecar with one row per detected
MIDI note. Two on-disk shapes coexist:

- **v2** rows (no ``note_uid`` column): one row per note with a
  ``partial_index`` carrying the v2 harmonic label. The synthesizer emits
  the legacy slim seven-channel layout (F0, H2–H5, higher harmonics,
  unmatched). No pitch bend.
- **v3** rows (``note_uid`` column present): one row per note plus an
  aligned per-frame contour sidecar with sub-semitone ``cents_from_pitch``
  values. The synthesizer emits an **MPE Lower Zone** SMF: one Master
  track configuring 15 member channels, one track per member channel
  carrying ``program_change``, ``CC 74``, ``note_on``, a bend-decimated
  ``pitch_bend`` stream, and ``note_off``. Channel allocation is
  deterministic on ``(start_utc, note_uid)`` with longest-idle pick and
  FIFO voice steal. Harmonic notes reuse their parent F0's bend stream in
  cents (cents conservation per ADR-069 §5.4).

Both paths are deterministic: identical inputs produce byte-identical
output.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable, Optional

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
    "MPE_BEND_RANGE_SEMITONES",
    "MPE_MEMBER_CHANNELS",
    "MPE_DEFAULT_QUANTIZE_CENTS",
    "MPE_PARTIAL_PROGRAMS",
    "MPE_HIGH_HARMONIC_PROGRAM",
    "notes_table_to_midi_bytes",
]


TICKS_PER_QUARTER = 480
TEMPO_BPM = 120.0

# v2 channel layout (legacy slim 7-channel).
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


CHANNEL_LAYOUT: tuple[ChannelSpec, ...] = (
    ChannelSpec(channel=CHANNEL_F0, program=0, name="F0"),
    ChannelSpec(channel=CHANNEL_HARMONIC_2, program=11, name="2nd harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_3, program=12, name="3rd harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_4, program=10, name="4th harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_5, program=8, name="5th harmonic"),
    ChannelSpec(channel=CHANNEL_HARMONIC_HIGH, program=88, name="higher harmonics"),
    ChannelSpec(channel=CHANNEL_UNMATCHED, program=90, name="unmatched"),
)

# MPE Lower Zone configuration (ADR-069 / spec §8). Channel 0 is the
# Master, channels 1..15 are the member channels carrying voices.
MPE_MEMBER_CHANNELS = tuple(range(1, 16))
MPE_BEND_RANGE_SEMITONES = 24
MPE_BEND_RANGE_CENTS = MPE_BEND_RANGE_SEMITONES * 100.0
MPE_DEFAULT_QUANTIZE_CENTS = 4.0

MPE_PARTIAL_PROGRAMS: dict[int, int] = {
    0: 0,  # F0 → Acoustic Grand Piano
    1: 11,  # H2 → Vibraphone
    2: 12,  # H3 → Marimba
    3: 10,  # H4 → Music Box
    4: 8,  # H5 → Celesta
}
MPE_HIGH_HARMONIC_PROGRAM = 88  # H6..H16 → New Age Pad


_REQUIRED_COLUMNS = (
    "event_id",
    "partial_index",
    "midi_pitch",
    "start_utc",
    "duration_s",
    "velocity",
)


def notes_table_to_midi_bytes(
    notes_table: pa.Table,
    *,
    contour_table: Optional[pa.Table] = None,
    time_origin_utc: float | None = None,
) -> bytes:
    """Synthesize a Standard MIDI File from a Piano Roll Notes pyarrow Table.

    The synthesizer detects the v3 MPE shape by the presence of a
    ``note_uid`` column on ``notes_table``. When v3 is detected,
    ``contour_table`` must follow the schema in
    ``humpback.workers.piano_roll_notes_worker.NOTE_CONTOURS_V3_SCHEMA`` —
    one row per frame per note with a ``cents_from_pitch`` value driving
    the per-voice pitch bend stream. For v2 input the function falls back
    to the existing slim 7-channel layout (regression guard for ADR-066 /
    ADR-068 callers that haven't migrated).

    Args:
        notes_table: Per-note rows. The v2 / v3 shape is auto-detected by
            the presence of ``note_uid``.
        contour_table: Per-frame contour rows. Required when ``notes_table``
            is v3-shaped; ignored on v2 input.
        time_origin_utc: Optional UTC epoch seconds anchored to tick 0.
            When ``None`` the earliest valid note's ``start_utc`` is used.

    Returns the bytes of an SMF Type 1 file.
    """
    for column in _REQUIRED_COLUMNS:
        if column not in notes_table.column_names:
            raise ValueError(f"notes table missing required column: {column}")

    is_v3 = "note_uid" in notes_table.column_names
    if is_v3 and contour_table is None:
        raise ValueError("v3 notes table requires a contour_table; none was provided")

    midi_file = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_QUARTER)
    midi_file.tracks.append(_build_tempo_track())

    valid_rows = list(_iter_valid_rows(notes_table, is_v3=is_v3))
    if not valid_rows and time_origin_utc is None:
        buf = io.BytesIO()
        midi_file.save(file=buf)
        return buf.getvalue()

    origin = (
        float(time_origin_utc)
        if time_origin_utc is not None
        else valid_rows[0]["start_utc"]
        if valid_rows
        else 0.0
    )

    if is_v3:
        contours_by_uid = _group_contour_rows(contour_table)
        master_track, channel_tracks = _build_mpe_tracks(
            valid_rows, contours_by_uid=contours_by_uid, time_origin=origin
        )
        midi_file.tracks.append(master_track)
        midi_file.tracks.extend(channel_tracks)
    else:
        for track in _build_channel_tracks(valid_rows, time_origin=origin):
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


# ---------------------------------------------------------------------------
# v2 channelized layout
# ---------------------------------------------------------------------------


def _build_channel_tracks(
    valid_rows: list[dict], *, time_origin: float
) -> list[mido.MidiTrack]:
    valid_rows.sort(key=lambda r: (r["start_utc"], r["event_id"], r["partial_index"]))
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
    if partial_index < 0:
        return CHANNEL_UNMATCHED
    if partial_index == 0:
        return CHANNEL_F0
    if 1 <= partial_index <= 4:
        return CHANNEL_HARMONIC_2 + (partial_index - 1)
    return CHANNEL_HARMONIC_HIGH


# ---------------------------------------------------------------------------
# v3 MPE Lower Zone synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MPEVoice:
    """One v3 note assigned to a member channel."""

    note_uid: str
    event_id: str
    partial_index: int
    midi_pitch: int
    velocity: int
    on_tick: int
    off_tick: int
    channel: int
    contour: list[dict]  # frames sorted by frame_index


def _build_mpe_tracks(
    valid_rows: list[dict],
    *,
    contours_by_uid: dict[str, list[dict]],
    time_origin: float,
) -> tuple[mido.MidiTrack, list[mido.MidiTrack]]:
    """Return (master_track, [per-member tracks])."""
    valid_rows.sort(key=lambda r: (r["start_utc"], r["note_uid"]))
    voices, steal_events = _allocate_channels(
        valid_rows, contours_by_uid=contours_by_uid, time_origin=time_origin
    )

    master_track = _build_mpe_master_track(voices)
    member_tracks = [
        _build_mpe_member_track(channel, voices, steal_events)
        for channel in MPE_MEMBER_CHANNELS
    ]
    return master_track, member_tracks


def _allocate_channels(
    valid_rows: list[dict],
    *,
    contours_by_uid: dict[str, list[dict]],
    time_origin: float,
) -> tuple[list[_MPEVoice], dict[str, int]]:
    """Deterministic longest-idle allocator with FIFO voice steal.

    Returns ``(voices, steal_events)`` where ``steal_events[note_uid]``
    is the tick at which an in-flight note was stolen (and therefore
    requires an explicit ``note_off`` at that tick rather than at its
    natural end). Notes whose voice was stolen still appear in
    ``voices`` with their original ``off_tick``; the per-member track
    builder consults ``steal_events`` to short-circuit them.
    """
    voices: list[_MPEVoice] = []
    sounding: list[_MPEVoice] = []  # currently ringing voices on member channels
    last_release_tick: dict[int, int] = {ch: -(1 << 31) for ch in MPE_MEMBER_CHANNELS}
    steal_events: dict[str, int] = {}

    for row in valid_rows:
        on_tick = _seconds_to_ticks(row["start_utc"] - time_origin)
        off_tick = on_tick + _seconds_to_ticks(row["duration_s"])

        # Retire any voices whose natural off_tick is at or before on_tick.
        still_sounding: list[_MPEVoice] = []
        for voice in sounding:
            if voice.off_tick <= on_tick:
                last_release_tick[voice.channel] = max(
                    last_release_tick[voice.channel], voice.off_tick
                )
            else:
                still_sounding.append(voice)
        sounding = still_sounding

        busy_channels = {v.channel for v in sounding}
        free_channels = [c for c in MPE_MEMBER_CHANNELS if c not in busy_channels]
        if free_channels:
            # Longest-idle: largest tick gap between now and the channel's
            # last release. Ties broken by ascending channel index.
            chosen = max(
                free_channels,
                key=lambda c: (on_tick - last_release_tick[c], -c),
            )
        else:
            # FIFO steal: oldest sounding voice gives up its channel.
            stolen = min(sounding, key=lambda v: (v.on_tick, v.note_uid))
            steal_events[stolen.note_uid] = on_tick
            sounding = [v for v in sounding if v.note_uid != stolen.note_uid]
            chosen = stolen.channel
            last_release_tick[chosen] = on_tick

        contour = contours_by_uid.get(row["note_uid"], [])
        voice = _MPEVoice(
            note_uid=row["note_uid"],
            event_id=row["event_id"],
            partial_index=int(row["partial_index"]),
            midi_pitch=_clamp_pitch(row["midi_pitch"]),
            velocity=_clamp_velocity(row["velocity"]),
            on_tick=on_tick,
            off_tick=off_tick,
            channel=chosen,
            contour=contour,
        )
        voices.append(voice)
        sounding.append(voice)

    return voices, steal_events


def _build_mpe_master_track(voices: list[_MPEVoice]) -> mido.MidiTrack:
    """Build the MPE Master track.

    Tick 0:
    - ``track_name`` "MPE Master".
    - RPN 6 with payload 15 on channel 0 (the Master) declaring 15
      members. Encoded as CC 101=0, CC 100=6, CC 6=15.
    - For each member channel: RPN 0/0 + Data Entry MSB=24 to set the
      per-voice bend range to ±24 semitones. Encoded with the same
      three-CC pattern but on the member channel itself.

    Plus, immediately before each note's ``on_tick``, a
    ``MetaMessage("text", text=f"p{partial_index}")`` so partial identity
    survives a round-trip.
    """
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="MPE Master", time=0))

    # RPN 6 with payload 15 on the Master (channel 0).
    track.extend(_rpn_messages(channel=0, msb=0, lsb=6, value_msb=15, time=0))

    # Per-member ±24 semitone bend range (RPN 0/0 → Data Entry MSB=24).
    for channel in MPE_MEMBER_CHANNELS:
        track.extend(_rpn_messages(channel=channel, msb=0, lsb=0, value_msb=24, time=0))

    # Per-note partial-index text meta events sorted by (on_tick, note_uid).
    sorted_voices = sorted(voices, key=lambda v: (v.on_tick, v.note_uid))
    previous_tick = 0
    for voice in sorted_voices:
        target_tick = max(0, voice.on_tick - 1)
        if target_tick < previous_tick:
            target_tick = previous_tick
        delta = target_tick - previous_tick
        track.append(
            mido.MetaMessage("text", text=f"p{voice.partial_index}", time=delta)
        )
        previous_tick = target_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _build_mpe_member_track(
    channel: int, voices: list[_MPEVoice], steal_events: dict[str, int]
) -> mido.MidiTrack:
    """Emit the program_change / CC 74 / note_on / bend / note_off stream."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=f"Voice {channel}", time=0))

    channel_voices = [v for v in voices if v.channel == channel]
    channel_voices.sort(key=lambda v: (v.on_tick, v.note_uid))

    events: list[tuple[int, tuple[int, int, int], mido.Message]] = []
    for index, voice in enumerate(channel_voices):
        program = _program_for_partial(voice.partial_index)
        cc74_value = _cc74_for_partial(voice.partial_index)
        program_tick = max(0, voice.on_tick - 1)

        # program_change at on_tick - 1 so receivers latch it before note_on.
        events.append(
            (
                program_tick,
                (0, index, 0),
                mido.Message(
                    "program_change",
                    program=program,
                    channel=channel,
                    time=0,
                ),
            )
        )
        events.append(
            (
                program_tick,
                (0, index, 1),
                mido.Message(
                    "control_change",
                    control=74,
                    value=cc74_value,
                    channel=channel,
                    time=0,
                ),
            )
        )

        # Effective off_tick: explicit steal, if any, supersedes the
        # natural end so the channel is free for the new voice.
        effective_off = steal_events.get(voice.note_uid, voice.off_tick)
        if effective_off <= voice.on_tick:
            effective_off = voice.on_tick + 1

        bend_events = _bend_events_for_voice(
            voice,
            channel=channel,
            on_tick=voice.on_tick,
            off_tick=effective_off,
            quantize_cents=MPE_DEFAULT_QUANTIZE_CENTS,
        )
        for bend_tick, bend_value in bend_events:
            events.append(
                (
                    bend_tick,
                    (1, index, bend_tick),
                    mido.Message(
                        "pitchwheel",
                        pitch=bend_value,
                        channel=channel,
                        time=0,
                    ),
                )
            )

        events.append(
            (
                voice.on_tick,
                (2, index, 0),
                mido.Message(
                    "note_on",
                    note=voice.midi_pitch,
                    velocity=voice.velocity,
                    channel=channel,
                    time=0,
                ),
            )
        )
        events.append(
            (
                effective_off,
                (3, index, 0),
                mido.Message(
                    "note_off",
                    note=voice.midi_pitch,
                    velocity=0,
                    channel=channel,
                    time=0,
                ),
            )
        )

    events.sort(key=lambda triple: (triple[0], triple[1]))

    previous_tick = 0
    for absolute_tick, _, message in events:
        delta = absolute_tick - previous_tick
        if delta < 0:
            delta = 0
        message.time = delta
        track.append(message)
        previous_tick = max(previous_tick, absolute_tick)

    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _bend_events_for_voice(
    voice: _MPEVoice,
    *,
    channel: int,
    on_tick: int,
    off_tick: int,
    quantize_cents: float,
) -> list[tuple[int, int]]:
    """Decimate the contour's ``cents_from_pitch`` stream to bend events.

    Always emits a bend at ``on_tick`` and another at ``off_tick`` so the
    voice starts and ends at a known offset. Intermediate frames produce
    a bend only when ``|delta cents| >= quantize_cents``.
    """
    if not voice.contour:
        center = _cents_to_bend(0.0)
        return [(on_tick, center), (off_tick, center)]

    duration_ticks = max(1, off_tick - on_tick)
    frames = sorted(voice.contour, key=lambda f: int(f["frame_index"]))
    frame_count = len(frames)

    events: list[tuple[int, int]] = []
    last_cents = float("nan")
    for i, frame in enumerate(frames):
        cents = float(frame["cents_from_pitch"])
        if i == 0:
            tick = on_tick
        elif i == frame_count - 1:
            tick = off_tick
        else:
            tick = on_tick + int(round(duration_ticks * (i / max(1, frame_count - 1))))
        if i == 0 or i == frame_count - 1 or abs(cents - last_cents) >= quantize_cents:
            events.append((tick, _cents_to_bend(cents)))
            last_cents = cents

    if not events:
        center = _cents_to_bend(0.0)
        events.append((on_tick, center))
    if events[0][0] != on_tick:
        events.insert(0, (on_tick, events[0][1]))
    if events[-1][0] != off_tick:
        events.append((off_tick, events[-1][1]))
    return events


def _rpn_messages(
    *, channel: int, msb: int, lsb: int, value_msb: int, time: int
) -> list[mido.Message]:
    return [
        mido.Message(
            "control_change", control=101, value=msb, channel=channel, time=time
        ),
        mido.Message("control_change", control=100, value=lsb, channel=channel, time=0),
        mido.Message(
            "control_change", control=6, value=value_msb, channel=channel, time=0
        ),
    ]


def _program_for_partial(partial_index: int) -> int:
    if partial_index in MPE_PARTIAL_PROGRAMS:
        return MPE_PARTIAL_PROGRAMS[partial_index]
    return MPE_HIGH_HARMONIC_PROGRAM


def _cc74_for_partial(partial_index: int) -> int:
    value = max(0, int(partial_index)) * 16
    return max(0, min(127, value))


def _cents_to_bend(cents: float) -> int:
    """Convert cents to a signed 14-bit pitch-bend value (mido range).

    ``mido.Message("pitchwheel", pitch=…)`` uses the signed range
    ``[-8192, 8191]`` where ``0`` is the centre. Cents are mapped against
    the configured ±24-semitone bend range so the full ±2400 cents
    excursion saturates at the endpoints rather than wrapping.
    """
    if not math.isfinite(cents):
        return 0
    bend = int(round(cents / MPE_BEND_RANGE_CENTS * 8192.0))
    return max(-8192, min(8191, bend))


def _group_contour_rows(contour_table: pa.Table) -> dict[str, list[dict]]:
    if contour_table is None or contour_table.num_rows == 0:
        return {}
    columns = ("note_uid", "frame_index", "time_offset_s", "cents_from_pitch")
    for column in columns:
        if column not in contour_table.column_names:
            raise ValueError(f"contour table missing required column: {column}")
    note_uids = contour_table.column("note_uid").to_pylist()
    frame_indices = contour_table.column("frame_index").to_pylist()
    times = contour_table.column("time_offset_s").to_pylist()
    cents = contour_table.column("cents_from_pitch").to_pylist()
    by_uid: dict[str, list[dict]] = {}
    for uid, idx, t_off, c in zip(note_uids, frame_indices, times, cents):
        by_uid.setdefault(str(uid), []).append(
            {
                "frame_index": int(idx),
                "time_offset_s": float(t_off),
                "cents_from_pitch": float(c),
            }
        )
    for rows in by_uid.values():
        rows.sort(key=lambda r: r["frame_index"])
    return by_uid


# ---------------------------------------------------------------------------
# Shared row decoding / utilities
# ---------------------------------------------------------------------------


def _iter_valid_rows(notes_table: pa.Table, *, is_v3: bool) -> Iterable[dict]:
    columns = notes_table.column_names
    event_ids = notes_table.column("event_id").to_pylist()
    partials = notes_table.column("partial_index").to_pylist()
    pitches = notes_table.column("midi_pitch").to_pylist()
    starts = notes_table.column("start_utc").to_pylist()
    durations = notes_table.column("duration_s").to_pylist()
    velocities = notes_table.column("velocity").to_pylist()
    note_uids = (
        notes_table.column("note_uid").to_pylist()
        if is_v3 and "note_uid" in columns
        else [None] * len(event_ids)
    )

    for event_id, partial, pitch, start, duration, velocity, note_uid in zip(
        event_ids,
        partials,
        pitches,
        starts,
        durations,
        velocities,
        note_uids,
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
        if is_v3 and note_uid is None:
            continue
        yield {
            "event_id": str(event_id),
            "partial_index": int(partial),
            "midi_pitch": int(pitch),
            "start_utc": start_f,
            "duration_s": duration_f,
            "velocity": int(velocity),
            "note_uid": str(note_uid) if note_uid is not None else None,
        }


def _seconds_to_ticks(seconds: float) -> int:
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
