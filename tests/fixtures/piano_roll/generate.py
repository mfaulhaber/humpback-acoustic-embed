"""Generator for the synthetic three-event piano-roll fixture.

Run as a module to refresh the committed ``synthetic_three_events.wav`` and
``synthetic_three_events.json`` next to this file:

    uv run python -m tests.fixtures.piano_roll.generate

The fixture covers three events with known fundamentals and integer harmonics
that the Piano Roll Notes worker is expected to recover as the listed MIDI
pitches. Both backend integration tests and the frontend Playwright spec
consume this fixture so the same ground truth drives both layers.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import TypedDict

import numpy as np

SAMPLE_RATE = 22050
DURATION_S = 6.0


class FixtureEvent(TypedDict):
    event_id: str
    start_s: float
    end_s: float
    fundamental_hz: float
    n_harmonics: int


class ExpectedNote(TypedDict):
    event_id: str
    midi_pitch: int
    partial_index: int


EVENTS: list[FixtureEvent] = [
    {
        "event_id": "ev1",
        "start_s": 0.50,
        "end_s": 1.20,
        "fundamental_hz": 220.0,
        "n_harmonics": 3,
    },
    {
        "event_id": "ev2",
        "start_s": 2.00,
        "end_s": 2.80,
        "fundamental_hz": 330.0,
        "n_harmonics": 2,
    },
    {
        "event_id": "ev3",
        "start_s": 4.00,
        "end_s": 4.80,
        "fundamental_hz": 523.25,
        "n_harmonics": 2,
    },
]

EXPECTED_NOTES: list[ExpectedNote] = [
    {"event_id": "ev1", "midi_pitch": 57, "partial_index": 0},
    {"event_id": "ev1", "midi_pitch": 69, "partial_index": 1},
    {"event_id": "ev2", "midi_pitch": 64, "partial_index": 0},
    {"event_id": "ev3", "midi_pitch": 72, "partial_index": 0},
    {"event_id": "ev3", "midi_pitch": 84, "partial_index": 1},
]


def _harmonic_stack(
    fundamental_hz: float, duration_s: float, n_harmonics: int
) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        audio += (1.0 / k) * np.sin(2 * np.pi * fundamental_hz * k * t)
    return (0.5 * audio).astype(np.float32)


def build_audio() -> np.ndarray:
    buffer = np.zeros(int(DURATION_S * SAMPLE_RATE), dtype=np.float32)
    for event in EVENTS:
        snippet = _harmonic_stack(
            event["fundamental_hz"],
            event["end_s"] - event["start_s"],
            event["n_harmonics"],
        )
        start_idx = int(event["start_s"] * SAMPLE_RATE)
        end_idx = start_idx + snippet.shape[0]
        buffer[start_idx:end_idx] += snippet
    return buffer


def fixture_dir() -> Path:
    return Path(__file__).resolve().parent


def write_wav(buffer: np.ndarray, path: Path) -> None:
    clipped = np.clip(buffer, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def write_metadata(path: Path) -> None:
    payload = {
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "events": EVENTS,
        "expected_notes": EXPECTED_NOTES,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = fixture_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = build_audio()
    write_wav(audio, out_dir / "synthetic_three_events.wav")
    write_metadata(out_dir / "synthetic_three_events.json")


if __name__ == "__main__":
    main()
