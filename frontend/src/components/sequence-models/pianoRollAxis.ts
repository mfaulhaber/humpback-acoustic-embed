// Y-axis MIDI helpers for the Event Encoder Piano Roll Notes view.
// v3 notes (ADR-069) can carry pitches outside the 88-key acoustic-piano
// band, so the axis covers MIDI 12..120 with the 88-key band (21..108)
// rendered as the "piano" zone and the surrounding bands tinted to mark
// them as out-of-piano.

export const MIDI_MIN_PITCH = 12;
export const MIDI_MAX_PITCH = 120;
export const MIDI_PITCH_COUNT = MIDI_MAX_PITCH - MIDI_MIN_PITCH + 1;

export const MIDI_PIANO_MIN_PITCH = 21;
export const MIDI_PIANO_MAX_PITCH = 108;

const NOTE_NAMES = [
  "C",
  "C#",
  "D",
  "D#",
  "E",
  "F",
  "F#",
  "G",
  "G#",
  "A",
  "A#",
  "B",
];

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

export function isBlackKey(midiPitch: number): boolean {
  return BLACK_KEYS.has(((midiPitch % 12) + 12) % 12);
}

export function isInPianoBand(midiPitch: number): boolean {
  return (
    midiPitch >= MIDI_PIANO_MIN_PITCH && midiPitch <= MIDI_PIANO_MAX_PITCH
  );
}

export function midiToFrequency(midiPitch: number): number {
  return 440 * 2 ** ((midiPitch - 69) / 12);
}

export function midiNoteName(midiPitch: number): string {
  const pc = ((midiPitch % 12) + 12) % 12;
  const octave = Math.floor(midiPitch / 12) - 1;
  return `${NOTE_NAMES[pc]}${octave}`;
}

export function midiPitchAtY(
  y: number,
  plotTop: number,
  plotBottom: number,
): number {
  const height = Math.max(1, plotBottom - plotTop);
  const ratio = (plotBottom - y) / height;
  const raw = MIDI_MIN_PITCH + ratio * MIDI_PITCH_COUNT;
  return Math.max(MIDI_MIN_PITCH, Math.min(MIDI_MAX_PITCH, raw));
}

export function midiPitchToY(
  midi: number,
  plotTop: number,
  plotBottom: number,
): number {
  const height = Math.max(1, plotBottom - plotTop);
  const ratio = (midi - MIDI_MIN_PITCH + 0.5) / MIDI_PITCH_COUNT;
  return plotBottom - ratio * height;
}

export function partialIndexLabel(partialIndex: number): string {
  if (partialIndex === 0) return "F0";
  if (partialIndex > 0) return `${partialIndex + 1}x F0`;
  return "–";
}
