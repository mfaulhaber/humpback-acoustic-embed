export const MIDI_MIN_PITCH = 21;
export const MIDI_MAX_PITCH = 108;
export const MIDI_PITCH_COUNT = MIDI_MAX_PITCH - MIDI_MIN_PITCH + 1;

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
