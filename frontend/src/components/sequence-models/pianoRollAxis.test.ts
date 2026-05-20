import { describe, expect, it } from "vitest";
import {
  MIDI_MAX_PITCH,
  MIDI_MIN_PITCH,
  MIDI_PITCH_COUNT,
  isBlackKey,
  midiNoteName,
  midiPitchToY,
  midiToFrequency,
  partialIndexLabel,
} from "./pianoRollAxis";

describe("pianoRollAxis", () => {
  it("covers the 88-key range", () => {
    expect(MIDI_MIN_PITCH).toBe(21);
    expect(MIDI_MAX_PITCH).toBe(108);
    expect(MIDI_PITCH_COUNT).toBe(88);
  });

  it("identifies black keys", () => {
    expect(isBlackKey(60)).toBe(false); // C4
    expect(isBlackKey(61)).toBe(true); // C#4
    expect(isBlackKey(69)).toBe(false); // A4
    expect(isBlackKey(70)).toBe(true); // A#4
  });

  it("maps MIDI to frequency", () => {
    expect(midiToFrequency(69)).toBeCloseTo(440, 6);
    expect(midiToFrequency(57)).toBeCloseTo(220, 6); // A3
    expect(midiToFrequency(21)).toBeCloseTo(27.5, 4); // A0
  });

  it("formats note names with octave", () => {
    expect(midiNoteName(60)).toBe("C4");
    expect(midiNoteName(21)).toBe("A0");
    expect(midiNoteName(108)).toBe("C8");
    expect(midiNoteName(61)).toBe("C#4");
  });

  it("places higher MIDI at lower y", () => {
    const high = midiPitchToY(108, 0, 880);
    const low = midiPitchToY(21, 0, 880);
    expect(high).toBeLessThan(low);
  });

  it("labels partial indices", () => {
    expect(partialIndexLabel(0)).toBe("F0");
    expect(partialIndexLabel(1)).toBe("2x F0");
    expect(partialIndexLabel(2)).toBe("3x F0");
    expect(partialIndexLabel(-1)).toBe("–");
  });
});
