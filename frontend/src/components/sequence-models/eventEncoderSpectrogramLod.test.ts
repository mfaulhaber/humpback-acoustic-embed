import { describe, expect, it } from "vitest";

import {
  chooseSpectrogramLod,
  EVENT_ENCODER_SPECTROGRAM_LODS,
  type SpectrogramLod,
} from "./eventEncoderSpectrogramLod";

describe("chooseSpectrogramLod", () => {
  it("chooses a coarse LOD for broad smooth viewports", () => {
    const lod = chooseSpectrogramLod({
      viewportSpan: 900,
      viewportWidth: 1200,
    });

    expect(lod.key).toBe("15m");
    expect(lod.tileDuration).toBe(150);
  });

  it("chooses a medium LOD for minute-scale smooth viewports", () => {
    expect(
      chooseSpectrogramLod({
        viewportSpan: 160,
        viewportWidth: 1200,
      }).key,
    ).toBe("5m");
    expect(
      chooseSpectrogramLod({
        viewportSpan: 30,
        viewportWidth: 1200,
      }).key,
    ).toBe("1m");
  });

  it("chooses fine LODs for close smooth viewports", () => {
    expect(
      chooseSpectrogramLod({
        viewportSpan: 12,
        viewportWidth: 1200,
      }).key,
    ).toBe("30s");
    expect(
      chooseSpectrogramLod({
        viewportSpan: 5,
        viewportWidth: 1200,
      }).key,
    ).toBe("10s");
  });

  it("keeps the current LOD near a threshold when hysteresis allows it", () => {
    const lods: SpectrogramLod[] = [
      { key: "coarse", tileDuration: 100, tileWidthPx: 100 },
      { key: "fine", tileDuration: 50, tileWidthPx: 100 },
    ];

    const next = chooseSpectrogramLod({
      viewportSpan: 69,
      viewportWidth: 100,
      currentKey: "coarse",
      lods,
      hysteresisRatio: 1.3,
    });

    expect(next.key).toBe("coarse");
  });

  it("switches LOD when the better match clears hysteresis", () => {
    const lods: SpectrogramLod[] = [
      { key: "coarse", tileDuration: 100, tileWidthPx: 100 },
      { key: "fine", tileDuration: 50, tileWidthPx: 100 },
    ];

    const next = chooseSpectrogramLod({
      viewportSpan: 52,
      viewportWidth: 100,
      currentKey: "coarse",
      lods,
      hysteresisRatio: 1.3,
    });

    expect(next.key).toBe("fine");
  });

  it("falls back to the current or first valid LOD for unavailable dimensions", () => {
    expect(
      chooseSpectrogramLod({
        viewportSpan: 0,
        viewportWidth: 1200,
        currentKey: "1m",
      }).key,
    ).toBe("1m");
    expect(
      chooseSpectrogramLod({
        viewportSpan: 30,
        viewportWidth: 0,
      }).key,
    ).toBe(EVENT_ENCODER_SPECTROGRAM_LODS[0].key);
  });
});
