import { describe, it, expect } from "vitest";
import { freqLabels } from "./FrequencyAxis";
import { formatTimeLabel, timeLabelStepSec } from "./TimeAxis";

describe("FrequencyAxis label generation", () => {
  it("generates labels for 0-3000 Hz range", () => {
    const labels = freqLabels([0, 3000]);
    expect(labels.length).toBeGreaterThan(0);
    expect(labels.some((l) => l.label.includes("k"))).toBe(true);
  });

  it("generates labels for narrow range (0-500 Hz)", () => {
    const labels = freqLabels([0, 500]);
    const steps = labels.filter((l) => l.label !== "Hz");
    for (let i = 1; i < steps.length; i++) {
      expect(steps[i].hz - steps[i - 1].hz).toBe(100);
    }
  });

  it("generates labels for wide range (0-10000 Hz)", () => {
    const labels = freqLabels([0, 10000]);
    expect(labels.some((l) => l.hz === 2000)).toBe(true);
  });

  it("always ends with Hz label when lo freq not in steps", () => {
    const labels = freqLabels([50, 3000]);
    expect(labels[labels.length - 1].label).toBe("Hz");
  });
});

describe("TimeAxis label formatting", () => {
  it("shows MM-DD HH:MM for wide zooms (>=21600s)", () => {
    const epoch = Date.UTC(2024, 5, 15, 14, 30, 0) / 1000;
    const label = formatTimeLabel(epoch, 86400);
    expect(label).toBe("06-15 14:30");
  });

  it("shows HH:MM:SS for narrow zooms (<=300s)", () => {
    const epoch = Date.UTC(2024, 5, 15, 14, 30, 45) / 1000;
    const label = formatTimeLabel(epoch, 60);
    expect(label).toBe("14:30:45");
  });

  it("shows HH:MM for mid zooms", () => {
    const epoch = Date.UTC(2024, 5, 15, 14, 30, 45) / 1000;
    const label = formatTimeLabel(epoch, 3600);
    expect(label).toBe("14:30");
  });
});

describe("TimeAxis step sizes", () => {
  it("returns sensible steps for various spans", () => {
    expect(timeLabelStepSec(86400)).toBe(14400);
    expect(timeLabelStepSec(21600)).toBe(3600);
    expect(timeLabelStepSec(3600)).toBe(600);
    expect(timeLabelStepSec(900)).toBe(120);
    expect(timeLabelStepSec(300)).toBe(30);
    expect(timeLabelStepSec(60)).toBe(10);
    expect(timeLabelStepSec(10)).toBe(5);
  });
});

describe("OverlayContext epochToX/xToEpoch round-trip", () => {
  it("converts epoch to X and back accurately", () => {
    const centerTimestamp = 5000;
    const pxPerSec = 10;
    const canvasWidth = 600;

    const epochToX = (epoch: number) => (epoch - centerTimestamp) * pxPerSec + canvasWidth / 2;
    const xToEpoch = (x: number) => centerTimestamp + (x - canvasWidth / 2) / pxPerSec;

    const testEpochs = [4970, 5000, 5030, 4985, 5015];
    for (const epoch of testEpochs) {
      const x = epochToX(epoch);
      const recovered = xToEpoch(x);
      expect(recovered).toBeCloseTo(epoch, 10);
    }
  });

  it("maps center epoch to canvas center", () => {
    const centerTimestamp = 5000;
    const pxPerSec = 10;
    const canvasWidth = 600;
    const epochToX = (epoch: number) => (epoch - centerTimestamp) * pxPerSec + canvasWidth / 2;

    expect(epochToX(centerTimestamp)).toBe(canvasWidth / 2);
  });
});
