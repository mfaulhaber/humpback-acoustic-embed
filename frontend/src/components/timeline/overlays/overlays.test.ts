import { describe, it, expect } from "vitest";

describe("overlay positioning accuracy", () => {
  it("epochToX produces correct pixel positions for known values", () => {
    const centerTimestamp = 5000;
    const pxPerSec = 10;
    const canvasWidth = 600;

    const epochToX = (epoch: number) => (epoch - centerTimestamp) * pxPerSec + canvasWidth / 2;

    expect(epochToX(5000)).toBe(300);
    expect(epochToX(5010)).toBe(400);
    expect(epochToX(4990)).toBe(200);
    expect(epochToX(5030)).toBe(600);
    expect(epochToX(4970)).toBe(0);
  });

  it("xToEpoch inverts epochToX exactly", () => {
    const centerTimestamp = 5000;
    const pxPerSec = 10;
    const canvasWidth = 600;

    const epochToX = (epoch: number) => (epoch - centerTimestamp) * pxPerSec + canvasWidth / 2;
    const xToEpoch = (x: number) => centerTimestamp + (x - canvasWidth / 2) / pxPerSec;

    const testPixels = [0, 100, 300, 450, 600];
    for (const px of testPixels) {
      const epoch = xToEpoch(px);
      expect(epochToX(epoch)).toBeCloseTo(px, 10);
    }
  });

  it("bar width calculation is accurate for known durations", () => {
    const pxPerSec = 10;
    const epochToX = (epoch: number) => (epoch - 5000) * pxPerSec + 300;

    const startEpoch = 5005;
    const endEpoch = 5015;
    const x = epochToX(startEpoch);
    const w = epochToX(endEpoch) - x;

    expect(x).toBe(350);
    expect(w).toBe(100);
  });

  it("regions at job boundaries map to correct pixel positions", () => {
    const jobStart = 1000;
    const centerTimestamp = 1500;
    const pxPerSec = 2;
    const canvasWidth = 800;

    const epochToX = (epoch: number) => (epoch - centerTimestamp) * pxPerSec + canvasWidth / 2;

    const regionStartSec = 100;
    const regionEndSec = 200;
    const startEpoch = jobStart + regionStartSec;
    const endEpoch = jobStart + regionEndSec;

    const x = epochToX(startEpoch);
    const w = epochToX(endEpoch) - x;

    expect(x).toBe((1100 - 1500) * 2 + 400);
    expect(w).toBe(100 * 2);
  });
});
