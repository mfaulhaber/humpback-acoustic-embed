import React from "react";
import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import type { MotifOccurrence } from "@/api/sequenceModels";
import { MOTIF_PALETTE } from "@/lib/motifColor";
import { OverlayContext } from "./OverlayContext";
import { RegionBoundaryMarkers } from "./RegionBoundaryMarkers";
import { MotifHighlightOverlay } from "./MotifHighlightOverlay";

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

describe("RegionBoundaryMarkers", () => {
  const overlayValue = {
    viewStart: 100,
    viewEnd: 200,
    pxPerSec: 10,
    canvasWidth: 1000,
    canvasHeight: 120,
    epochToX: (epoch: number) => (epoch - 100) * 10,
    xToEpoch: (x: number) => 100 + x / 10,
  };

  it("can render boundary lines without dimming outside the active region", () => {
    const { container } = render(
      React.createElement(
        OverlayContext.Provider,
        { value: overlayValue },
        React.createElement(RegionBoundaryMarkers, {
          startEpoch: 120,
          endEpoch: 150,
          dimOutside: false,
          lineColor: "white",
          lineStyle: "solid",
        }),
      ),
    );

    expect(container.querySelectorAll('[style*="rgba(0, 0, 0, 0.4)"]')).toHaveLength(0);
    expect(container.querySelectorAll('[style*="1.5px solid white"]')).toHaveLength(2);
  });
});

describe("MotifHighlightOverlay", () => {
  const overlayValue = {
    viewStart: 100,
    viewEnd: 200,
    pxPerSec: 10,
    canvasWidth: 1000,
    canvasHeight: 120,
    epochToX: (epoch: number) => (epoch - 100) * 10,
    xToEpoch: (x: number) => 100 + x / 10,
  };

  function makeOcc(motifKey: string, start: number, end: number, idx: number): MotifOccurrence {
    return {
      occurrence_id: `${motifKey}-${idx}`,
      motif_key: motifKey,
      states: [],
      source_kind: "region_crnn",
      group_key: "",
      event_source_key: "",
      audio_source_key: null,
      token_start_index: 0,
      token_end_index: 0,
      raw_start_index: 0,
      raw_end_index: 0,
      start_timestamp: start,
      end_timestamp: end,
      duration_seconds: end - start,
      event_core_fraction: 0,
      background_fraction: 0,
      mean_call_probability: null,
      anchor_event_id: null,
      anchor_timestamp: 0,
      relative_start_seconds: 0,
      relative_end_seconds: 0,
      anchor_strategy: "",
    };
  }

  it("renders a band per visible occurrence", () => {
    const occurrences = [
      makeOcc("a", 110, 120, 0),
      makeOcc("b", 130, 140, 1),
      makeOcc("c", 300, 310, 2), // outside view
    ];
    const { container } = render(
      React.createElement(
        OverlayContext.Provider,
        { value: overlayValue },
        React.createElement(MotifHighlightOverlay, {
          occurrences,
          activeOccurrenceIndex: 0,
          colorIndex: 0,
          numLabels: 8,
        }),
      ),
    );
    expect(container.querySelectorAll('[data-testid="mt-motif-highlight-band"]')).toHaveLength(2);
  });

  it("with a colorForMotifKey mapper, distinct keys get distinct background colors", () => {
    const occurrences = [
      makeOcc("a", 110, 120, 0),
      makeOcc("b", 130, 140, 1),
      makeOcc("a", 150, 160, 2),
    ];
    const palette = MOTIF_PALETTE;
    const colorForMotifKey = (key: string) => palette[key === "a" ? 0 : 1];
    const { container } = render(
      React.createElement(
        OverlayContext.Provider,
        { value: overlayValue },
        React.createElement(MotifHighlightOverlay, {
          occurrences,
          activeOccurrenceIndex: 0,
          colorIndex: 0,
          numLabels: 8,
          colorForMotifKey,
        }),
      ),
    );
    const bands = container.querySelectorAll<HTMLDivElement>('[data-testid="mt-motif-highlight-band"]');
    expect(bands).toHaveLength(3);
    // Same motif_key → same background; different key → different.
    expect((bands[0] as HTMLElement).style.background).toEqual(
      (bands[2] as HTMLElement).style.background,
    );
    expect((bands[0] as HTMLElement).style.background).not.toEqual(
      (bands[1] as HTMLElement).style.background,
    );
  });

  it("with a color mapper, only the active occurrence renders a dashed outline", () => {
    const occurrences = [
      makeOcc("a", 110, 120, 0),
      makeOcc("b", 130, 140, 1),
    ];
    const palette = MOTIF_PALETTE;
    const colorForMotifKey = (key: string) => palette[key === "a" ? 0 : 1];
    const { container } = render(
      React.createElement(
        OverlayContext.Provider,
        { value: overlayValue },
        React.createElement(MotifHighlightOverlay, {
          occurrences,
          activeOccurrenceIndex: 1,
          colorIndex: 0,
          numLabels: 8,
          colorForMotifKey,
        }),
      ),
    );
    const bands = container.querySelectorAll<HTMLDivElement>('[data-testid="mt-motif-highlight-band"]');
    expect((bands[0] as HTMLElement).style.outline).toBe("none");
    expect((bands[1] as HTMLElement).style.outline).toContain("dashed");
    expect((bands[1] as HTMLElement).getAttribute("data-active")).toBe("true");
  });

  it("without a color mapper, falls back to the legacy single-color behavior", () => {
    const occurrences = [
      makeOcc("a", 110, 120, 0),
      makeOcc("b", 130, 140, 1),
    ];
    const { container } = render(
      React.createElement(
        OverlayContext.Provider,
        { value: overlayValue },
        React.createElement(MotifHighlightOverlay, {
          occurrences,
          activeOccurrenceIndex: 0,
          colorIndex: 0,
          numLabels: 8,
        }),
      ),
    );
    const bands = container.querySelectorAll<HTMLDivElement>('[data-testid="mt-motif-highlight-band"]');
    // Both rectangles share the same fallback hue (legacy single-color
    // behavior); the active rectangle uses the heavier alpha variant.
    expect((bands[0] as HTMLElement).style.outline).toBe("none");
    expect((bands[1] as HTMLElement).style.outline).toBe("none");
  });
});
