import { describe, expect, it } from "vitest";
import React from "react";
import { fireEvent, render } from "@testing-library/react";

import { OverlayContext } from "./OverlayContext";
import type { OverlayContextValue } from "./OverlayContext";
import { DetectionOverlay } from "./DetectionOverlay";
import { VocalizationOverlay } from "./VocalizationOverlay";
import type { DetectionRow, TimelineVocalizationLabel } from "@/api/types";

function makeOverlayValue(target: HTMLElement | null): OverlayContextValue {
  return {
    viewStart: 0,
    viewEnd: 100,
    pxPerSec: 10,
    canvasWidth: 1000,
    canvasHeight: 120,
    epochToX: (epoch: number) => epoch * 10,
    xToEpoch: (x: number) => x / 10,
    tooltipPortalTarget: target,
  };
}

const detection: DetectionRow = {
  start_utc: 1,
  end_utc: 5,
  humpback: 1,
  orca: 0,
  ship: 0,
  background: 0,
  avg_confidence: 0.5,
  peak_confidence: 0.7,
} as DetectionRow;

const vocLabel: TimelineVocalizationLabel = {
  start_utc: 1,
  end_utc: 5,
  label: "moan",
  source: "manual",
  confidence: null,
} as TimelineVocalizationLabel;

describe("DetectionOverlay tooltip portal", () => {
  it("renders the tooltip into the portal target when supplied", () => {
    const portal = document.createElement("div");
    portal.setAttribute("data-testid", "portal");
    document.body.appendChild(portal);

    const value = makeOverlayValue(portal);
    const { container, getByTestId } = render(
      React.createElement(
        OverlayContext.Provider,
        { value },
        React.createElement(DetectionOverlay, {
          detections: [detection],
          visible: true,
        }),
      ),
    );

    const bar = container.querySelector('[data-testid="detection-overlay"] > div') as HTMLElement;
    fireEvent.mouseEnter(bar);

    const tooltip = getByTestId("detection-overlay-tooltip");
    expect(portal.contains(tooltip)).toBe(true);
    expect(container.querySelector('[data-testid="detection-overlay"]')!.contains(tooltip)).toBe(false);

    document.body.removeChild(portal);
  });

  it("renders the tooltip inline when portal target is null", () => {
    const value = makeOverlayValue(null);
    const { container, getByTestId } = render(
      React.createElement(
        OverlayContext.Provider,
        { value },
        React.createElement(DetectionOverlay, {
          detections: [detection],
          visible: true,
        }),
      ),
    );

    const bar = container.querySelector('[data-testid="detection-overlay"] > div') as HTMLElement;
    fireEvent.mouseEnter(bar);

    const tooltip = getByTestId("detection-overlay-tooltip");
    expect(container.querySelector('[data-testid="detection-overlay"]')!.contains(tooltip)).toBe(true);
  });
});

describe("VocalizationOverlay tooltip portal", () => {
  it("renders the tooltip into the portal target when supplied", () => {
    const portal = document.createElement("div");
    document.body.appendChild(portal);

    const value = makeOverlayValue(portal);
    const { container, getByTestId } = render(
      React.createElement(
        OverlayContext.Provider,
        { value },
        React.createElement(VocalizationOverlay, {
          labels: [vocLabel],
          visible: true,
        }),
      ),
    );

    const bar = container.querySelector('[data-testid="vocalization-overlay"] > div') as HTMLElement;
    fireEvent.mouseEnter(bar);

    const tooltip = getByTestId("vocalization-overlay-tooltip");
    expect(portal.contains(tooltip)).toBe(true);
    expect(container.querySelector('[data-testid="vocalization-overlay"]')!.contains(tooltip)).toBe(false);

    document.body.removeChild(portal);
  });

  it("renders the tooltip inline when portal target is null", () => {
    const value = makeOverlayValue(null);
    const { container, getByTestId } = render(
      React.createElement(
        OverlayContext.Provider,
        { value },
        React.createElement(VocalizationOverlay, {
          labels: [vocLabel],
          visible: true,
        }),
      ),
    );

    const bar = container.querySelector('[data-testid="vocalization-overlay"] > div') as HTMLElement;
    fireEvent.mouseEnter(bar);

    const tooltip = getByTestId("vocalization-overlay-tooltip");
    expect(container.querySelector('[data-testid="vocalization-overlay"]')!.contains(tooltip)).toBe(true);
  });
});
