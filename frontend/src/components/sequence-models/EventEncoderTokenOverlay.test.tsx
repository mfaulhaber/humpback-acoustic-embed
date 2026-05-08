import React from "react";
import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { EventEncoderTimelineEvent } from "@/api/sequenceModels";
import { OverlayContext } from "@/components/timeline/overlays/OverlayContext";
import type { OverlayContextValue } from "@/components/timeline/overlays/OverlayContext";

import { EventEncoderTokenOverlay } from "./EventEncoderTokenOverlay";

const overlayValue: OverlayContextValue = {
  viewStart: 100,
  viewEnd: 200,
  pxPerSec: 10,
  canvasWidth: 1000,
  canvasHeight: 120,
  epochToX: (epoch: number) => (epoch - 100) * 10,
  xToEpoch: (x: number) => 100 + x / 10,
  tooltipPortalTarget: null,
};

const events: EventEncoderTimelineEvent[] = [
  {
    event_id: "evt-1",
    region_id: "region-a",
    source_sequence_key: "hydrophone:test",
    sequence_index: 0,
    start_timestamp: 110,
    end_timestamp: 112,
    token_id: 17,
    token_label: "T17",
    token_confidence: 0.75,
    distance_to_centroid: 0.2,
    second_centroid_distance: null,
    descriptor_values: {},
    descriptor_vector_values: {},
  },
  {
    event_id: "evt-2",
    region_id: "region-a",
    source_sequence_key: "hydrophone:test",
    sequence_index: 1,
    start_timestamp: 115,
    end_timestamp: 115.1,
    token_id: 199,
    token_label: "T199",
    token_confidence: 0.55,
    distance_to_centroid: 0.4,
    second_centroid_distance: 0.8,
    descriptor_values: {},
    descriptor_vector_values: {},
  },
];

function renderOverlay(onSelectEvent = vi.fn()) {
  return render(
    <OverlayContext.Provider value={overlayValue}>
      <EventEncoderTokenOverlay
        events={events}
        selectedEventId="evt-1"
        selectedK={200}
        onSelectEvent={onSelectEvent}
      />
    </OverlayContext.Provider>,
  );
}

describe("EventEncoderTokenOverlay", () => {
  it("positions bars with overlay context geometry", () => {
    const { getByTestId } = renderOverlay();

    const first = getByTestId("eej-token-bar-evt-1");
    expect(first.style.left).toBe("100px");
    expect(first.style.width).toBe("20px");

    const second = getByTestId("eej-token-bar-evt-2");
    expect(second.style.left).toBe("150px");
    expect(second.style.width).toBe("3px");
  });

  it("marks the selected event and renders multi-digit token badges", () => {
    const { getByTestId } = renderOverlay();

    expect(getByTestId("eej-token-bar-evt-1").getAttribute("data-selected")).toBe(
      "true",
    );
    expect(getByTestId("eej-token-bar-evt-2").getAttribute("data-selected")).toBe(
      "false",
    );
    expect(getByTestId("eej-token-badge-evt-1").textContent).toBe("T17");
    expect(getByTestId("eej-token-badge-evt-2").textContent).toBe("T199");
  });

  it("selects an event when its bar is clicked", () => {
    const onSelectEvent = vi.fn();
    const { getByTestId } = renderOverlay(onSelectEvent);

    fireEvent.click(getByTestId("eej-token-bar-evt-2"));

    expect(onSelectEvent).toHaveBeenCalledWith("evt-2");
  });
});
