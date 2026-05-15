import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { EventDetailPanel } from "./EventDetailPanel";
import type { EffectiveEvent } from "@/components/timeline/overlays/EventBarOverlay";

const event: EffectiveEvent = {
  eventId: "ev-1",
  regionId: "r-1",
  startSec: 10,
  endSec: 12,
  originalStartSec: 10,
  originalEndSec: 12,
  confidence: 0.9,
  correctionType: null,
  effectiveType: null,
  typeSource: null,
};

describe("EventDetailPanel", () => {
  it("marks an event for deletion immediately", () => {
    const onDelete = vi.fn();
    render(
      <EventDetailPanel
        event={event}
        onDelete={onDelete}
        isPlaying={false}
        onPlaySlice={() => {}}
        jobStartEpoch={0}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete Event" }));
    expect(onDelete).toHaveBeenCalledWith("ev-1");
    expect(screen.queryByRole("heading", { name: "Delete event" })).toBeNull();
  });
});
