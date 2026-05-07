import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SpanNavBar, type RegionGroup, type SpanInfo } from "./SpanNavBar";

const spans: SpanInfo[] = [
  {
    id: 1,
    eventId: "event-1",
    regionId: "region-alpha",
    startTimestamp: 1_718_438_400,
    endTimestamp: 1_718_438_405,
  },
  {
    id: 2,
    eventId: "event-2",
    regionId: "region-alpha",
    startTimestamp: 1_718_438_405,
    endTimestamp: 1_718_438_410,
  },
  {
    id: 3,
    eventId: "event-3",
    regionId: "region-beta",
    startTimestamp: 1_718_438_410,
    endTimestamp: 1_718_438_415,
  },
];

const regions: RegionGroup[] = [
  { regionId: "region-alpha", startIndex: 0, endIndex: 1 },
  { regionId: "region-beta", startIndex: 2, endIndex: 2 },
];

describe("SpanNavBar", () => {
  it("displays active span and active region context", () => {
    render(
      <SpanNavBar
        spans={spans}
        regions={regions}
        activeIndex={1}
        activeRegionIndex={0}
        onPrevEvent={() => {}}
        onNextEvent={() => {}}
        onPrevRegion={() => {}}
        onNextRegion={() => {}}
        itemLabel="Span"
      />,
    );

    expect(screen.getByTestId("sequence-span-label").textContent ?? "").toContain(
      "Span 2/3",
    );
    expect(screen.getByTestId("sequence-region-nav").textContent ?? "").toContain(
      "Region 1/2",
    );
    expect(screen.getByTestId("sequence-region-nav").textContent ?? "").toContain(
      "region-a",
    );
  });

  it("wires event and region navigation callbacks", () => {
    const onPrevEvent = vi.fn();
    const onNextEvent = vi.fn();
    const onPrevRegion = vi.fn();
    const onNextRegion = vi.fn();

    render(
      <SpanNavBar
        spans={spans}
        regions={regions}
        activeIndex={1}
        activeRegionIndex={0}
        onPrevEvent={onPrevEvent}
        onNextEvent={onNextEvent}
        onPrevRegion={onPrevRegion}
        onNextRegion={onNextRegion}
      />,
    );

    fireEvent.click(screen.getByTestId("sequence-span-prev"));
    fireEvent.click(screen.getByTestId("sequence-span-next"));
    fireEvent.click(screen.getByTestId("sequence-region-next"));

    expect(onPrevEvent).toHaveBeenCalledTimes(1);
    expect(onNextEvent).toHaveBeenCalledTimes(1);
    expect(onPrevRegion).not.toHaveBeenCalled();
    expect(onNextRegion).toHaveBeenCalledTimes(1);
  });

  it("renders nothing when active span is missing", () => {
    const { container } = render(
      <SpanNavBar
        spans={spans}
        regions={regions}
        activeIndex={99}
        activeRegionIndex={0}
        onPrevEvent={() => {}}
        onNextEvent={() => {}}
        onPrevRegion={() => {}}
        onNextRegion={() => {}}
      />,
    );
    expect(container.firstChild).toBeNull();
  });
});
