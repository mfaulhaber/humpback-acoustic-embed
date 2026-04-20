import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TimelineProvider } from "../provider/TimelineProvider";
import { FULL_ZOOM } from "../provider/types";
import { ZoomSelector } from "./ZoomSelector";
import { PlaybackControls } from "./PlaybackControls";
import { EditToolbar } from "./EditToolbar";
import { EventNav } from "./EventNav";
import type { ReactNode } from "react";

function Wrapper({ children }: { children: ReactNode }) {
  return (
    <TimelineProvider
      jobStart={1000}
      jobEnd={87400}
      zoomLevels={FULL_ZOOM}
      defaultZoom="1h"
      playback="slice"
      audioUrlBuilder={() => ""}
    >
      {children}
    </TimelineProvider>
  );
}

describe("ZoomSelector", () => {
  it("renders all preset levels from context", () => {
    render(<ZoomSelector />, { wrapper: Wrapper });
    for (const preset of FULL_ZOOM) {
      expect(screen.getByText(preset.key)).toBeDefined();
    }
  });
});

describe("PlaybackControls", () => {
  it("compact variant hides skip buttons", () => {
    const { container } = render(<PlaybackControls variant="compact" />, { wrapper: Wrapper });
    const skipBackButtons = container.querySelectorAll("svg");
    const svgClasses = Array.from(skipBackButtons).map((svg) => svg.classList.toString());
    expect(svgClasses.some((c) => c.includes("skip"))).toBe(false);
  });

  it("default variant shows timestamp", () => {
    render(<PlaybackControls />, { wrapper: Wrapper });
    expect(screen.getByText(/UTC/)).toBeDefined();
  });
});

describe("EditToolbar", () => {
  it("Save button disabled when pendingCount === 0", () => {
    render(<EditToolbar pendingCount={0} onSave={() => {}} onCancel={() => {}} />);
    const saveBtn = screen.getByText("Save");
    expect(saveBtn.hasAttribute("disabled")).toBe(true);
  });

  it("Save button enabled when pendingCount > 0", () => {
    render(<EditToolbar pendingCount={3} onSave={() => {}} onCancel={() => {}} />);
    const saveBtn = screen.getByText("Save");
    expect(saveBtn.hasAttribute("disabled")).toBe(false);
  });

  it("shows pending count", () => {
    render(<EditToolbar pendingCount={5} onSave={() => {}} onCancel={() => {}} />);
    expect(screen.getByText("5 pending")).toBeDefined();
  });
});

describe("EventNav", () => {
  it("Prev disabled at index 0", () => {
    const { container } = render(
      <EventNav currentIndex={0} totalCount={5} onPrev={() => {}} onNext={() => {}} />,
    );
    const buttons = container.querySelectorAll("button");
    expect(buttons[0].hasAttribute("disabled")).toBe(true);
  });

  it("Next disabled at last index", () => {
    const { container } = render(
      <EventNav currentIndex={4} totalCount={5} onPrev={() => {}} onNext={() => {}} />,
    );
    const buttons = container.querySelectorAll("button");
    expect(buttons[1].hasAttribute("disabled")).toBe(true);
  });

  it("shows current/total count", () => {
    render(<EventNav currentIndex={2} totalCount={10} onPrev={() => {}} onNext={() => {}} />);
    expect(screen.getByText("3/10")).toBeDefined();
  });
});
