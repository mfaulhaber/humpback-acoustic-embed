import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";
import type { ReactNode } from "react";

import { TimelineProvider } from "../provider/TimelineProvider";
import { FULL_ZOOM } from "../provider/types";
import { useOverlayContext } from "../overlays/OverlayContext";
import { Spectrogram } from "./Spectrogram";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal("ResizeObserver", ResizeObserverStub);
  vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockReturnValue(undefined);
});

afterAll(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function Wrap({ children }: { children: ReactNode }) {
  return (
    <TimelineProvider
      jobStart={1000}
      jobEnd={2000}
      zoomLevels={FULL_ZOOM}
      playback="slice"
      audioUrlBuilder={() => ""}
      disableKeyboardShortcuts
    >
      {children}
    </TimelineProvider>
  );
}

describe("Spectrogram overlay layers", () => {
  it("renders a clipped band layer with overflow: hidden", () => {
    const { getByTestId } = render(
      <Wrap>
        <Spectrogram tileUrlBuilder={() => "x"} jobId="job1" />
      </Wrap>,
    );
    const band = getByTestId("overlay-band-layer") as HTMLElement;
    expect(band.style.overflow).toBe("hidden");
    expect(band.style.position).toBe("absolute");
  });

  it("renders an unclipped tooltip layer as a sibling of the band layer", () => {
    const { getByTestId } = render(
      <Wrap>
        <Spectrogram tileUrlBuilder={() => "x"} jobId="job1" />
      </Wrap>,
    );
    const band = getByTestId("overlay-band-layer") as HTMLElement;
    const tooltip = getByTestId("overlay-tooltip-layer") as HTMLElement;
    expect(tooltip.style.overflow).not.toBe("hidden");
    expect(tooltip.style.pointerEvents).toBe("none");
    expect(band.parentElement).toBe(tooltip.parentElement);
  });

  it("exposes the tooltip layer DOM node via OverlayContext.tooltipPortalTarget", () => {
    let captured: HTMLElement | null = null;
    function Capture() {
      const ctx = useOverlayContext();
      captured = ctx.tooltipPortalTarget;
      return null;
    }
    const { getByTestId } = render(
      <Wrap>
        <Spectrogram tileUrlBuilder={() => "x"} jobId="job1">
          <Capture />
        </Spectrogram>
      </Wrap>,
    );
    const tooltip = getByTestId("overlay-tooltip-layer");
    expect(captured).toBe(tooltip);
  });
});
