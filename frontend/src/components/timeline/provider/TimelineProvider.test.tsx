import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { TimelineProvider } from "./TimelineProvider";
import { useTimelineContext } from "./useTimelineContext";
import { FULL_ZOOM, REVIEW_ZOOM } from "./types";
import type { TimelinePlaybackHandle, ZoomPreset } from "./types";
import { createRef, type ReactNode } from "react";

beforeAll(() => {
  vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockReturnValue(undefined);
});

afterEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  vi.restoreAllMocks();
});

function wrapper(props?: {
  zoomLevels?: ZoomPreset[];
  jobStart?: number;
  jobEnd?: number;
  defaultZoom?: string;
  disableKeyboardShortcuts?: boolean;
  onZoomChange?: (key: string) => void;
  onPlayStateChange?: (playing: boolean) => void;
}) {
  const zoomLevels = props?.zoomLevels ?? FULL_ZOOM;
  const jobStart = props?.jobStart ?? 1000;
  const jobEnd = props?.jobEnd ?? 87400;
  const defaultZoom = props?.defaultZoom;

  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <TimelineProvider
        jobStart={jobStart}
        jobEnd={jobEnd}
        zoomLevels={zoomLevels}
        defaultZoom={defaultZoom}
        playback="slice"
        audioUrlBuilder={() => ""}
        disableKeyboardShortcuts={props?.disableKeyboardShortcuts}
        onZoomChange={props?.onZoomChange}
        onPlayStateChange={props?.onPlayStateChange}
      >
        {children}
      </TimelineProvider>
    );
  };
}

describe("TimelineProvider zoom presets", () => {
  it("validates FULL_ZOOM presets have positive span and tileDuration", () => {
    for (const preset of FULL_ZOOM) {
      expect(preset.span).toBeGreaterThan(0);
      expect(preset.tileDuration).toBeGreaterThan(0);
    }
  });

  it("validates REVIEW_ZOOM presets have positive span and tileDuration", () => {
    for (const preset of REVIEW_ZOOM) {
      expect(preset.span).toBeGreaterThan(0);
      expect(preset.tileDuration).toBeGreaterThan(0);
    }
  });

  it("selects defaultZoom by key", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ defaultZoom: "5m" }),
    });
    expect(result.current.activePreset.key).toBe("5m");
  });

  it("falls back to index 0 if defaultZoom key not found", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ defaultZoom: "invalid" }),
    });
    expect(result.current.activePreset.key).toBe(FULL_ZOOM[0].key);
  });
});

describe("pxPerSec derivation", () => {
  it("derives pxPerSec from viewportSpan and canvasWidth", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ defaultZoom: "1m" }),
    });

    act(() => {
      result.current.setViewportDimensions(600, 300);
    });

    expect(result.current.pxPerSec).toBe(600 / 60);
  });

  it("returns 1 when viewport width is 0", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper(),
    });
    expect(result.current.pxPerSec).toBe(1);
  });
});

describe("pan clamping", () => {
  it("clamps pan to jobStart", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ jobStart: 1000, jobEnd: 2000 }),
    });

    act(() => {
      result.current.pan(500);
    });

    expect(result.current.centerTimestamp).toBe(1000);
  });

  it("clamps pan to jobEnd", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ jobStart: 1000, jobEnd: 2000 }),
    });

    act(() => {
      result.current.pan(3000);
    });

    expect(result.current.centerTimestamp).toBe(2000);
  });

  it("allows pan within bounds", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ jobStart: 1000, jobEnd: 2000 }),
    });

    act(() => {
      result.current.pan(1500);
    });

    expect(result.current.centerTimestamp).toBe(1500);
  });
});

describe("timeline drag panning", () => {
  it("pans from pointer delta using pxPerSec", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ jobStart: 1000, jobEnd: 2000, defaultZoom: "1m" }),
    });

    act(() => {
      result.current.setViewportDimensions(600, 300);
    });
    const startCenter = result.current.centerTimestamp;

    act(() => {
      expect(result.current.beginDragPan(300)).toBe(true);
    });
    act(() => {
      result.current.updateDragPan(360);
    });

    expect(result.current.centerTimestamp).toBe(startCenter - 6);
    expect(result.current.isDraggingTimeline).toBe(true);

    act(() => {
      result.current.endDragPan();
    });
    expect(result.current.isDraggingTimeline).toBe(false);
  });

  it("does not start a drag while playback is active", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ jobStart: 1000, jobEnd: 2000 }),
    });

    act(() => {
      result.current.play(1500, 10);
    });

    act(() => {
      expect(result.current.beginDragPan(300)).toBe(false);
    });
    expect(result.current.isDraggingTimeline).toBe(false);
  });
});

describe("usePlayback slice mode", () => {
  it("starts with isPlaying false", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper(),
    });
    expect(result.current.isPlaying).toBe(false);
  });

  it("sets playback epoch on play and clears it on pause", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper(),
    });

    act(() => {
      result.current.play(1500, 10);
    });
    expect(result.current.playbackEpoch).toBe(1500);

    act(() => {
      result.current.pause();
    });
    expect(result.current.playbackEpoch).toBeNull();
  });
});

describe("useTimelineContext throws outside provider", () => {
  it("throws when used outside TimelineProvider", () => {
    expect(() => {
      renderHook(() => useTimelineContext());
    }).toThrow("useTimelineContext must be used within a TimelineProvider");
  });
});

describe("TimelineProvider renders without ref (backwards compatible)", () => {
  it("renders and provides context without ref", () => {
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper(),
    });
    expect(result.current.isPlaying).toBe(false);
    expect(result.current.playbackEpoch).toBeNull();
  });
});

describe("onZoomChange callback", () => {
  it("fires when zoom level changes", () => {
    const onZoomChange = vi.fn();
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ onZoomChange, defaultZoom: "1h" }),
    });

    act(() => {
      result.current.zoomIn();
    });

    expect(onZoomChange).toHaveBeenCalledWith("15m");
  });
});

describe("onPlayStateChange callback", () => {
  it("fires on play", () => {
    const onPlayStateChange = vi.fn();
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ onPlayStateChange }),
    });

    act(() => {
      result.current.play(1500, 10);
    });

    expect(onPlayStateChange).toHaveBeenCalledWith(true);
  });

  it("fires on pause", () => {
    const onPlayStateChange = vi.fn();
    const { result } = renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ onPlayStateChange }),
    });

    act(() => {
      result.current.play(1500, 10);
    });
    act(() => {
      result.current.pause();
    });

    expect(onPlayStateChange).toHaveBeenCalledWith(false);
  });
});

describe("disableKeyboardShortcuts", () => {
  it("does not register keyboard listener when disabled", () => {
    const addSpy = vi.spyOn(window, "addEventListener");
    renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ disableKeyboardShortcuts: true }),
    });

    const keydownCalls = addSpy.mock.calls.filter(([event]) => event === "keydown");
    expect(keydownCalls).toHaveLength(0);
    addSpy.mockRestore();
  });

  it("registers keyboard listener when not disabled", () => {
    const addSpy = vi.spyOn(window, "addEventListener");
    renderHook(() => useTimelineContext(), {
      wrapper: wrapper({ disableKeyboardShortcuts: false }),
    });

    const keydownCalls = addSpy.mock.calls.filter(([event]) => event === "keydown");
    expect(keydownCalls.length).toBeGreaterThan(0);
    addSpy.mockRestore();
  });
});
