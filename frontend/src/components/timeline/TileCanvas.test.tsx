import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { render } from "@testing-library/react";
import {
  __notifyTileLoadedForTest,
  subscribeTileLoaded,
  TileCanvas,
} from "./TileCanvas";

describe("subscribeTileLoaded", () => {
  it("invokes the callback on tile-loaded notifications", () => {
    const cb = vi.fn();
    const unsubscribe = subscribeTileLoaded(cb);

    __notifyTileLoadedForTest();
    __notifyTileLoadedForTest();

    expect(cb).toHaveBeenCalledTimes(2);
    unsubscribe();
  });

  it("stops invoking the callback after unsubscribe", () => {
    const cb = vi.fn();
    const unsubscribe = subscribeTileLoaded(cb);

    __notifyTileLoadedForTest();
    unsubscribe();
    __notifyTileLoadedForTest();

    expect(cb).toHaveBeenCalledTimes(1);
  });

  it("notifies multiple subscribers in registration order", () => {
    const order: string[] = [];
    const a = subscribeTileLoaded(() => order.push("a"));
    const b = subscribeTileLoaded(() => order.push("b"));

    __notifyTileLoadedForTest();

    expect(order).toEqual(["a", "b"]);
    a();
    b();
  });
});

describe("rAF dedup pattern (consumer contract)", () => {
  // Mirror the dedup pattern used inside the TileCanvas useEffect: at most one
  // pending rAF per subscriber, even if many notifications arrive in the same
  // frame.

  let rafCallbacks: Array<() => void>;
  let rafSpy: ReturnType<typeof vi.fn>;
  let cafSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    rafCallbacks = [];
    rafSpy = vi.fn((cb: () => void) => {
      rafCallbacks.push(cb);
      return rafCallbacks.length;
    });
    cafSpy = vi.fn();
    vi.stubGlobal("requestAnimationFrame", rafSpy);
    vi.stubGlobal("cancelAnimationFrame", cafSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function makeScheduler(draw: () => void) {
    let pending = 0;
    const schedule = () => {
      if (pending !== 0) return;
      pending = requestAnimationFrame(() => {
        pending = 0;
        draw();
      });
    };
    const cancel = () => {
      if (pending !== 0) {
        cancelAnimationFrame(pending);
        pending = 0;
      }
    };
    return { schedule, cancel };
  }

  it("collapses two notifications in the same frame into one draw", () => {
    const draw = vi.fn();
    const { schedule } = makeScheduler(draw);

    schedule();
    schedule();

    expect(rafSpy).toHaveBeenCalledTimes(1);

    rafCallbacks[0]();
    expect(draw).toHaveBeenCalledTimes(1);
  });

  it("schedules a fresh redraw on the next notification after the frame fires", () => {
    const draw = vi.fn();
    const { schedule } = makeScheduler(draw);

    schedule();
    rafCallbacks[0]();
    schedule();

    expect(rafSpy).toHaveBeenCalledTimes(2);
    rafCallbacks[1]();
    expect(draw).toHaveBeenCalledTimes(2);
  });

  it("cancels any pending rAF on cleanup", () => {
    const draw = vi.fn();
    const { schedule, cancel } = makeScheduler(draw);

    schedule();
    cancel();

    expect(cafSpy).toHaveBeenCalledTimes(1);
  });

  it("cleanup is a no-op when no rAF is pending", () => {
    const draw = vi.fn();
    const { cancel } = makeScheduler(draw);

    cancel();

    expect(cafSpy).not.toHaveBeenCalled();
  });
});

describe("TileCanvas component wires the loader subscription", () => {
  let rafSpy: ReturnType<typeof vi.fn>;
  let rafCallbacks: Array<() => void>;

  beforeEach(() => {
    rafCallbacks = [];
    rafSpy = vi.fn((cb: () => void) => {
      rafCallbacks.push(cb);
      return rafCallbacks.length;
    });
    vi.stubGlobal("requestAnimationFrame", rafSpy);
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function renderTileCanvas(jobIdSuffix = "") {
    return render(
      createElement(TileCanvas, {
        jobId: `j${jobIdSuffix}`,
        jobStart: 0,
        jobEnd: 1000,
        centerTimestamp: 100,
        zoomLevel: "5m",
        freqRange: [0, 3000] as [number, number],
        width: 800,
        height: 200,
        tileDurationOverride: 60,
        viewportSpanOverride: 300,
        tileUrlBuilder: () => `tile-${jobIdSuffix}-${Math.random()}`,
      }),
    );
  }

  it("schedules a redraw on a tile-loaded notification while mounted", () => {
    const view = renderTileCanvas("-mount");
    rafSpy.mockClear();
    rafCallbacks.length = 0;

    __notifyTileLoadedForTest();
    expect(rafSpy).toHaveBeenCalledTimes(1);

    view.unmount();
  });

  it("does not schedule a redraw after unmount (subscription removed)", () => {
    const view = renderTileCanvas("-unmount");
    view.unmount();

    rafSpy.mockClear();
    rafCallbacks.length = 0;

    __notifyTileLoadedForTest();
    expect(rafSpy).not.toHaveBeenCalled();
  });

  it("two notifications in the same frame schedule exactly one redraw", () => {
    const view = renderTileCanvas("-dedup");
    rafSpy.mockClear();
    rafCallbacks.length = 0;

    __notifyTileLoadedForTest();
    __notifyTileLoadedForTest();
    expect(rafSpy).toHaveBeenCalledTimes(1);

    rafCallbacks[0]();
    __notifyTileLoadedForTest();
    expect(rafSpy).toHaveBeenCalledTimes(2);

    view.unmount();
  });
});

describe("loader Image.onerror does not notify subscribers", () => {
  it("captures Image, drives onerror, and verifies subscribers are not called", () => {
    type FakeImage = {
      crossOrigin?: string;
      onload?: () => void;
      onerror?: () => void;
      src?: string;
    };
    let last: FakeImage | null = null;
    const ImageStub = vi.fn(function (this: FakeImage) {
      last = this;
    }) as unknown as typeof Image;
    vi.stubGlobal("Image", ImageStub);

    const cb = vi.fn();
    const unsub = subscribeTileLoaded(cb);

    render(
      createElement(TileCanvas, {
        jobId: "err",
        jobStart: 0,
        jobEnd: 1000,
        centerTimestamp: 100,
        zoomLevel: "5m",
        freqRange: [0, 3000] as [number, number],
        width: 800,
        height: 200,
        tileDurationOverride: 60,
        viewportSpanOverride: 300,
        tileUrlBuilder: () => `err-${Math.random()}`,
      }),
    );

    expect(last).not.toBeNull();
    cb.mockClear();

    last!.onerror?.();
    expect(cb).not.toHaveBeenCalled();

    unsub();
    vi.unstubAllGlobals();
  });
});
