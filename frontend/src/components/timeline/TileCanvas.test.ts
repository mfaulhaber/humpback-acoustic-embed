import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __notifyTileLoadedForTest,
  subscribeTileLoaded,
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
