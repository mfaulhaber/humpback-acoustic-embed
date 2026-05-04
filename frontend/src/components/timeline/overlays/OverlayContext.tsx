import { createContext, useContext } from "react";

export interface OverlayContextValue {
  viewStart: number;
  viewEnd: number;
  pxPerSec: number;
  canvasWidth: number;
  canvasHeight: number;
  epochToX: (epoch: number) => number;
  xToEpoch: (x: number) => number;
  /**
   * Mount target for overlay elements that must remain visible past the
   * canvas edge (e.g. tooltips). Sibling of the clipped band layer with
   * no overflow clipping. Null until the layer's ref attaches, and null
   * when the overlay renders outside a Spectrogram (back-compat).
   */
  tooltipPortalTarget: HTMLElement | null;
}

export const OverlayContext = createContext<OverlayContextValue | null>(null);

export function useOverlayContext(): OverlayContextValue {
  const ctx = useContext(OverlayContext);
  if (!ctx) {
    throw new Error("useOverlayContext must be used within a Spectrogram component");
  }
  return ctx;
}
