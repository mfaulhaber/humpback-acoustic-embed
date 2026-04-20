import { createContext, useContext } from "react";

export interface OverlayContextValue {
  viewStart: number;
  viewEnd: number;
  pxPerSec: number;
  canvasWidth: number;
  canvasHeight: number;
  epochToX: (epoch: number) => number;
  xToEpoch: (x: number) => number;
}

export const OverlayContext = createContext<OverlayContextValue | null>(null);

export function useOverlayContext(): OverlayContextValue {
  const ctx = useContext(OverlayContext);
  if (!ctx) {
    throw new Error("useOverlayContext must be used within a Spectrogram component");
  }
  return ctx;
}
