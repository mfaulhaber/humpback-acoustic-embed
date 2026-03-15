import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

type SpectrogramMarkerBounds = {
  startSec: number;
  endSec: number;
};

interface SpectrogramPopupProps {
  imageUrl: string;
  position: { x: number; y: number };
  markerBounds?: SpectrogramMarkerBounds | null;
  durationSec?: number;
  onClose: () => void;
}

const SPECTROGRAM_PLOT_BOUNDS = {
  left: 0.11,
  right: 0.985,
  bottom: 0.15,
  top: 0.97,
} as const;

export function SpectrogramPopup({
  imageUrl,
  position,
  markerBounds = null,
  durationSec,
  onClose,
}: SpectrogramPopupProps) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const imageRef = useRef<HTMLImageElement>(null);

  const syncImageSize = useCallback(() => {
    if (!imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    setImageSize({
      width: rect.width,
      height: rect.height,
    });
  }, []);

  useEffect(() => {
    setLoaded(false);
    setError(false);
    setImageSize({ width: 0, height: 0 });
  }, [imageUrl]);

  useEffect(() => {
    if (!loaded || !imageRef.current || typeof ResizeObserver === "undefined") {
      return;
    }
    syncImageSize();
    const observer = new ResizeObserver(() => syncImageSize());
    observer.observe(imageRef.current);
    return () => observer.disconnect();
  }, [loaded, syncImageSize]);

  const markerPositions = useMemo(() => {
    if (
      !markerBounds ||
      typeof durationSec !== "number" ||
      durationSec <= 0 ||
      imageSize.width <= 0 ||
      imageSize.height <= 0
    ) {
      return null;
    }

    const { startSec, endSec } = markerBounds;
    if (startSec < 0 || endSec > durationSec || endSec <= startSec) {
      return null;
    }

    const plotLeft = imageSize.width * SPECTROGRAM_PLOT_BOUNDS.left;
    const plotRight = imageSize.width * SPECTROGRAM_PLOT_BOUNDS.right;
    const plotTop = imageSize.height * (1 - SPECTROGRAM_PLOT_BOUNDS.top);
    const plotBottom = imageSize.height * (1 - SPECTROGRAM_PLOT_BOUNDS.bottom);
    const plotWidth = plotRight - plotLeft;

    return {
      startX: plotLeft + (startSec / durationSec) * plotWidth,
      endX: plotLeft + (endSec / durationSec) * plotWidth,
      y1: plotTop,
      y2: plotBottom,
    };
  }, [durationSec, imageSize.height, imageSize.width, markerBounds]);

  // Viewport-aware positioning: flip if near right/bottom edges
  const margin = 16;
  const popupWidth = 660;
  const popupHeight = 360;
  const left =
    position.x + popupWidth + margin > window.innerWidth
      ? Math.max(margin, position.x - popupWidth - margin)
      : position.x + margin;
  const top =
    position.y + popupHeight + margin > window.innerHeight
      ? Math.max(margin, position.y - popupHeight - margin)
      : position.y + margin;

  return (
    <div className="fixed inset-0 z-50" onClick={onClose}>
      <div
        className="absolute bg-background border rounded-lg shadow-xl p-2"
        style={{ left, top }}
        onClick={(e) => e.stopPropagation()}
        data-testid="spectrogram-popup"
      >
        <div
          className="relative inline-block"
          style={loaded && !error ? undefined : { width: popupWidth, height: popupHeight }}
        >
          <img
            ref={imageRef}
            src={imageUrl}
            alt="Spectrogram"
            className={loaded && !error ? "block" : "hidden"}
            data-testid="spectrogram-image"
            onLoad={() => {
              setLoaded(true);
              syncImageSize();
            }}
            onError={() => {
              setError(true);
              setLoaded(false);
            }}
          />
          {markerPositions && (
            <svg
              className="absolute inset-0 pointer-events-none"
              width={imageSize.width}
              height={imageSize.height}
              viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
              data-testid="spectrogram-overlay"
            >
              <line
                x1={markerPositions.startX}
                x2={markerPositions.startX}
                y1={markerPositions.y1}
                y2={markerPositions.y2}
                stroke="#000000"
                strokeWidth="2"
                data-testid="spectrogram-marker-start"
              />
              <line
                x1={markerPositions.endX}
                x2={markerPositions.endX}
                y1={markerPositions.y1}
                y2={markerPositions.y2}
                stroke="#000000"
                strokeWidth="2"
                data-testid="spectrogram-marker-end"
              />
            </svg>
          )}
        </div>
        {!loaded && !error && (
          <div className="absolute inset-2 flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        )}
        {error && (
          <div className="absolute inset-2 flex items-center justify-center text-sm text-destructive">
            Failed to load spectrogram
          </div>
        )}
      </div>
    </div>
  );
}
