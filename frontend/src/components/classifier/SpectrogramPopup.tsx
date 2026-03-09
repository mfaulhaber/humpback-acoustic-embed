import { useState } from "react";
import { Loader2 } from "lucide-react";

interface SpectrogramPopupProps {
  imageUrl: string;
  position: { x: number; y: number };
  onClose: () => void;
}

export function SpectrogramPopup({ imageUrl, position, onClose }: SpectrogramPopupProps) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

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
      >
        {!loaded && !error && (
          <div
            className="flex items-center justify-center"
            style={{ width: popupWidth, height: popupHeight }}
          >
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        )}
        {error && (
          <div
            className="flex items-center justify-center text-sm text-destructive"
            style={{ width: popupWidth, height: popupHeight }}
          >
            Failed to load spectrogram
          </div>
        )}
        <img
          src={imageUrl}
          alt="Spectrogram"
          className={loaded ? "block" : "hidden"}
          onLoad={() => setLoaded(true)}
          onError={() => setError(true)}
        />
      </div>
    </div>
  );
}
