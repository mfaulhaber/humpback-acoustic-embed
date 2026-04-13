import { useCallback, useRef, useState } from "react";
import { regionAudioSliceUrl } from "@/api/client";
import { formatTimeDecimal } from "@/utils/format";
import type { EffectiveEvent } from "./EventBarOverlay";

interface EventDetailPanelProps {
  event: EffectiveEvent | null;
  regionJobId: string | null;
  onDelete: (eventId: string) => void;
}

export function EventDetailPanel({
  event,
  regionJobId,
  onDelete,
}: EventDetailPanelProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const handlePlaySlice = useCallback(() => {
    if (!event || !regionJobId) return;
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      audio.currentTime = 0;
      setIsPlaying(false);
      return;
    }

    const duration = event.endSec - event.startSec;
    audio.src = regionAudioSliceUrl(regionJobId, event.startSec, duration);
    audio.play().catch(() => setIsPlaying(false));
    setIsPlaying(true);
  }, [event, regionJobId, isPlaying]);

  if (!event) {
    return (
      <div className="border-t px-4 py-3 text-sm text-muted-foreground">
        Click an event bar to view details
      </div>
    );
  }

  const duration = event.endSec - event.startSec;
  const isAdjusted = event.correctionType === "adjust";
  const isDeleted = event.correctionType === "delete";
  const isAdded = event.correctionType === "add";

  return (
    <div className="border-t px-4 py-3">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1 text-sm">
          <div className="flex items-center gap-2">
            <span className="font-medium">Event</span>
            {isAdjusted && (
              <span className="rounded bg-purple-500/20 px-1.5 py-0.5 text-xs text-purple-300">
                adjusted
              </span>
            )}
            {isDeleted && (
              <span className="rounded bg-red-500/20 px-1.5 py-0.5 text-xs text-red-300">
                deleted
              </span>
            )}
            {isAdded && (
              <span className="rounded bg-green-500/20 px-1.5 py-0.5 text-xs text-green-300">
                added
              </span>
            )}
          </div>

          <div className="grid grid-cols-3 gap-x-6 gap-y-0.5 text-xs text-muted-foreground">
            <div>
              <span className="text-foreground">Start:</span>{" "}
              {formatTimeDecimal(event.startSec)}
              {isAdjusted && (
                <span className="ml-1 text-purple-400">
                  (was {formatTimeDecimal(event.originalStartSec)})
                </span>
              )}
            </div>
            <div>
              <span className="text-foreground">End:</span>{" "}
              {formatTimeDecimal(event.endSec)}
              {isAdjusted && (
                <span className="ml-1 text-purple-400">
                  (was {formatTimeDecimal(event.originalEndSec)})
                </span>
              )}
            </div>
            <div>
              <span className="text-foreground">Duration:</span>{" "}
              {duration.toFixed(1)}s
            </div>
            <div>
              <span className="text-foreground">Confidence:</span>{" "}
              {event.confidence > 0 ? event.confidence.toFixed(3) : "—"}
            </div>
          </div>
        </div>

        <div className="flex shrink-0 gap-2">
          <button
            className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent"
            onClick={handlePlaySlice}
            disabled={!regionJobId}
          >
            {isPlaying ? "Stop" : "Play Slice"}
          </button>
          <button
            className={
              isDeleted
                ? "rounded-md border border-green-500/50 px-3 py-1.5 text-xs text-green-400 hover:bg-green-500/10"
                : "rounded-md border border-red-500/50 px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10"
            }
            onClick={() => onDelete(event.eventId)}
          >
            {isDeleted ? "Undo Delete" : "Delete Event"}
          </button>
        </div>
      </div>

      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        style={{ display: "none" }}
      />
    </div>
  );
}
