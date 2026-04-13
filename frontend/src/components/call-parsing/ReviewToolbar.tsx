import { useCallback, useRef, useState } from "react";
import { regionAudioSliceUrl } from "@/api/client";
import type { Region } from "@/api/types";
import { formatTime } from "@/utils/format";

interface ReviewToolbarProps {
  region: Region | null;
  regionJobId: string | null;
  eventCount: number;
  pendingChangeCount: number;
  isDirty: boolean;
  addMode: boolean;
  onToggleAddMode: () => void;
  onSave: () => void;
  onCancel: () => void;
  /** Current viewport start time in seconds (for Play from pan position). */
  viewStart?: number;
}

export function ReviewToolbar({
  region,
  regionJobId,
  eventCount,
  pendingChangeCount,
  isDirty,
  addMode,
  onToggleAddMode,
  onSave,
  onCancel,
  viewStart,
}: ReviewToolbarProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const handlePlay = useCallback(() => {
    if (!region || !regionJobId) return;
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      audio.currentTime = 0;
      setIsPlaying(false);
      return;
    }

    const playStart = viewStart ?? region.padded_start_sec;
    const duration = Math.min(region.padded_end_sec - playStart, 30);
    audio.src = regionAudioSliceUrl(regionJobId, playStart, duration);
    audio.play().catch(() => setIsPlaying(false));
    setIsPlaying(true);
  }, [region, regionJobId, isPlaying, viewStart]);

  const handleCancel = useCallback(() => {
    if (isDirty) {
      const ok = window.confirm(
        `Discard ${pendingChangeCount} unsaved change${pendingChangeCount !== 1 ? "s" : ""}?`,
      );
      if (!ok) return;
    }
    onCancel();
  }, [isDirty, pendingChangeCount, onCancel]);

  if (!region) return null;

  return (
    <div className="flex items-center justify-between border-b px-4 py-2">
      <div className="flex items-center gap-3 text-sm">
        <span className="text-muted-foreground">
          Region {formatTime(region.start_sec)} – {formatTime(region.end_sec)}
        </span>
        <span className="text-xs text-muted-foreground">
          {eventCount} event{eventCount !== 1 ? "s" : ""}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <button
          className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent"
          onClick={handlePlay}
          disabled={!regionJobId}
        >
          {isPlaying ? "Stop" : "Play"}
        </button>

        <button
          className={
            addMode
              ? "rounded-md border border-green-500 bg-green-500/20 px-3 py-1.5 text-xs text-green-300"
              : "rounded-md border px-3 py-1.5 text-xs hover:bg-accent"
          }
          onClick={onToggleAddMode}
        >
          + Add
        </button>

        <button
          className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent disabled:opacity-40"
          disabled={!isDirty}
          onClick={onSave}
        >
          Save
          {isDirty && (
            <span className="ml-1.5 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-yellow-500 px-1 text-[10px] font-medium text-black">
              {pendingChangeCount}
            </span>
          )}
        </button>

        <button
          className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent"
          onClick={handleCancel}
        >
          Cancel
        </button>

        <button
          className="rounded-md border px-3 py-1.5 text-xs opacity-40"
          disabled
          title="Retrain model (coming soon)"
        >
          Retrain
        </button>
      </div>

      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        style={{ display: "none" }}
      />
    </div>
  );
}

