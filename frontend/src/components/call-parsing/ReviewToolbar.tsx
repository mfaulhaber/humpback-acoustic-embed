import { useCallback } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import type { Region } from "@/api/types";
import { formatRecordingTime } from "@/utils/format";

export interface RetrainStatus {
  status: "queued" | "running" | "complete" | "failed";
  modelId?: string;
  modelName?: string;
  error?: string;
}

interface ReviewToolbarProps {
  region: Region | null;
  eventCount: number;
  pendingChangeCount: number;
  isDirty: boolean;
  addMode: boolean;
  onToggleAddMode: () => void;
  onSave: () => void;
  onCancel: () => void;
  isPlaying: boolean;
  onPlay: () => void;
  hasCorrections: boolean;
  onRetrain: () => void;
  retrainStatus: RetrainStatus | null;
  onResegment: () => void;
  regions: Region[];
  selectedRegionId: string | null;
  onPrevRegion: () => void;
  onNextRegion: () => void;
  onPrevEvent?: () => void;
  onNextEvent?: () => void;
  currentEventIndex?: number;
  totalEventCount?: number;
  jobStartEpoch: number;
}

export function ReviewToolbar({
  region,
  eventCount,
  pendingChangeCount,
  isDirty,
  addMode,
  onToggleAddMode,
  onSave,
  onCancel,
  isPlaying,
  onPlay,
  hasCorrections,
  onRetrain,
  retrainStatus,
  onResegment,
  regions,
  selectedRegionId,
  onPrevRegion,
  onNextRegion,
  onPrevEvent,
  onNextEvent,
  currentEventIndex = 0,
  totalEventCount = 0,
  jobStartEpoch,
}: ReviewToolbarProps) {
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

  const regionIdx = regions.findIndex((r) => r.region_id === selectedRegionId);
  const hasPrev = regionIdx > 0;
  const hasNext = regionIdx >= 0 && regionIdx < regions.length - 1;

  const isTraining =
    retrainStatus?.status === "queued" ||
    retrainStatus?.status === "running";
  const trainingComplete = retrainStatus?.status === "complete";
  const trainingFailed = retrainStatus?.status === "failed";
  const canRetrain = hasCorrections && !isDirty && !isTraining;

  return (
    <div className="flex items-center justify-between border-b px-4 py-2">
      <div className="flex items-center gap-2 text-sm">
        <button
          className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
          disabled={!hasPrev}
          onClick={onPrevRegion}
          title="Previous region"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button
          className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
          disabled={!hasNext}
          onClick={onNextRegion}
          title="Next region"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
        <span className="text-muted-foreground ml-1">
          Region {regionIdx >= 0 ? regionIdx + 1 : "?"}/{regions.length} · {formatRecordingTime(region.start_sec, jobStartEpoch)} – {formatRecordingTime(region.end_sec, jobStartEpoch)}
        </span>

        {onPrevEvent && onNextEvent && (
          <>
            <div className="mx-1 h-4 w-px bg-border" />
            <button
              className="rounded-md border px-1.5 py-0.5 text-[10px] hover:bg-accent disabled:opacity-30"
              disabled={currentEventIndex <= 0}
              onClick={onPrevEvent}
              title="Previous event (A)"
            >
              <ChevronLeft className="h-3 w-3" />
            </button>
            <span className="text-xs text-muted-foreground">
              Event {totalEventCount > 0 ? currentEventIndex + 1 : 0}/{totalEventCount}
            </span>
            <button
              className="rounded-md border px-1.5 py-0.5 text-[10px] hover:bg-accent disabled:opacity-30"
              disabled={currentEventIndex >= totalEventCount - 1}
              onClick={onNextEvent}
              title="Next event (D)"
            >
              <ChevronRight className="h-3 w-3" />
            </button>
          </>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent"
          onClick={onPlay}
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

        <div className="mx-1 h-5 w-px bg-border" />

        {/* Training status indicators */}
        {isTraining && (
          <span className="flex items-center gap-1.5 text-xs text-yellow-500">
            <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-yellow-500 border-t-transparent" />
            Training...
          </span>
        )}

        {trainingComplete && (
          <>
            <span className="text-xs text-green-500">Model ready</span>
            <button
              className="rounded-md border border-green-500 bg-green-500/15 px-3 py-1.5 text-xs text-green-400 hover:bg-green-500/25"
              onClick={onResegment}
            >
              Re-segment
            </button>
          </>
        )}

        {trainingFailed && (
          <span
            className="text-xs text-red-500"
            title={retrainStatus?.error ?? "Training failed"}
          >
            Training failed
          </span>
        )}

        {/* Retrain button — hidden while training is in progress */}
        {!isTraining && (
          <button
            className="rounded-md border border-blue-500 bg-blue-500/15 px-3 py-1.5 text-xs text-blue-400 hover:bg-blue-500/25 disabled:border-border disabled:bg-transparent disabled:text-muted-foreground disabled:opacity-40"
            disabled={!canRetrain}
            onClick={onRetrain}
            title={
              !hasCorrections
                ? "Save boundary corrections first"
                : isDirty
                  ? "Save pending changes first"
                  : "Train a new model from corrections"
            }
          >
            Retrain
          </button>
        )}
      </div>
    </div>
  );
}
