import { formatRecordingTime } from "@/utils/format";
import { DeleteConfirmButton } from "@/components/shared/DeleteConfirmationDialog";
import type { EffectiveEvent } from "@/components/timeline/overlays/EventBarOverlay";

interface EventDetailPanelProps {
  event: EffectiveEvent | null;
  onDelete: (eventId: string) => void;
  isPlaying: boolean;
  onPlaySlice: () => void;
  jobStartEpoch: number;
}

export function EventDetailPanel({
  event,
  onDelete,
  isPlaying,
  onPlaySlice,
  jobStartEpoch,
}: EventDetailPanelProps) {
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
              {formatRecordingTime(event.startSec, jobStartEpoch)}
              {isAdjusted && (
                <span className="ml-1 text-purple-400">
                  (was {formatRecordingTime(event.originalStartSec, jobStartEpoch)})
                </span>
              )}
            </div>
            <div>
              <span className="text-foreground">End:</span>{" "}
              {formatRecordingTime(event.endSec, jobStartEpoch)}
              {isAdjusted && (
                <span className="ml-1 text-purple-400">
                  (was {formatRecordingTime(event.originalEndSec, jobStartEpoch)})
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
            onClick={onPlaySlice}
          >
            {isPlaying ? "Stop" : "Play Slice"}
          </button>
          {isDeleted ? (
            <button
              className="rounded-md border border-green-500/50 px-3 py-1.5 text-xs text-green-400 hover:bg-green-500/10"
              onClick={() => onDelete(event.eventId)}
            >
              Undo Delete
            </button>
          ) : (
            <DeleteConfirmButton
              size="sm"
              className="h-8 px-3 text-xs"
              resourceType="event"
              resourceName={event.eventId}
              confirmationText={
                <>
                  Mark event{" "}
                  <strong className="font-semibold text-foreground">
                    {event.eventId}
                  </strong>{" "}
                  for deletion?
                </>
              }
              consequence="This event will be marked as deleted in pending corrections. Save the corrections to apply it to effective event readers."
              onConfirm={() => onDelete(event.eventId)}
            >
              Delete Event
            </DeleteConfirmButton>
          )}
        </div>
      </div>
    </div>
  );
}
