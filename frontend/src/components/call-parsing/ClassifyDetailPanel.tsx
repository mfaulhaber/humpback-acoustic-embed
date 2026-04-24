import { Badge } from "@/components/ui/badge";
import { typeColor } from "./TypePalette";
import { formatRecordingTime } from "@/utils/format";
import {
  APPROVED_RING_COLOR,
  CORRECTED_RING_COLOR,
} from "@/components/timeline/overlays/EventBarOverlay";

export interface AggregatedEvent {
  eventId: string;
  regionId: string;
  startSec: number;
  endSec: number;
  predictedType: string | null;
  predictedScore: number | null;
  correctedType: string | null | undefined; // undefined = no correction
  allScores: { type_name: string; score: number; above_threshold: boolean }[];
  correctionType?: "adjust" | "add" | "delete" | null;
  originalStartSec?: number;
  originalEndSec?: number;
}

function formatDuration(sec: number): string {
  return sec < 1 ? `${(sec * 1000).toFixed(0)}ms` : `${sec.toFixed(2)}s`;
}

interface ClassifyDetailPanelProps {
  event: AggregatedEvent | null;
  jobStartEpoch: number;
}

export function ClassifyDetailPanel({
  event,
  jobStartEpoch,
}: ClassifyDetailPanelProps) {
  if (!event) {
    return (
      <div className="px-4 py-3 text-sm text-muted-foreground border-t">
        No event selected
      </div>
    );
  }

  const duration = event.endSec - event.startSec;
  const hasCorrected = event.correctedType !== undefined;
  const displayType = hasCorrected ? event.correctedType : event.predictedType;
  const isNegative = hasCorrected && event.correctedType === null;
  const isApproved =
    hasCorrected &&
    typeof event.correctedType === "string" &&
    event.correctedType === event.predictedType;
  const typeSource = isNegative
    ? "negative"
    : isApproved
      ? "approved"
      : hasCorrected
        ? "correction"
        : null;

  const ringColor =
    typeSource === "approved"
      ? APPROVED_RING_COLOR
      : typeSource === "correction"
        ? CORRECTED_RING_COLOR
        : undefined;

  const isAdjusted = event.correctionType === "adjust";
  const isDeleted = event.correctionType === "delete";
  const isAdded = event.correctionType === "add";

  return (
    <div className="px-4 py-3 border-t space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Type:</span>
          {isNegative ? (
            <Badge
              variant="outline"
              className="text-xs text-red-500 border-red-400"
            >
              (Negative)
            </Badge>
          ) : displayType ? (
            <Badge
              className="text-xs text-white"
              style={{
                backgroundColor: typeColor(displayType),
                boxShadow: ringColor
                  ? `0 0 0 1.5px ${ringColor}`
                  : undefined,
              }}
            >
              {displayType}
            </Badge>
          ) : (
            <span className="text-xs text-muted-foreground">none</span>
          )}
          {typeSource === "approved" && (
            <span
              className="text-xs font-medium"
              style={{ color: APPROVED_RING_COLOR }}
            >
              approved
            </span>
          )}
          {typeSource === "correction" && (
            <span
              className="text-xs font-medium"
              style={{ color: CORRECTED_RING_COLOR }}
            >
              corrected
            </span>
          )}
          {!hasCorrected && event.predictedScore != null && (
            <span className="text-xs text-muted-foreground">
              score: {event.predictedScore.toFixed(3)}
            </span>
          )}
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
        <div className="text-xs text-muted-foreground">
          {formatRecordingTime(event.startSec, jobStartEpoch)} –{" "}
          {formatRecordingTime(event.endSec, jobStartEpoch)} (
          {formatDuration(duration)})
          {isAdjusted &&
            event.originalStartSec != null &&
            event.originalEndSec != null && (
              <span className="ml-1 text-purple-400">
                was {formatRecordingTime(event.originalStartSec, jobStartEpoch)}{" "}
                – {formatRecordingTime(event.originalEndSec, jobStartEpoch)}
              </span>
            )}
        </div>
      </div>

      {/* All model scores */}
      {event.allScores.length > 0 && (
        <div className="text-xs">
          <span className="text-muted-foreground font-medium">
            All scores:
          </span>
          <div className="mt-1 grid grid-cols-3 gap-x-4 gap-y-0.5">
            {event.allScores.map((s) => (
              <div key={s.type_name} className="flex items-center gap-1">
                <span
                  className="w-2 h-2 rounded-full inline-block"
                  style={{ backgroundColor: typeColor(s.type_name) }}
                />
                <span className={s.above_threshold ? "font-medium" : ""}>
                  {s.type_name}
                </span>
                <span className="text-muted-foreground ml-auto">
                  {s.score.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
