import { Badge } from "@/components/ui/badge";
import { typeColor } from "./TypePalette";

export interface AggregatedEvent {
  eventId: string;
  regionId: string;
  startSec: number;
  endSec: number;
  predictedType: string | null;
  predictedScore: number | null;
  correctedType: string | null | undefined; // undefined = no correction
  allScores: { type_name: string; score: number; above_threshold: boolean }[];
}

function formatDuration(sec: number): string {
  return sec < 1 ? `${(sec * 1000).toFixed(0)}ms` : `${sec.toFixed(2)}s`;
}

interface ClassifyDetailPanelProps {
  event: AggregatedEvent | null;
}

export function ClassifyDetailPanel({ event }: ClassifyDetailPanelProps) {
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

  return (
    <div className="px-4 py-3 border-t space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Type:</span>
          {isNegative ? (
            <Badge variant="outline" className="text-xs">
              (Negative)
            </Badge>
          ) : displayType ? (
            <Badge
              className="text-xs text-white"
              style={{ backgroundColor: typeColor(displayType) }}
            >
              {displayType}
            </Badge>
          ) : (
            <span className="text-xs text-muted-foreground">none</span>
          )}
          {hasCorrected && (
            <span className="text-xs text-green-600 font-medium">
              corrected
            </span>
          )}
          {!hasCorrected && event.predictedScore != null && (
            <span className="text-xs text-muted-foreground">
              score: {event.predictedScore.toFixed(3)}
            </span>
          )}
        </div>
        <div className="text-xs text-muted-foreground">
          {event.startSec.toFixed(2)}s – {event.endSec.toFixed(2)}s (
          {formatDuration(duration)})
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
