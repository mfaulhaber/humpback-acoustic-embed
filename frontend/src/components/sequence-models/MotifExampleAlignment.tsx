import { Button } from "@/components/ui/button";
import { regionAudioSliceUrl } from "@/api/client";
import { type MotifOccurrence } from "@/api/sequenceModels";
import { labelColor } from "./constants";

function fmt(sec: number): string {
  return `${sec >= 0 ? "+" : ""}${sec.toFixed(2)}s`;
}

export function MotifExampleAlignment({
  occurrences,
  regionDetectionJobId,
  onJumpToTimestamp,
  activeOccurrenceIndex,
  onActiveOccurrenceChange,
  numLabels,
}: {
  occurrences: MotifOccurrence[];
  regionDetectionJobId: string;
  onJumpToTimestamp: (timestamp: number) => void;
  activeOccurrenceIndex?: number;
  onActiveOccurrenceChange?: (idx: number) => void;
  /** Total label count (HMM ``n_states`` or masked-transformer ``k``).
   *  Required to match the main timeline's coloring — the categorical
   *  palette wraps at 20 entries, so for >20 labels we need the HSL
   *  ramp ``labelColor`` provides.
   */
  numLabels?: number;
}) {
  if (occurrences.length === 0) {
    return (
      <div className="text-sm text-slate-500" data-testid="motif-examples-empty">
        No examples.
      </div>
    );
  }

  const rows = occurrences.slice(0, 10);
  const minRel = Math.min(...rows.map((o) => o.relative_start_seconds), -1);
  const maxRel = Math.max(...rows.map((o) => o.relative_end_seconds), 1);
  const span = Math.max(0.001, maxRel - minRel);
  const zeroPct = ((0 - minRel) / span) * 100;

  return (
    <div className="space-y-2" data-testid="motif-example-alignment">
      <div className="relative h-5 border-b border-slate-200 text-xs text-slate-500">
        <div
          className="absolute top-0 h-5 border-l border-red-500"
          style={{ left: `${zeroPct}%` }}
        />
        <span className="absolute left-0">{fmt(minRel)}</span>
        <span
          className="absolute -translate-x-1/2 text-red-700"
          style={{ left: `${zeroPct}%` }}
        >
          0
        </span>
        <span className="absolute right-0">{fmt(maxRel)}</span>
      </div>
      {rows.map((occ, idx) => {
        const left = ((occ.relative_start_seconds - minRel) / span) * 100;
        const width =
          ((occ.relative_end_seconds - occ.relative_start_seconds) / span) * 100;
        const isActive = activeOccurrenceIndex === idx;
        return (
          <div
            key={occ.occurrence_id}
            data-testid={`motif-example-row-${idx}`}
            data-active={isActive ? "true" : "false"}
            className={`grid grid-cols-[180px_1fr_150px] gap-3 items-center text-xs rounded ${
              isActive ? "bg-blue-50 ring-1 ring-blue-300" : ""
            }`}
          >
            <div className="truncate">
              {occ.event_source_key} / {occ.anchor_strategy}
            </div>
            <div className="relative h-7 rounded border border-slate-200 bg-slate-50">
              <div
                className="absolute top-0 h-full border-l border-red-400"
                style={{ left: `${zeroPct}%` }}
              />
              <div
                className="absolute top-1 h-5 overflow-hidden rounded-sm border border-white/80"
                style={{ left: `${left}%`, width: `${Math.max(width, 1.5)}%` }}
              >
                <div className="flex h-full">
                  {occ.states.map((state, idx) => (
                    <div
                      key={`${occ.occurrence_id}-${idx}`}
                      className="h-full flex-1"
                      title={`State ${state}`}
                      style={{
                        background: labelColor(
                          state,
                          Math.max(numLabels ?? 1, 1),
                        ),
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
            <div className="flex justify-end gap-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  const start = Math.max(0, occ.start_timestamp - 1);
                  const duration = Math.max(
                    1,
                    occ.end_timestamp - occ.start_timestamp + 2,
                  );
                  new Audio(
                    regionAudioSliceUrl(regionDetectionJobId, start, duration),
                  ).play();
                }}
              >
                Play
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  onActiveOccurrenceChange?.(idx);
                  onJumpToTimestamp(
                    (occ.start_timestamp + occ.end_timestamp) / 2,
                  );
                }}
              >
                Jump
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
