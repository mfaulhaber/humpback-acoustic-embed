import { useState } from "react";
import {
  useConvergenceMetrics,
  useStartActiveLearningCycle,
} from "@/hooks/queries/useLabeling";
import { RefreshCw } from "lucide-react";

interface Props {
  selectedModelId: string;
  selectedJobId: string;
  onCycleStarted: () => void;
}

export function ActiveLearningDashboard({
  selectedModelId,
  selectedJobId,
  onCycleStarted,
}: Props) {
  const convergenceQuery = useConvergenceMetrics(selectedModelId);
  const startCycle = useStartActiveLearningCycle();
  const [cycleName, setCycleName] = useState("");

  const metrics = convergenceQuery.data;

  const handleStartCycle = () => {
    const name = cycleName.trim() || `cycle-${(metrics?.cycles_completed ?? 0) + 1}`;
    startCycle.mutate(
      {
        vocalization_model_id: selectedModelId,
        detection_job_ids: [selectedJobId],
        name,
      },
      {
        onSuccess: () => {
          setCycleName("");
          onCycleStarted();
        },
      },
    );
  };

  if (convergenceQuery.isLoading) {
    return <div className="text-xs text-slate-400 p-2">Loading metrics...</div>;
  }

  return (
    <div className="space-y-3">
      {/* Cycle count + accuracy trend */}
      <div className="flex items-center justify-between">
        <div className="text-xs text-slate-500">
          <strong>{metrics?.cycles_completed ?? 0}</strong> training cycle(s)
        </div>
        {(metrics?.accuracy_trend ?? []).length > 0 && (
          <div className="text-xs text-slate-500">
            Latest accuracy:{" "}
            <strong>
              {(
                (metrics!.accuracy_trend[metrics!.accuracy_trend.length - 1] ?? 0) *
                100
              ).toFixed(1)}
              %
            </strong>
          </div>
        )}
      </div>

      {/* Accuracy trend sparkline */}
      {(metrics?.accuracy_trend ?? []).length > 1 && (
        <div className="border rounded p-2 bg-white">
          <div className="text-[10px] text-slate-400 mb-1">
            Accuracy Trend
          </div>
          <div className="flex items-end gap-1 h-10">
            {metrics!.accuracy_trend.map((acc, i) => (
              <div
                key={i}
                className="flex-1 bg-emerald-400 rounded-t"
                style={{ height: `${acc * 100}%` }}
                title={`Cycle ${i + 1}: ${(acc * 100).toFixed(1)}%`}
              />
            ))}
          </div>
        </div>
      )}

      {/* Label distribution */}
      {metrics && Object.keys(metrics.label_distribution).length > 0 && (
        <div className="border rounded p-2 bg-white">
          <div className="text-[10px] text-slate-400 mb-1">
            Label Distribution
          </div>
          <div className="space-y-1">
            {Object.entries(metrics.label_distribution)
              .sort(([, a], [, b]) => b - a)
              .map(([label, count]) => {
                const total = Object.values(
                  metrics.label_distribution,
                ).reduce((s, c) => s + c, 0);
                return (
                  <div
                    key={label}
                    className="flex items-center gap-2 text-xs"
                  >
                    <span className="w-16 text-slate-600 truncate">
                      {label}
                    </span>
                    <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-violet-400 rounded-full"
                        style={{
                          width: `${total > 0 ? (count / total) * 100 : 0}%`,
                        }}
                      />
                    </div>
                    <span className="w-8 text-right text-slate-500 font-mono">
                      {count}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Per-class F1 */}
      {metrics &&
        metrics.uncertainty_histogram.length > 0 && (
          <div className="border rounded p-2 bg-white">
            <div className="text-[10px] text-slate-400 mb-1">
              Per-Class F1
            </div>
            <div className="space-y-1">
              {metrics.uncertainty_histogram.map((entry) => {
                const cls = String(entry.class ?? "");
                const f1 = Number(entry.f1 ?? 0);
                return (
                  <div
                    key={cls}
                    className="flex items-center gap-2 text-xs"
                  >
                    <span className="w-16 text-slate-600 truncate">
                      {cls}
                    </span>
                    <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          f1 > 0.8
                            ? "bg-emerald-400"
                            : f1 > 0.5
                              ? "bg-amber-400"
                              : "bg-red-400"
                        }`}
                        style={{ width: `${f1 * 100}%` }}
                      />
                    </div>
                    <span className="w-10 text-right text-slate-500 font-mono">
                      {(f1 * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

      {/* Start next cycle */}
      <div className="border rounded p-2 bg-white">
        <div className="text-[10px] text-slate-400 mb-1">
          Start Next Cycle
        </div>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={cycleName}
            onChange={(e) => setCycleName(e.target.value)}
            placeholder={`cycle-${(metrics?.cycles_completed ?? 0) + 1}`}
            className="flex-1 border rounded px-2 py-1 text-xs"
          />
          <button
            onClick={handleStartCycle}
            disabled={startCycle.isPending}
            className="flex items-center gap-1 px-2 py-1 bg-emerald-500 text-white rounded text-xs hover:bg-emerald-600 disabled:opacity-50"
          >
            <RefreshCw
              className={`h-3 w-3 ${startCycle.isPending ? "animate-spin" : ""}`}
            />
            Retrain
          </button>
        </div>
        {startCycle.isSuccess && (
          <div className="text-[10px] text-emerald-600 mt-1">
            Training job queued
          </div>
        )}
        {startCycle.isError && (
          <div className="text-[10px] text-red-600 mt-1">
            {(startCycle.error as Error).message}
          </div>
        )}
      </div>
    </div>
  );
}
