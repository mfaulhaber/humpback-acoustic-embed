import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Plot from "react-plotly.js";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  type HMMSequenceJobDetail,
  isHMMSequenceJobActive,
  useCancelHMMSequenceJob,
  useHMMDwell,
  useHMMSequenceJob,
  useHMMStates,
  useHMMTransitions,
} from "@/api/sequenceModels";

const STATE_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
  "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
];

function StateTimeline({
  items,
  nStates,
  selectedSpan,
}: {
  items: Record<string, unknown>[];
  nStates: number;
  selectedSpan: number;
}) {
  const spanItems = useMemo(
    () => items.filter((r) => r.merged_span_id === selectedSpan),
    [items, selectedSpan],
  );

  if (spanItems.length === 0) {
    return <div className="text-sm text-slate-500">No data for this span.</div>;
  }

  const traces = useMemo(() => {
    const byState: Record<number, { x: number[]; y: number[] }> = {};
    for (let s = 0; s < nStates; s++) {
      byState[s] = { x: [], y: [] };
    }
    for (const row of spanItems) {
      const state = row.viterbi_state as number;
      const start = row.start_time_sec as number;
      const end = row.end_time_sec as number;
      byState[state].x.push(start, end, end);
      byState[state].y.push(state, state, null as unknown as number);
    }
    return Array.from({ length: nStates }, (_, s) => ({
      x: byState[s].x,
      y: byState[s].y,
      mode: "lines" as const,
      line: { color: STATE_COLORS[s % STATE_COLORS.length], width: 8 },
      name: `State ${s}`,
      connectgaps: false,
    }));
  }, [spanItems, nStates]);

  return (
    <div data-testid="hmm-state-timeline">
      <Plot
        data={traces}
        layout={{
          height: 200,
          margin: { l: 50, r: 20, t: 10, b: 40 },
          xaxis: { title: { text: "Time (sec)" } },
          yaxis: {
            title: { text: "State" },
            dtick: 1,
            range: [-0.5, nStates - 0.5],
          },
          showlegend: true,
          legend: { orientation: "h", y: 1.15 },
        }}
        config={{ responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function TransitionHeatmap({
  matrix,
  nStates,
}: {
  matrix: number[][];
  nStates: number;
}) {
  const labels = Array.from({ length: nStates }, (_, i) => `S${i}`);
  const annotations = useMemo(() => {
    const anns: Array<{
      x: string;
      y: string;
      text: string;
      showarrow: boolean;
      font: { size: number; color: string };
    }> = [];
    for (let i = 0; i < nStates; i++) {
      for (let j = 0; j < nStates; j++) {
        anns.push({
          x: labels[j],
          y: labels[i],
          text: matrix[i][j].toFixed(2),
          showarrow: false,
          font: { size: 10, color: matrix[i][j] > 0.5 ? "white" : "black" },
        });
      }
    }
    return anns;
  }, [matrix, nStates, labels]);

  return (
    <div data-testid="hmm-transition-heatmap">
      <Plot
        data={[
          {
            z: matrix,
            x: labels,
            y: labels,
            type: "heatmap",
            colorscale: "Blues",
            showscale: true,
          },
        ]}
        layout={{
          height: 300,
          margin: { l: 50, r: 20, t: 10, b: 40 },
          xaxis: { title: { text: "To State" } },
          yaxis: { title: { text: "From State" }, autorange: "reversed" as const },
          annotations,
        }}
        config={{ responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function DwellHistogramsGrid({
  histograms,
  nStates,
}: {
  histograms: Record<string, number[]>;
  nStates: number;
}) {
  return (
    <div
      className="grid grid-cols-2 gap-4"
      data-testid="hmm-dwell-histograms"
    >
      {Array.from({ length: nStates }, (_, s) => {
        const bins = histograms[String(s)] ?? [];
        return (
          <div key={s}>
            <Plot
              data={[
                {
                  x: bins.map((_, i) => i + 1),
                  y: bins,
                  type: "bar",
                  marker: {
                    color: STATE_COLORS[s % STATE_COLORS.length],
                  },
                },
              ]}
              layout={{
                height: 180,
                margin: { l: 40, r: 10, t: 30, b: 30 },
                title: { text: `State ${s}`, font: { size: 12 } },
                xaxis: { title: { text: "Dwell (frames)" } },
                yaxis: { title: { text: "Count" } },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>
        );
      })}
    </div>
  );
}

export function HMMSequenceDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data, isLoading, error } = useHMMSequenceJob(jobId ?? null);
  const cancelMutation = useCancelHMMSequenceJob();

  const isComplete = data?.job.status === "complete";

  const { data: statesData } = useHMMStates(
    jobId ?? null,
    0,
    5000,
    isComplete,
  );
  const { data: transData } = useHMMTransitions(jobId ?? null, isComplete);
  const { data: dwellData } = useHMMDwell(jobId ?? null, isComplete);

  const spanIds = useMemo(() => {
    if (!statesData?.items) return [];
    const ids = new Set<number>();
    for (const row of statesData.items) {
      ids.add(row.merged_span_id as number);
    }
    return Array.from(ids).sort((a, b) => a - b);
  }, [statesData]);

  const [selectedSpan, setSelectedSpan] = useState<number | null>(null);
  const activeSpan = selectedSpan ?? spanIds[0] ?? 0;

  if (isLoading) {
    return (
      <div className="text-sm text-slate-500" data-testid="hmm-detail-loading">
        Loading…
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="text-sm text-red-700" data-testid="hmm-detail-error">
        Job not found.
      </div>
    );
  }

  const { job, summary } = data as HMMSequenceJobDetail;
  const active = isHMMSequenceJobActive(job);

  return (
    <div className="space-y-4" data-testid="hmm-detail-page">
      <div>
        <Link
          to="/app/sequence-models/hmm-sequence"
          className="text-sm text-blue-700 hover:underline"
        >
          ← Back to jobs
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle data-testid="hmm-detail-id">{job.id}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div>
            <span className="font-medium">Status:</span>{" "}
            <span data-testid="hmm-detail-status">{job.status}</span>
          </div>
          <div className="grid grid-cols-4 gap-2">
            <div>
              <span className="font-medium text-slate-500">states</span>
              <div>{job.n_states}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">pca_dims</span>
              <div>{job.pca_dims}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">cov_type</span>
              <div>{job.covariance_type}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">n_iter</span>
              <div>{job.n_iter}</div>
            </div>
          </div>
          {isComplete && (
            <div className="grid grid-cols-4 gap-2">
              <div>
                <span className="font-medium text-slate-500">
                  log_likelihood
                </span>
                <div>{job.train_log_likelihood?.toFixed(1) ?? "—"}</div>
              </div>
              <div>
                <span className="font-medium text-slate-500">train_seqs</span>
                <div>{job.n_train_sequences ?? "—"}</div>
              </div>
              <div>
                <span className="font-medium text-slate-500">train_frames</span>
                <div>{job.n_train_frames ?? "—"}</div>
              </div>
              <div>
                <span className="font-medium text-slate-500">decoded_seqs</span>
                <div>{job.n_decoded_sequences ?? "—"}</div>
              </div>
            </div>
          )}
          <div>
            <span className="font-medium">Source CEJ:</span>{" "}
            {job.continuous_embedding_job_id.slice(0, 8)}
          </div>
          <div>
            <span className="font-medium">Created (UTC):</span>{" "}
            {job.created_at}
          </div>
          {job.error_message ? (
            <div
              className="text-red-700"
              data-testid="hmm-detail-error-message"
            >
              <span className="font-medium">Error:</span> {job.error_message}
            </div>
          ) : null}
          {active ? (
            <Button
              size="sm"
              variant="outline"
              disabled={cancelMutation.isPending}
              onClick={() => cancelMutation.mutate(job.id)}
              data-testid="hmm-detail-cancel"
            >
              Cancel
            </Button>
          ) : null}
        </CardContent>
      </Card>

      {job.status === "running" && (
        <div className="text-sm text-slate-500" data-testid="hmm-detail-running">
          Job is running — charts will appear when complete.
        </div>
      )}

      {isComplete && statesData && (
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle>State Timeline</CardTitle>
              {spanIds.length > 1 && (
                <select
                  data-testid="hmm-span-selector"
                  className="border rounded-md px-2 py-1 text-sm"
                  value={activeSpan}
                  onChange={(e) => setSelectedSpan(Number(e.target.value))}
                >
                  {spanIds.map((id) => (
                    <option key={id} value={id}>
                      Span {id}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <StateTimeline
              items={statesData.items}
              nStates={job.n_states}
              selectedSpan={activeSpan}
            />
          </CardContent>
        </Card>
      )}

      {isComplete && transData && (
        <Card>
          <CardHeader>
            <CardTitle>Transition Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <TransitionHeatmap
              matrix={transData.matrix}
              nStates={transData.n_states}
            />
          </CardContent>
        </Card>
      )}

      {isComplete && dwellData && (
        <Card>
          <CardHeader>
            <CardTitle>Dwell-Time Histograms</CardTitle>
          </CardHeader>
          <CardContent>
            <DwellHistogramsGrid
              histograms={dwellData.histograms}
              nStates={dwellData.n_states}
            />
          </CardContent>
        </Card>
      )}

      {isComplete && summary && (
        <Card>
          <CardHeader>
            <CardTitle>State Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <table
              className="w-full text-xs"
              data-testid="hmm-state-summary-table"
            >
              <thead>
                <tr className="text-left">
                  <th className="pr-2">State</th>
                  <th className="pr-2">Occupancy</th>
                  <th className="pr-2">Mean Dwell (frames)</th>
                </tr>
              </thead>
              <tbody>
                {summary.map((s) => (
                  <tr key={s.state} className="border-t">
                    <td className="pr-2 py-1">{s.state}</td>
                    <td className="pr-2 py-1">
                      {(s.occupancy * 100).toFixed(1)}%
                    </td>
                    <td className="pr-2 py-1">{s.mean_dwell_frames.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
