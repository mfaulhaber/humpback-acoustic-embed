import Plot from "react-plotly.js";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import {
  useMaskedTransformerExemplars,
  useMaskedTransformerJob,
  useMaskedTransformerLossCurve,
  useMaskedTransformerOverlay,
  useMaskedTransformerRunLengths,
  useRunMaskedTransformerAnalysis,
  type ExemplarRecord,
  type OverlayPoint,
} from "@/api/sequenceModels";
import { KPicker, useSelectedK } from "./KPicker";
import { LossCurveChart } from "./LossCurveChart";
import { TokenRunLengthHistograms } from "./TokenRunLengthHistograms";

const FULL_REPORT_OPTIONS = {
  retrieval_modes: [
    "unrestricted",
    "exclude_same_event",
    "exclude_same_event_and_region",
  ],
  embedding_variants: [
    "raw_l2",
    "centered_l2",
    "remove_pc1",
    "remove_pc3",
    "remove_pc5",
    "remove_pc10",
    "whiten_pca",
  ],
  include_event_level: true,
  include_geometry_report: true,
  include_query_rows: true,
  include_neighbor_rows: false,
};

export function MTTrainingDetailPage() {
  const { jobId = "" } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { data, isLoading } = useMaskedTransformerJob(jobId);
  const kValues = data?.job.k_values ?? [];
  const k = useSelectedK(kValues);
  const isComplete = data?.job.status === "complete";
  const loss = useMaskedTransformerLossCurve(jobId, isComplete);
  const runLengths = useMaskedTransformerRunLengths(jobId, k, isComplete && k != null);
  const overlay = useMaskedTransformerOverlay(jobId, k, 0, 5000, isComplete && k != null);
  const exemplars = useMaskedTransformerExemplars(jobId, k, isComplete && k != null);
  const analysisMutation = useRunMaskedTransformerAnalysis();

  if (isLoading || !data) {
    return (
      <div className="p-2 text-sm text-muted-foreground" data-testid="mt-training-detail-page">
        Loading...
      </div>
    );
  }

  const { job, sources } = data;

  const runAnalysis = () => {
    analysisMutation.mutate(
      {
        jobId,
        body: {
          ...FULL_REPORT_OPTIONS,
          k,
        },
      },
      {
        onSuccess: () =>
          navigate(`/app/sequence-models/mt-training/${jobId}/analysis`),
      },
    );
  };

  return (
    <div className="space-y-4 p-2" data-testid="mt-training-detail-page">
      <div className="rounded-md border p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-2">
            <h1 className="text-lg font-semibold">MT Training · {job.id.slice(0, 8)}</h1>
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={job.status} />
              <Badge variant="outline">preset: {job.preset}</Badge>
              <Badge variant="outline">sources: {Math.max(sources.length, 1)}</Badge>
              <Badge variant="outline">chunks: {job.total_chunks ?? "-"}</Badge>
              {job.chosen_device ? (
                <Badge variant={job.fallback_reason ? "destructive" : "secondary"}>
                  device: {job.chosen_device}
                </Badge>
              ) : null}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              onClick={runAnalysis}
              disabled={!isComplete || analysisMutation.isPending}
              data-testid="mt-training-analysis-button"
            >
              Analyze
            </Button>
            <Link className="text-xs underline text-muted-foreground" to="/app/sequence-models/mt-training">
              Back
            </Link>
          </div>
        </div>
      </div>

      <div className="rounded-md border p-4">
        <h2 className="mb-3 text-sm font-semibold">Sources</h2>
        <table className="w-full text-xs" data-testid="mt-training-source-table">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-2 py-1 text-left">Order</th>
              <th className="px-2 py-1 text-left">Embedding Job</th>
              <th className="px-2 py-1 text-left">Classify Job</th>
              <th className="px-2 py-1 text-left">Alias</th>
            </tr>
          </thead>
          <tbody>
            {(sources.length ? sources : [
              {
                source_order: 0,
                continuous_embedding_job_id: job.continuous_embedding_job_id,
                event_classification_job_id: job.event_classification_job_id ?? "",
                source_alias: null,
              },
            ]).map((source) => (
              <tr key={`${source.source_order}:${source.continuous_embedding_job_id}`} className="border-b">
                <td className="px-2 py-1">{source.source_order}</td>
                <td className="break-all px-2 py-1 font-mono">
                  {source.continuous_embedding_job_id}
                </td>
                <td className="break-all px-2 py-1 font-mono">
                  {source.event_classification_job_id}
                </td>
                <td className="px-2 py-1">{source.source_alias ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between rounded-md border p-3">
        <KPicker kValues={kValues} />
        <div className="text-xs text-muted-foreground">
          retrieval head: {job.retrieval_head_enabled ? job.retrieval_head_arch : "off"}
        </div>
      </div>

      {loss.data ? <LossCurveChart data={loss.data} /> : null}
      {runLengths.data ? (
        <TokenRunLengthHistograms
          k={runLengths.data.k}
          runLengths={runLengths.data.run_lengths}
        />
      ) : null}
      {overlay.data ? <TrainingOverlay data={overlay.data.items} /> : null}
      {exemplars.data ? <TrainingExemplars states={exemplars.data.states} /> : null}
    </div>
  );
}

function TrainingOverlay({ data }: { data: OverlayPoint[] }) {
  return (
    <div className="rounded-md border p-2" data-testid="mt-training-overlay">
      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">
        Token overlay
      </div>
      <Plot
        data={[
          {
            x: data.map((row) => row.umap_x),
            y: data.map((row) => row.umap_y),
            mode: "markers",
            type: "scattergl",
            marker: {
              color: data.map((row) => row.viterbi_state),
              colorscale: "Viridis",
              size: 7,
            },
            text: data.map((row) => `${row.sequence_id}:${row.position_in_sequence}`),
          },
        ]}
        layout={{
          autosize: true,
          height: 280,
          margin: { l: 32, r: 12, t: 8, b: 32 },
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

function TrainingExemplars({
  states,
}: {
  states: Record<string, ExemplarRecord[]>;
}) {
  const rows = Object.entries(states).flatMap(([state, records]) =>
    records.map((record) => ({ state, record })),
  );
  return (
    <div className="rounded-md border p-4" data-testid="mt-training-exemplars">
      <h2 className="mb-3 text-sm font-semibold">Exemplars</h2>
      {rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">No exemplars available.</p>
      ) : (
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {rows.slice(0, 24).map(({ state, record }, index) => (
            <div key={`${state}-${index}`} className="rounded-md border p-2 text-xs">
              <div className="flex items-center justify-between">
                <Badge variant="outline">token {state}</Badge>
                <span className="text-muted-foreground">{record.exemplar_type}</span>
              </div>
              <div className="mt-2 font-mono">
                {record.sequence_id}:{record.position_in_sequence}
              </div>
              <div className="text-muted-foreground">
                {record.start_timestamp.toFixed(2)}-{record.end_timestamp.toFixed(2)}
              </div>
              <div className="text-muted-foreground">
                p={record.max_state_probability.toFixed(3)}
                {record.extras?.tier ? ` · ${record.extras.tier}` : ""}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
