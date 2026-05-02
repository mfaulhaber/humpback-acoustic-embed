import { useMemo } from "react";
import Plot from "react-plotly.js";
import { Link, useParams } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { regionAudioSliceUrl } from "@/api/client";
import {
  type ExemplarRecord,
  useMaskedTransformerExemplars,
  useMaskedTransformerJob,
  useMaskedTransformerLabelDistribution,
  useMaskedTransformerLossCurve,
  useMaskedTransformerOverlay,
  useMaskedTransformerRunLengths,
  useMaskedTransformerTokens,
} from "@/api/sequenceModels";
import { DiscreteSequenceBar, type DiscreteSequenceItem } from "./DiscreteSequenceBar";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import { KPicker, useSelectedK } from "./KPicker";
import { LossCurveChart } from "./LossCurveChart";
import { MotifExtractionPanel } from "./MotifExtractionPanel";
import { TokenRunLengthHistograms } from "./TokenRunLengthHistograms";
import { labelColor } from "./constants";

export function MaskedTransformerDetailPage() {
  const { jobId = "" } = useParams<{ jobId: string }>();
  const { data, isLoading } = useMaskedTransformerJob(jobId);

  if (isLoading || !data) {
    return (
      <div className="space-y-4 p-2" data-testid="masked-transformer-detail-page">
        <div className="text-sm text-muted-foreground">Loading…</div>
      </div>
    );
  }

  const { job, region_detection_job_id, region_start_timestamp, region_end_timestamp, source_kind } = data;
  const isComplete = job.status === "complete";
  const kValues = job.k_values ?? [];

  return (
    <div className="space-y-4 p-2" data-testid="masked-transformer-detail-page">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">
              Masked Transformer · {job.id.slice(0, 8)}
            </CardTitle>
            <Link
              className="text-xs underline text-muted-foreground"
              to="/app/sequence-models/masked-transformer"
            >
              ← back
            </Link>
          </div>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex items-center gap-3">
            <StatusBadge status={job.status} />
            {job.chosen_device && (
              <Badge
                variant={job.fallback_reason ? "destructive" : "secondary"}
                data-testid="mt-detail-device-badge"
              >
                device: {job.chosen_device}
                {job.fallback_reason ? ` (fallback)` : ""}
              </Badge>
            )}
            <Badge variant="outline">preset: {job.preset}</Badge>
            <Badge variant="outline">source: {source_kind}</Badge>
            <span className="text-muted-foreground text-xs">
              {kValues.length > 0 ? `k = ${kValues.join(", ")}` : "no k configured"}
            </span>
          </div>
          {job.error_message && (
            <div className="text-xs text-red-600" data-testid="mt-detail-error">
              {job.error_message}
            </div>
          )}
        </CardContent>
      </Card>

      {isComplete && kValues.length > 0 && (
        <KPicker kValues={kValues} />
      )}

      {isComplete && <LossCurveSection jobId={jobId} />}

      {isComplete && kValues.length > 0 && (
        <TimelineSection
          jobId={jobId}
          kValues={kValues}
          regionDetectionJobId={region_detection_job_id}
          regionStartTimestamp={region_start_timestamp}
          regionEndTimestamp={region_end_timestamp}
        />
      )}

      {isComplete && kValues.length > 0 && (
        <RunLengthsSection jobId={jobId} kValues={kValues} />
      )}

      {isComplete && kValues.length > 0 && <OverlaySection jobId={jobId} kValues={kValues} />}

      {isComplete && kValues.length > 0 && <ExemplarsSection jobId={jobId} kValues={kValues} />}

      {isComplete && kValues.length > 0 && (
        <LabelDistributionSection jobId={jobId} kValues={kValues} />
      )}

      {isComplete && kValues.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Motifs</CardTitle>
          </CardHeader>
          <CardContent>
            <MotifSection jobId={jobId} kValues={kValues} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function LossCurveSection({ jobId }: { jobId: string }) {
  const { data } = useMaskedTransformerLossCurve(jobId);
  if (!data) return null;
  return <LossCurveChart data={data} />;
}

function TimelineSection({
  jobId,
  kValues,
  regionDetectionJobId,
  regionStartTimestamp,
  regionEndTimestamp,
}: {
  jobId: string;
  kValues: number[];
  regionDetectionJobId: string | null;
  regionStartTimestamp: number | null;
  regionEndTimestamp: number | null;
}) {
  const k = useSelectedK(kValues);
  const { data } = useMaskedTransformerTokens(jobId, k);
  const items: DiscreteSequenceItem[] = useMemo(() => {
    return (data?.items ?? []).map((row) => ({
      start_timestamp: row.start_timestamp,
      end_timestamp: row.end_timestamp,
      label: row.label,
      confidence: row.confidence,
    }));
  }, [data]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Token Timeline (k={k ?? "?"})</CardTitle>
      </CardHeader>
      <CardContent>
        {regionDetectionJobId && regionStartTimestamp != null && regionEndTimestamp != null ? (
          <TimelineProvider
            key={`mt-timeline-${jobId}-${k}`}
            jobStart={regionStartTimestamp}
            jobEnd={regionEndTimestamp}
            zoomLevels={REVIEW_ZOOM}
            defaultZoom={REVIEW_ZOOM[0].key}
            playback="slice"
            audioUrlBuilder={(startEpoch, durationSec) =>
              regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec)
            }
          >
            <DiscreteSequenceBar
              items={items}
              numLabels={k ?? 0}
              mode="single-row"
              testId="mt-token-strip"
              ariaLabel="Masked-transformer token timeline"
              tooltipFormatter={(item) =>
                `Token ${item.label} · ${item.start_timestamp.toFixed(2)}s–${item.end_timestamp.toFixed(2)}s · conf ${(item.confidence ?? 0).toFixed(2)}`
              }
            />
          </TimelineProvider>
        ) : (
          <div className="text-xs text-muted-foreground">
            Token strip preview ({items.length} chunks). Region context unavailable.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RunLengthsSection({
  jobId,
  kValues,
}: {
  jobId: string;
  kValues: number[];
}) {
  const k = useSelectedK(kValues);
  const { data } = useMaskedTransformerRunLengths(jobId, k);
  if (!data || !k) return null;
  return (
    <TokenRunLengthHistograms
      runLengths={data.run_lengths}
      k={k}
    />
  );
}

function OverlaySection({
  jobId,
  kValues,
}: {
  jobId: string;
  kValues: number[];
}) {
  const k = useSelectedK(kValues);
  const { data } = useMaskedTransformerOverlay(jobId, k);
  if (!data || data.items.length === 0) return null;
  const points = data.items;
  const labels = Array.from(new Set(points.map((p) => p.viterbi_state))).sort(
    (a, b) => a - b,
  );
  const traces = labels.map((label) => {
    const subset = points.filter((p) => p.viterbi_state === label);
    return {
      x: subset.map((p) => p.umap_x),
      y: subset.map((p) => p.umap_y),
      mode: "markers" as const,
      type: "scatter" as const,
      name: `${label}`,
      marker: { color: labelColor(label, k ?? labels.length), size: 5 },
    };
  });
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Overlay (UMAP, k={k})</CardTitle>
      </CardHeader>
      <CardContent>
        <Plot
          data={traces}
          layout={{
            autosize: true,
            height: 360,
            margin: { l: 32, r: 8, t: 8, b: 32 },
            xaxis: { title: { text: "UMAP-x" } },
            yaxis: { title: { text: "UMAP-y" } },
            legend: { orientation: "h", y: -0.2 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </CardContent>
    </Card>
  );
}

function ExemplarsSection({
  jobId,
  kValues,
}: {
  jobId: string;
  kValues: number[];
}) {
  const k = useSelectedK(kValues);
  const { data } = useMaskedTransformerExemplars(jobId, k);
  if (!data) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Exemplars (k={k})</CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className="grid gap-2"
          style={{ gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))" }}
          data-testid="mt-exemplar-gallery"
        >
          {Object.entries(data.states).map(([labelKey, records]) => (
            <div key={labelKey} className="border rounded-md p-2 text-xs">
              <div className="font-semibold mb-1">Token {labelKey}</div>
              <ExemplarList records={records} />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function ExemplarList({ records }: { records: ExemplarRecord[] }) {
  if (records.length === 0) {
    return <div className="text-muted-foreground">no exemplars</div>;
  }
  return (
    <ul className="space-y-1">
      {records.slice(0, 5).map((r, i) => (
        <li key={i}>
          <div>
            {r.start_timestamp.toFixed(2)}s · prob {(r.max_state_probability ?? 0).toFixed(2)}
          </div>
          <div className="text-muted-foreground">{r.exemplar_type}</div>
          {typeof r.extras?.tier === "string" && r.extras.tier && (
            <Badge variant="outline" className="text-[10px]">
              {String(r.extras.tier)}
            </Badge>
          )}
        </li>
      ))}
    </ul>
  );
}

function LabelDistributionSection({
  jobId,
  kValues,
}: {
  jobId: string;
  kValues: number[];
}) {
  const k = useSelectedK(kValues);
  const { data } = useMaskedTransformerLabelDistribution(jobId, k);
  // Collapse tier dimension to match the existing HMM chart visual.
  // ``useMemo`` must run unconditionally to satisfy hooks rules; it
  // returns an empty record when ``data`` is not yet available.
  const collapsed = useMemo(() => {
    const out: Record<string, Record<string, number>> = {};
    if (!data) return out;
    for (const [stateKey, tiers] of Object.entries(data.states)) {
      const inner: Record<string, number> = {};
      for (const counts of Object.values(tiers)) {
        for (const [label, count] of Object.entries(counts)) {
          inner[label] = (inner[label] ?? 0) + count;
        }
      }
      out[stateKey] = inner;
    }
    return out;
  }, [data]);
  if (!data) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Label Distribution (k={k})</CardTitle>
      </CardHeader>
      <CardContent>
        <table className="text-xs" data-testid="mt-label-distribution">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left">Token</th>
              <th className="px-2 py-1 text-left">Top labels</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(collapsed).map(([stateKey, counts]) => {
              const sorted = Object.entries(counts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5);
              return (
                <tr key={stateKey} className="border-t">
                  <td className="px-2 py-1 font-mono">{stateKey}</td>
                  <td className="px-2 py-1">
                    {sorted.length === 0 ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      sorted.map(([label, count]) => (
                        <span key={label} className="mr-2">
                          {label}: {count}
                        </span>
                      ))
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function MotifSection({ jobId, kValues }: { jobId: string; kValues: number[] }) {
  const k = useSelectedK(kValues);
  if (k == null) return null;
  return (
    <MotifExtractionPanel
      regionDetectionJobId=""
      onJumpToTimestamp={() => {}}
      parent={{
        kind: "masked_transformer",
        maskedTransformerJobId: jobId,
        k,
      }}
    />
  );
}
