import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { Link, useParams } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { regionAudioSliceUrl, regionTileUrl } from "@/api/client";
import {
  type ExemplarRecord,
  useMaskedTransformerExemplars,
  useMaskedTransformerJob,
  useMaskedTransformerLabelDistribution,
  useMaskedTransformerLossCurve,
  useMaskedTransformerOverlay,
  useMaskedTransformerReconstructionError,
  useMaskedTransformerRunLengths,
  useMaskedTransformerTokens,
} from "@/api/sequenceModels";
import { DiscreteSequenceBar, type DiscreteSequenceItem } from "./DiscreteSequenceBar";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import type { TimelinePlaybackHandle } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { PlaybackControls } from "@/components/timeline/controls/PlaybackControls";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import {
  CONFIDENCE_GRADIENT,
  COLORS,
  FREQ_AXIS_WIDTH_PX,
} from "@/components/timeline/constants";
import { CollapsiblePanelCard } from "./CollapsiblePanelCard";
import { KPicker, useSelectedK } from "./KPicker";
import { LossCurveChart } from "./LossCurveChart";
import {
  MotifExtractionPanel,
  type MotifPanelSelection,
} from "./MotifExtractionPanel";
import { MotifTimelineLegend } from "./MotifTimelineLegend";
import { TokenRunLengthHistograms } from "./TokenRunLengthHistograms";
import { labelColor } from "./constants";

const EMPTY_MOTIF_SELECTION: MotifPanelSelection = {
  motifKey: null,
  motif: null,
  occurrences: [],
  occurrencesTotal: 0,
  activeOccurrenceIndex: 0,
};

function parseMotifKeyToStates(motifKey: string | null): number[] {
  if (motifKey == null) return [];
  return motifKey
    .split("-")
    .map((s) => Number.parseInt(s, 10))
    .filter((n) => Number.isFinite(n));
}

export function MaskedTransformerDetailPage() {
  const { jobId = "" } = useParams<{ jobId: string }>();
  const { data, isLoading } = useMaskedTransformerJob(jobId);
  const timelineHandleRef = useRef<TimelinePlaybackHandle>(null!);
  const [motifSelection, setMotifSelection] = useState<MotifPanelSelection>(
    EMPTY_MOTIF_SELECTION,
  );
  const handleMotifPrev = useCallback(() => {
    setMotifSelection((prev) => {
      if (prev.occurrencesTotal === 0) return prev;
      const nextIdx = Math.max(0, prev.activeOccurrenceIndex - 1);
      const occ = prev.occurrences[nextIdx];
      if (occ) {
        timelineHandleRef.current?.seekTo(
          (occ.start_timestamp + occ.end_timestamp) / 2,
        );
      }
      return { ...prev, activeOccurrenceIndex: nextIdx };
    });
  }, []);
  const handleMotifNext = useCallback(() => {
    setMotifSelection((prev) => {
      if (prev.occurrencesTotal === 0) return prev;
      const nextIdx = Math.min(
        prev.occurrencesTotal - 1,
        prev.activeOccurrenceIndex + 1,
      );
      const occ = prev.occurrences[nextIdx];
      if (occ) {
        timelineHandleRef.current?.seekTo(
          (occ.start_timestamp + occ.end_timestamp) / 2,
        );
      }
      return { ...prev, activeOccurrenceIndex: nextIdx };
    });
  }, []);
  const handleMotifSelectionChange = useCallback((sel: MotifPanelSelection) => {
    setMotifSelection(sel);
  }, []);
  const handleActiveOccurrenceChange = useCallback((idx: number) => {
    setMotifSelection((prev) => ({ ...prev, activeOccurrenceIndex: idx }));
  }, []);

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

      {isComplete && kValues.length > 0 && (
        <TimelineSection
          jobId={jobId}
          kValues={kValues}
          regionDetectionJobId={region_detection_job_id}
          regionStartTimestamp={region_start_timestamp}
          regionEndTimestamp={region_end_timestamp}
          timelineHandleRef={timelineHandleRef}
          motifSelection={motifSelection}
          onMotifPrev={handleMotifPrev}
          onMotifNext={handleMotifNext}
        />
      )}

      {isComplete && kValues.length > 0 && (
        <CollapsiblePanelCard
          title="Motifs"
          storageKey="mt:motifs"
          testId="mt-motifs-panel"
        >
          <MotifSection
            jobId={jobId}
            kValues={kValues}
            regionDetectionJobId={region_detection_job_id}
            timelineHandleRef={timelineHandleRef}
            onSelectionChange={handleMotifSelectionChange}
            activeOccurrenceIndex={motifSelection.activeOccurrenceIndex}
            onActiveOccurrenceChange={handleActiveOccurrenceChange}
          />
        </CollapsiblePanelCard>
      )}

      {isComplete && (
        <CollapsiblePanelCard
          title="Loss Curve"
          storageKey="mt:loss-curve"
          testId="mt-loss-curve-panel"
        >
          <LossCurveSection jobId={jobId} />
        </CollapsiblePanelCard>
      )}

      {isComplete && kValues.length > 0 && (
        <CollapsiblePanelCard
          title="Run-Length Histograms"
          storageKey="mt:run-lengths"
          testId="mt-run-lengths-panel"
        >
          <RunLengthsSection jobId={jobId} kValues={kValues} />
        </CollapsiblePanelCard>
      )}

      {isComplete && kValues.length > 0 && (
        <CollapsiblePanelCard
          title={<OverlayTitle kValues={kValues} />}
          storageKey="mt:overlay"
          testId="mt-overlay-panel"
        >
          <OverlaySection jobId={jobId} kValues={kValues} />
        </CollapsiblePanelCard>
      )}

      {isComplete && kValues.length > 0 && (
        <CollapsiblePanelCard
          title={<ExemplarsTitle kValues={kValues} />}
          storageKey="mt:exemplars"
          testId="mt-exemplars-panel"
        >
          <ExemplarsSection jobId={jobId} kValues={kValues} />
        </CollapsiblePanelCard>
      )}

      {isComplete && kValues.length > 0 && (
        <CollapsiblePanelCard
          title={<LabelDistributionTitle kValues={kValues} />}
          storageKey="mt:label-distribution"
          testId="mt-label-distribution-panel"
        >
          <LabelDistributionSection jobId={jobId} kValues={kValues} />
        </CollapsiblePanelCard>
      )}
    </div>
  );
}

function OverlayTitle({ kValues }: { kValues: number[] }) {
  const k = useSelectedK(kValues);
  return <>Overlay (UMAP, k={k})</>;
}

function ExemplarsTitle({ kValues }: { kValues: number[] }) {
  const k = useSelectedK(kValues);
  return <>Exemplars (k={k})</>;
}

function LabelDistributionTitle({ kValues }: { kValues: number[] }) {
  const k = useSelectedK(kValues);
  return <>Label Distribution (k={k})</>;
}

function LossCurveSection({ jobId }: { jobId: string }) {
  const { data } = useMaskedTransformerLossCurve(jobId);
  if (!data) return null;
  return <LossCurveChart data={data} />;
}

interface ChunkScore {
  start_timestamp: number;
  end_timestamp: number;
  score: number;
}

// API caps the per-page limit at 50000; jobs typically have 15-30k chunks.
const CHUNK_FETCH_LIMIT = 50000;
const CONFIDENCE_STRIP_HEIGHT = 24;

function lerpHexColor(a: string, b: string, t: number): string {
  const pa = [
    parseInt(a.slice(1, 3), 16),
    parseInt(a.slice(3, 5), 16),
    parseInt(a.slice(5, 7), 16),
  ];
  const pb = [
    parseInt(b.slice(1, 3), 16),
    parseInt(b.slice(3, 5), 16),
    parseInt(b.slice(5, 7), 16),
  ];
  const r = Math.round(pa[0] + (pb[0] - pa[0]) * t);
  const g = Math.round(pa[1] + (pb[1] - pa[1]) * t);
  const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

function gradientColor(score: number): string {
  const s = Math.max(0, Math.min(1, score));
  const grad = CONFIDENCE_GRADIENT;
  for (let i = 1; i < grad.length; i++) {
    const [prevT, prevC] = grad[i - 1];
    const [curT, curC] = grad[i];
    if (s <= curT) {
      const t = (s - prevT) / (curT - prevT);
      return lerpHexColor(prevC, curC, t);
    }
  }
  return grad[grad.length - 1][1];
}

/**
 * Per-chunk confidence strip aligned to the timeline viewport. Iterates the
 * actual chunk items rather than a uniform-bin score array so it stays cheap
 * for sparse chunk coverage over long region-detection job spans.
 */
function ChunkConfidenceStrip({
  items,
  height = CONFIDENCE_STRIP_HEIGHT,
  label,
  testId,
  scoreNormalizer,
}: {
  items: ChunkScore[];
  height?: number;
  label: string;
  testId: string;
  /** Optional normalizer mapping raw score → [0, 1]. Defaults to identity. */
  scoreNormalizer?: (score: number) => number;
}) {
  const ctx = useTimelineContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(Math.floor(entry.contentRect.width));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const canvasWidth = Math.max(0, containerWidth - FREQ_AXIS_WIDTH_PX);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvasWidth <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = height * dpr;
    const g = canvas.getContext("2d");
    if (!g) return;
    g.scale(dpr, dpr);
    g.fillStyle = COLORS.bgDark;
    g.fillRect(0, 0, canvasWidth, height);
    for (const item of items) {
      if (item.end_timestamp < ctx.viewStart || item.start_timestamp > ctx.viewEnd) {
        continue;
      }
      const x0 = (item.start_timestamp - ctx.viewStart) * ctx.pxPerSec;
      const x1 = (item.end_timestamp - ctx.viewStart) * ctx.pxPerSec;
      const px = Math.max(0, x0);
      const pw = Math.max(1, Math.min(canvasWidth, x1) - px);
      const norm = scoreNormalizer ? scoreNormalizer(item.score) : item.score;
      g.fillStyle = gradientColor(norm);
      g.fillRect(px, 0, pw, height);
    }
  }, [items, canvasWidth, height, ctx.viewStart, ctx.viewEnd, ctx.pxPerSec, scoreNormalizer]);

  return (
    <div ref={containerRef} className="relative" data-testid={testId}>
      <div className="flex">
        <div
          className="flex items-center justify-end pr-1 text-[9px] text-muted-foreground shrink-0"
          style={{ width: FREQ_AXIS_WIDTH_PX, height }}
        >
          {label}
        </div>
        <canvas
          ref={canvasRef}
          className="block"
          style={{ width: canvasWidth, height }}
        />
      </div>
    </div>
  );
}

function TimelineBody({
  regionDetectionJobId,
  tokenItems,
  tokenScores,
  reconstructionScores,
  reconstructionMax,
  k,
}: {
  regionDetectionJobId: string;
  tokenItems: DiscreteSequenceItem[];
  tokenScores: ChunkScore[];
  reconstructionScores: ChunkScore[];
  reconstructionMax: number;
  k: number | null;
}) {
  const ctx = useTimelineContext();

  // Region-detection jobs typically span 24h while CRNN chunks only cover the
  // few minutes inside detected regions. Without seeking, the default
  // mid-job center sits in empty space. Seek once to the first chunk so the
  // viewer lands on data.
  const seekedRef = useRef(false);
  useEffect(() => {
    if (seekedRef.current) return;
    if (tokenItems.length === 0) return;
    const first = tokenItems.reduce((min, item) =>
      item.start_timestamp < min.start_timestamp ? item : min,
    );
    seekedRef.current = true;
    ctx.seekTo(first.start_timestamp);
  }, [tokenItems, ctx]);

  const tileUrlBuilder = useCallback(
    (
      _jid: string,
      zoomLevel: string,
      tileIndex: number,
      _freqMin: number,
      _freqMax: number,
    ) => regionTileUrl(regionDetectionJobId, zoomLevel, tileIndex),
    [regionDetectionJobId],
  );

  const reconstructionNormalizer = useCallback(
    (score: number) => (reconstructionMax > 0 ? score / reconstructionMax : 0),
    [reconstructionMax],
  );

  return (
    <>
      <div className="flex" style={{ height: 200 }}>
        <Spectrogram
          jobId={regionDetectionJobId}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
        />
      </div>
      <DiscreteSequenceBar
        items={tokenItems}
        numLabels={k ?? 0}
        mode="single-row"
        testId="mt-token-strip"
        ariaLabel="Masked-transformer token timeline"
        tooltipFormatter={(item) =>
          `Token ${item.label} · ${item.start_timestamp.toFixed(2)}s–${item.end_timestamp.toFixed(2)}s · conf ${(item.confidence ?? 0).toFixed(2)}`
        }
      />
      <ChunkConfidenceStrip
        items={tokenScores}
        label="conf"
        testId="mt-token-confidence-strip"
      />
      <ChunkConfidenceStrip
        items={reconstructionScores}
        label="recon"
        testId="mt-reconstruction-error-strip"
        scoreNormalizer={reconstructionNormalizer}
      />
      <div className="flex justify-center py-1">
        <ZoomSelector />
      </div>
      <div className="flex justify-center">
        <PlaybackControls />
      </div>
    </>
  );
}

function TimelineSection({
  jobId,
  kValues,
  regionDetectionJobId,
  regionStartTimestamp,
  regionEndTimestamp,
  timelineHandleRef,
  motifSelection,
  onMotifPrev,
  onMotifNext,
}: {
  jobId: string;
  kValues: number[];
  regionDetectionJobId: string | null;
  regionStartTimestamp: number | null;
  regionEndTimestamp: number | null;
  timelineHandleRef: React.RefObject<TimelinePlaybackHandle>;
  motifSelection: MotifPanelSelection;
  onMotifPrev: () => void;
  onMotifNext: () => void;
}) {
  const k = useSelectedK(kValues);
  const { data: tokensData } = useMaskedTransformerTokens(
    jobId,
    k,
    0,
    CHUNK_FETCH_LIMIT,
  );
  const { data: reconstructionData } = useMaskedTransformerReconstructionError(
    jobId,
    0,
    CHUNK_FETCH_LIMIT,
  );

  const tokenItems: DiscreteSequenceItem[] = useMemo(() => {
    return (tokensData?.items ?? []).map((row) => ({
      start_timestamp: row.start_timestamp,
      end_timestamp: row.end_timestamp,
      label: row.label,
      confidence: row.confidence,
    }));
  }, [tokensData]);

  const tokenScores: ChunkScore[] = useMemo(() => {
    return (tokensData?.items ?? []).map((row) => ({
      start_timestamp: row.start_timestamp,
      end_timestamp: row.end_timestamp,
      score: row.confidence,
    }));
  }, [tokensData]);

  const reconstructionScores: ChunkScore[] = useMemo(() => {
    return (reconstructionData?.items ?? []).map((row) => ({
      start_timestamp: row.start_timestamp,
      end_timestamp: row.end_timestamp,
      score: row.score,
    }));
  }, [reconstructionData]);

  const reconstructionMax = useMemo(() => {
    let m = 0;
    for (const r of reconstructionScores) {
      if (r.score > m) m = r.score;
    }
    return m;
  }, [reconstructionScores]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Token Timeline (k={k ?? "?"})</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <MotifTimelineLegend
          selectedMotifKey={motifSelection.motifKey}
          selectedStates={parseMotifKeyToStates(motifSelection.motifKey)}
          numLabels={k ?? 0}
          occurrencesTotal={motifSelection.occurrencesTotal}
          activeOccurrenceIndex={motifSelection.activeOccurrenceIndex}
          onPrev={onMotifPrev}
          onNext={onMotifNext}
        />
        {regionDetectionJobId && regionStartTimestamp != null && regionEndTimestamp != null ? (
          <div data-testid="mt-timeline-viewer">
            <TimelineProvider
              key={`mt-timeline-${jobId}-${k}`}
              ref={timelineHandleRef}
              jobStart={regionStartTimestamp}
              jobEnd={regionEndTimestamp}
              zoomLevels={REVIEW_ZOOM}
              defaultZoom={REVIEW_ZOOM[0].key}
              playback="slice"
              audioUrlBuilder={(startEpoch, durationSec) =>
                regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec)
              }
            >
              <TimelineBody
                regionDetectionJobId={regionDetectionJobId}
                tokenItems={tokenItems}
                tokenScores={tokenScores}
                reconstructionScores={reconstructionScores}
                reconstructionMax={reconstructionMax}
                k={k}
              />
            </TimelineProvider>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            Token strip preview ({tokenItems.length} chunks). Region context unavailable.
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
  );
}

function MotifSection({
  jobId,
  kValues,
  regionDetectionJobId,
  timelineHandleRef,
  onSelectionChange,
  activeOccurrenceIndex,
  onActiveOccurrenceChange,
}: {
  jobId: string;
  kValues: number[];
  regionDetectionJobId: string | null;
  timelineHandleRef: React.RefObject<TimelinePlaybackHandle>;
  onSelectionChange: (selection: MotifPanelSelection) => void;
  activeOccurrenceIndex: number;
  onActiveOccurrenceChange: (idx: number) => void;
}) {
  const k = useSelectedK(kValues);
  if (k == null) return null;
  if (!regionDetectionJobId) {
    return (
      <div className="text-xs text-muted-foreground">
        Region context unavailable; motif Play/Jump disabled.
      </div>
    );
  }
  return (
    <MotifExtractionPanel
      regionDetectionJobId={regionDetectionJobId}
      onJumpToTimestamp={(timestamp) =>
        timelineHandleRef.current?.seekTo(timestamp)
      }
      parent={{
        kind: "masked_transformer",
        maskedTransformerJobId: jobId,
        k,
      }}
      onSelectionChange={onSelectionChange}
      activeOccurrenceIndex={activeOccurrenceIndex}
      onActiveOccurrenceChange={onActiveOccurrenceChange}
      numLabels={k}
    />
  );
}
