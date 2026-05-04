import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { Link, useParams } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { regionAudioSliceUrl, regionTileUrl } from "@/api/client";
import {
  type ExemplarRecord,
  type MotifSummary,
  useContinuousEmbeddingJob,
  useEventClassificationJobsForSegmentation,
  useMaskedTransformerExemplars,
  useMaskedTransformerJob,
  useMaskedTransformerLabelDistribution,
  useMaskedTransformerLossCurve,
  useMaskedTransformerOverlay,
  useMaskedTransformerReconstructionError,
  useMaskedTransformerRunLengths,
  useMaskedTransformerTokens,
  useMotifExtractionJobs,
  useMotifs,
  useMotifsByLength,
  useRegenerateMTLabelDistribution,
} from "@/api/sequenceModels";
import { DiscreteSequenceBar, type DiscreteSequenceItem } from "./DiscreteSequenceBar";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import type { TimelinePlaybackHandle } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { MotifHighlightOverlay } from "@/components/timeline/overlays/MotifHighlightOverlay";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { PlaybackControls } from "@/components/timeline/controls/PlaybackControls";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import type { MotifOccurrence } from "@/api/sequenceModels";
import {
  CONFIDENCE_GRADIENT,
  COLORS,
  FREQ_AXIS_WIDTH_PX,
} from "@/components/timeline/constants";
import { colorForMotifKey } from "@/lib/motifColor";
import { CollapsiblePanelCard } from "./CollapsiblePanelCard";
import { KPicker, useSelectedK } from "./KPicker";
import { LossCurveChart } from "./LossCurveChart";
import {
  MotifExtractionPanel,
  type MotifPanelSelection,
} from "./MotifExtractionPanel";
import { MotifTimelineLegend } from "./MotifTimelineLegend";
import {
  MotifTokenCountSelector,
  type MotifTokenCount,
} from "./MotifTokenCountSelector";
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
  // byLength mode (Token Count selector). When ``byLengthLength`` is null
  // the page is in single-motif mode and the panel's row selection drives
  // the timeline overlay (existing behavior). When non-null, the page
  // overrides the overlay with all length-N motifs and hides the panel's
  // row highlight.
  const [byLengthLength, setByLengthLength] = useState<MotifTokenCount | null>(null);
  const [byLengthActiveIndex, setByLengthActiveIndex] = useState(0);
  const isByLengthMode = byLengthLength != null;

  // Resolve the upstream CEJ so we have the segmentation id (used by the
  // regenerate-label-distribution dialog to filter Classify candidates).
  const cejId = data?.job.continuous_embedding_job_id ?? null;
  const { data: cejDetail } = useContinuousEmbeddingJob(cejId);
  const segmentationJobId = cejDetail?.job.event_segmentation_job_id ?? null;

  // Lift the motif-extraction queries to the page level so byLength mode
  // can compute its own occurrences without duplicating fetches. The
  // panel makes the same calls below; React Query dedupes via shared
  // query keys.
  const kValues = data?.job.k_values ?? [];
  const k = useSelectedK(kValues);
  const motifJobsParams = k != null
    ? {
        masked_transformer_job_id: jobId,
        parent_kind: "masked_transformer" as const,
        k,
      }
    : undefined;
  const { data: motifJobs } = useMotifExtractionJobs(
    motifJobsParams,
    k != null && jobId !== "",
  );
  const activeMotifJob = motifJobs?.[0] ?? null;
  const isMotifJobComplete = activeMotifJob?.status === "complete";
  const { data: motifsResp, isLoading: motifsLoadingState } = useMotifs(
    activeMotifJob?.id ?? null,
    0,
    100,
    isMotifJobComplete,
  );
  const motifList: MotifSummary[] = motifsResp?.items ?? [];
  const motifsLoading = motifsLoadingState && activeMotifJob != null;
  const byLengthData = useMotifsByLength(
    activeMotifJob?.id ?? null,
    motifList,
    byLengthLength,
  );
  const availableLengths = useMemo(
    () => new Set(motifList.map((m) => m.length)),
    [motifList],
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
  const handlePlayMotif = useCallback(
    (occ: MotifOccurrence, idx: number) => {
      const handle = timelineHandleRef.current;
      if (!handle) return;
      const duration = Math.max(
        0.05,
        occ.end_timestamp - occ.start_timestamp,
      );
      handle.seekTo(occ.start_timestamp);
      handle.play(occ.start_timestamp, duration);
      setMotifSelection((prev) => ({ ...prev, activeOccurrenceIndex: idx }));
    },
    [],
  );
  const handleSelectByLength = useCallback(
    (next: MotifTokenCount | null) => {
      setByLengthLength(next);
      setByLengthActiveIndex(0);
    },
    [],
  );
  const handleUserSelectMotif = useCallback((_motifKey: string) => {
    // Picking a row in the panel exits byLength mode — single-motif
    // selection takes precedence per the discriminated-union design.
    setByLengthLength(null);
    setByLengthActiveIndex(0);
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
  // ``kValues`` and ``k`` were resolved above (before the early return)
  // because hooks must run unconditionally on every render. Re-aliased
  // here so the JSX below reads the same as before.

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
          <div className="flex items-center gap-3 flex-wrap">
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
            {job.event_classification_job_id ? (
              <Link
                className="no-underline"
                to={`/app/call-parsing/classify-review?job_id=${encodeURIComponent(
                  job.event_classification_job_id,
                )}`}
                data-testid="mt-detail-classify-badge"
              >
                <Badge variant="outline">
                  Labels: Classify #
                  {job.event_classification_job_id.slice(0, 8)}
                </Badge>
              </Link>
            ) : null}
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
          onPlayMotif={handlePlayMotif}
          byLengthLength={byLengthLength}
          onSelectByLength={handleSelectByLength}
          byLengthOccurrences={byLengthData.occurrences}
          byLengthActiveIndex={byLengthActiveIndex}
          onByLengthActiveIndexChange={setByLengthActiveIndex}
          availableLengths={availableLengths}
          motifsLoading={motifsLoading}
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
            onPlayMotif={handlePlayMotif}
            hideRowHighlight={isByLengthMode}
            onUserSelectMotif={handleUserSelectMotif}
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
          headerExtra={
            <RegenerateMTLabelDistributionTrigger
              mtJobId={jobId}
              kValues={kValues}
              segmentationJobId={segmentationJobId}
              boundClassifyId={job.event_classification_job_id}
            />
          }
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

// Hides the per-token confidence and reconstruction-error strips below the
// token bar. Per 2026-05-02-masked-transformer-motif-ux-design we are
// deferring these until they are needed; flip to ``true`` to restore.
const SHOW_CONFIDENCE_STRIPS = false;

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
  motifSelection,
  byLengthLength,
  byLengthOccurrences,
  byLengthActiveIndex,
  onByLengthActiveIndexChange,
  onSelectByLength,
  availableLengths,
  motifsLoading,
  onMotifPrev,
  onMotifNext,
  onPlayMotif,
  timelineHandleRef,
}: {
  regionDetectionJobId: string;
  tokenItems: DiscreteSequenceItem[];
  tokenScores: ChunkScore[];
  reconstructionScores: ChunkScore[];
  reconstructionMax: number;
  k: number | null;
  motifSelection: MotifPanelSelection;
  byLengthLength: MotifTokenCount | null;
  byLengthOccurrences: MotifOccurrence[];
  byLengthActiveIndex: number;
  onByLengthActiveIndexChange: (idx: number) => void;
  onSelectByLength: (next: MotifTokenCount | null) => void;
  availableLengths: ReadonlySet<number>;
  motifsLoading: boolean;
  onMotifPrev: () => void;
  onMotifNext: () => void;
  onPlayMotif: (occ: MotifOccurrence, idx: number) => void;
  timelineHandleRef: React.RefObject<TimelinePlaybackHandle>;
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

  const isByLengthMode = byLengthLength != null;
  const motifColorIndex = motifSelection.motif?.states[0] ?? 0;
  const showSingleOverlay =
    !isByLengthMode &&
    motifSelection.motifKey != null &&
    motifSelection.occurrences.length > 0;
  // ``visibleByLengthOccurrences`` is computed below; gate on the full
  // list here, then the overlay filters again by viewport at render time.

  // byLength visible-set + clamp + handlers. Lives here (inside the
  // TimelineProvider) so it can read view bounds from
  // ``useTimelineContext`` for the prev/next/Play handlers.
  const visibleByLengthOccurrences = useMemo(
    () =>
      byLengthOccurrences.filter(
        (o) =>
          o.end_timestamp >= ctx.viewStart && o.start_timestamp <= ctx.viewEnd,
      ),
    [byLengthOccurrences, ctx.viewStart, ctx.viewEnd],
  );

  // Keep the active index in range as the visible set changes.
  useEffect(() => {
    if (!isByLengthMode) return;
    if (visibleByLengthOccurrences.length === 0) return;
    if (byLengthActiveIndex >= visibleByLengthOccurrences.length) {
      onByLengthActiveIndexChange(0);
    }
  }, [
    isByLengthMode,
    visibleByLengthOccurrences,
    byLengthActiveIndex,
    onByLengthActiveIndexChange,
  ]);

  const handleByLengthPrev = useCallback(() => {
    if (visibleByLengthOccurrences.length === 0) return;
    const nextIdx = Math.max(0, byLengthActiveIndex - 1);
    const occ = visibleByLengthOccurrences[nextIdx];
    if (occ) {
      timelineHandleRef.current?.seekTo(
        (occ.start_timestamp + occ.end_timestamp) / 2,
      );
    }
    onByLengthActiveIndexChange(nextIdx);
  }, [
    visibleByLengthOccurrences,
    byLengthActiveIndex,
    onByLengthActiveIndexChange,
    timelineHandleRef,
  ]);

  const handleByLengthNext = useCallback(() => {
    if (visibleByLengthOccurrences.length === 0) return;
    const nextIdx = Math.min(
      visibleByLengthOccurrences.length - 1,
      byLengthActiveIndex + 1,
    );
    const occ = visibleByLengthOccurrences[nextIdx];
    if (occ) {
      timelineHandleRef.current?.seekTo(
        (occ.start_timestamp + occ.end_timestamp) / 2,
      );
    }
    onByLengthActiveIndexChange(nextIdx);
  }, [
    visibleByLengthOccurrences,
    byLengthActiveIndex,
    onByLengthActiveIndexChange,
    timelineHandleRef,
  ]);

  const handleByLengthPlay = useCallback(() => {
    const occ = visibleByLengthOccurrences[byLengthActiveIndex];
    if (occ) onPlayMotif(occ, byLengthActiveIndex);
  }, [visibleByLengthOccurrences, byLengthActiveIndex, onPlayMotif]);

  const tokenSelector = (
    <MotifTokenCountSelector
      value={byLengthLength}
      onChange={onSelectByLength}
      availableLengths={availableLengths}
      isMotifsLoading={motifsLoading}
    />
  );

  // Pick the legend's data source by mode. In byLength mode the legend
  // navigates the visible occurrence set. In single mode it preserves
  // existing behavior (full per-motif occurrence list, unfiltered).
  const legendOccurrencesTotal = isByLengthMode
    ? visibleByLengthOccurrences.length
    : motifSelection.occurrencesTotal;
  const legendActiveIndex = isByLengthMode
    ? Math.min(
        byLengthActiveIndex,
        Math.max(0, visibleByLengthOccurrences.length - 1),
      )
    : motifSelection.activeOccurrenceIndex;
  const legendSelectedMotifKey = isByLengthMode ? null : motifSelection.motifKey;
  const legendSelectedStates = isByLengthMode
    ? []
    : parseMotifKeyToStates(motifSelection.motifKey);

  return (
    <>
      <MotifTimelineLegend
        selectedMotifKey={legendSelectedMotifKey}
        selectedStates={legendSelectedStates}
        numLabels={k ?? 0}
        occurrencesTotal={legendOccurrencesTotal}
        activeOccurrenceIndex={legendActiveIndex}
        onPrev={isByLengthMode ? handleByLengthPrev : onMotifPrev}
        onNext={isByLengthMode ? handleByLengthNext : onMotifNext}
        tokenSelector={tokenSelector}
        onPlay={isByLengthMode ? handleByLengthPlay : undefined}
      />
      <div className="flex" style={{ height: 200 }}>
        <Spectrogram
          jobId={regionDetectionJobId}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
        >
          {showSingleOverlay && (
            <MotifHighlightOverlay
              occurrences={motifSelection.occurrences}
              activeOccurrenceIndex={motifSelection.activeOccurrenceIndex}
              colorIndex={motifColorIndex}
              numLabels={k ?? 0}
            />
          )}
          {isByLengthMode && visibleByLengthOccurrences.length > 0 && (
            <MotifHighlightOverlay
              occurrences={visibleByLengthOccurrences}
              activeOccurrenceIndex={legendActiveIndex}
              colorIndex={0}
              numLabels={k ?? 0}
              colorForMotifKey={colorForMotifKey}
            />
          )}
        </Spectrogram>
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
      {SHOW_CONFIDENCE_STRIPS && (
        <>
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
        </>
      )}
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
  onPlayMotif,
  byLengthLength,
  onSelectByLength,
  byLengthOccurrences,
  byLengthActiveIndex,
  onByLengthActiveIndexChange,
  availableLengths,
  motifsLoading,
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
  onPlayMotif: (occ: MotifOccurrence, idx: number) => void;
  byLengthLength: MotifTokenCount | null;
  onSelectByLength: (next: MotifTokenCount | null) => void;
  byLengthOccurrences: MotifOccurrence[];
  byLengthActiveIndex: number;
  onByLengthActiveIndexChange: (idx: number) => void;
  availableLengths: ReadonlySet<number>;
  motifsLoading: boolean;
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
                motifSelection={motifSelection}
                byLengthLength={byLengthLength}
                byLengthOccurrences={byLengthOccurrences}
                byLengthActiveIndex={byLengthActiveIndex}
                onByLengthActiveIndexChange={onByLengthActiveIndexChange}
                onSelectByLength={onSelectByLength}
                availableLengths={availableLengths}
                motifsLoading={motifsLoading}
                onMotifPrev={onMotifPrev}
                onMotifNext={onMotifNext}
                onPlayMotif={onPlayMotif}
                timelineHandleRef={timelineHandleRef}
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
          <ExemplarEventTypeChips exemplar={r} />
        </li>
      ))}
    </ul>
  );
}

function ExemplarEventTypeChips({ exemplar }: { exemplar: ExemplarRecord }) {
  const rawTypes = exemplar.extras?.event_types;
  const types: string[] = Array.isArray(rawTypes)
    ? (rawTypes.filter((v) => typeof v === "string") as string[])
    : [];
  const eventId = exemplar.extras?.event_id;

  if (types.length === 0) {
    return (
      <div
        className="flex flex-wrap gap-1 mt-1"
        data-testid="mt-exemplar-event-types"
      >
        <span
          className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-600 text-[10px] font-medium"
          data-testid="mt-exemplar-background-chip"
        >
          (background)
        </span>
      </div>
    );
  }
  return (
    <div
      className="flex flex-wrap gap-1 mt-1"
      data-testid="mt-exemplar-event-types"
    >
      {types.map((t) => {
        const chip = (
          <span
            key={t}
            className="px-1.5 py-0.5 rounded bg-blue-500 text-white text-[10px] font-medium"
            data-testid="mt-exemplar-type-chip"
          >
            {t}
          </span>
        );
        if (typeof eventId === "string" && eventId.length > 0) {
          return (
            <Link
              key={t}
              to={`/app/call-parsing/classify-review?event_id=${encodeURIComponent(eventId)}`}
              className="no-underline"
            >
              {chip}
            </Link>
          );
        }
        return chip;
      })}
    </div>
  );
}

function RegenerateMTLabelDistributionTrigger({
  mtJobId,
  kValues,
  segmentationJobId,
  boundClassifyId,
}: {
  mtJobId: string;
  kValues: number[];
  segmentationJobId: string | null;
  boundClassifyId: string | null;
}) {
  const k = useSelectedK(kValues);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chosenClassify, setChosenClassify] = useState<string>(
    boundClassifyId ?? "",
  );
  useEffect(() => {
    setChosenClassify(boundClassifyId ?? "");
  }, [boundClassifyId, open]);

  const classifyJobsQuery = useEventClassificationJobsForSegmentation(
    open ? segmentationJobId : null,
  );
  const classifyJobs = classifyJobsQuery.data ?? [];
  const mutation = useRegenerateMTLabelDistribution();

  const handleConfirm = () => {
    if (k == null) return;
    setError(null);
    mutation.mutate(
      {
        jobId: mtJobId,
        k,
        body:
          chosenClassify && chosenClassify !== boundClassifyId
            ? { event_classification_job_id: chosenClassify }
            : undefined,
      },
      {
        onSuccess: () => setOpen(false),
        onError: (err: unknown) => {
          setError(err instanceof Error ? err.message : String(err));
        },
      },
    );
  };

  return (
    <>
      <Button
        size="sm"
        variant="outline"
        onClick={() => setOpen(true)}
        disabled={k == null}
        data-testid="mt-regenerate-label-distribution"
      >
        Regenerate label distribution
      </Button>
      {open ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          data-testid="mt-regenerate-dialog"
          onClick={() => setOpen(false)}
        >
          <div
            className="bg-white rounded-md shadow-lg p-4 w-[440px] max-w-[90vw] space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-base font-medium">
              Regenerate label distribution for all k values
            </h3>
            <div className="text-sm space-y-2">
              <label className="block">
                <span className="text-xs font-medium text-slate-600">
                  Event Classification Job
                </span>
                <select
                  data-testid="mt-regenerate-classify-select"
                  className="w-full border rounded-md px-2 py-1 text-sm mt-1"
                  value={chosenClassify}
                  disabled={classifyJobsQuery.isLoading}
                  onChange={(e) => setChosenClassify(e.target.value)}
                >
                  {classifyJobs.length === 0 ? (
                    <option value="">— none —</option>
                  ) : null}
                  {classifyJobs.map((c) => (
                    <option key={c.id} value={c.id}>
                      #{c.id.slice(0, 8)}
                      {c.model_name ? ` · ${c.model_name}` : ""}
                      {c.n_events_classified != null
                        ? ` · ${c.n_events_classified} events`
                        : ""}
                      {c.id === boundClassifyId ? " · (current)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              {error ? (
                <div
                  className="text-red-700 text-xs"
                  data-testid="mt-regenerate-error"
                >
                  {error}
                </div>
              ) : null}
            </div>
            <div className="flex justify-end gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setOpen(false)}
                disabled={mutation.isPending}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleConfirm}
                disabled={
                  mutation.isPending || chosenClassify === "" || k == null
                }
                data-testid="mt-regenerate-confirm"
              >
                {mutation.isPending ? "Regenerating…" : "Regenerate"}
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </>
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
        {Object.entries(data.states).map(([stateKey, counts]) => {
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
  onPlayMotif,
  hideRowHighlight,
  onUserSelectMotif,
}: {
  jobId: string;
  kValues: number[];
  regionDetectionJobId: string | null;
  timelineHandleRef: React.RefObject<TimelinePlaybackHandle>;
  onSelectionChange: (selection: MotifPanelSelection) => void;
  activeOccurrenceIndex: number;
  onActiveOccurrenceChange: (idx: number) => void;
  onPlayMotif: (occurrence: MotifOccurrence, idx: number) => void;
  hideRowHighlight?: boolean;
  onUserSelectMotif?: (motifKey: string) => void;
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
      onPlayMotif={onPlayMotif}
      parent={{
        kind: "masked_transformer",
        maskedTransformerJobId: jobId,
        k,
      }}
      onSelectionChange={onSelectionChange}
      activeOccurrenceIndex={activeOccurrenceIndex}
      onActiveOccurrenceChange={onActiveOccurrenceChange}
      numLabels={k}
      hideRowHighlight={hideRowHighlight}
      onUserSelectMotif={onUserSelectMotif}
    />
  );
}
