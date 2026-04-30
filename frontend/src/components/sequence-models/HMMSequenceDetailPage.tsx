import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Plot from "react-plotly.js";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  type ExemplarRecord,
  type HMMSequenceJobDetail,
  type LabelDistribution,
  type OverlayResponse,
  type StateTierComposition,
  isHMMSequenceJobActive,
  useCancelHMMSequenceJob,
  useContinuousEmbeddingJob,
  useGenerateInterpretations,
  useHMMDwell,
  useHMMExemplars,
  useHMMLabelDistribution,
  useHMMOverlay,
  useHMMSequenceJob,
  useHMMStates,
  useHMMTransitions,
} from "@/api/sequenceModels";
import { regionTileUrl, regionAudioSliceUrl } from "@/api/client";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { PlaybackControls } from "@/components/timeline/controls/PlaybackControls";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import { RegionBoundaryMarkers } from "@/components/timeline/overlays/RegionBoundaryMarkers";
import { STATE_COLORS } from "./constants";
import { SpanNavBar, type SpanInfo, type RegionGroup } from "./SpanNavBar";
import { HMMStateBar, type ViterbiWindow } from "./HMMStateBar";

function StateTimeline({
  spanItems,
  nStates,
}: {
  spanItems: Record<string, unknown>[];
  nStates: number;
}) {
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
      const start = row.start_timestamp as number;
      const end = row.end_timestamp as number;
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

function binRunLengths(runLengths: number[]): { x: number[]; y: number[] } {
  if (runLengths.length === 0) return { x: [], y: [] };
  const freq = new Map<number, number>();
  for (const len of runLengths) {
    freq.set(len, (freq.get(len) ?? 0) + 1);
  }
  const sorted = Array.from(freq.entries()).sort((a, b) => a[0] - b[0]);
  return { x: sorted.map(([k]) => k), y: sorted.map(([, v]) => v) };
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
        const rawRuns = histograms[String(s)] ?? [];
        const binned = binRunLengths(rawRuns);
        return (
          <div key={s}>
            <Plot
              data={[
                {
                  x: binned.x,
                  y: binned.y,
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

function PcaUmapScatter({
  data,
  nStates,
}: {
  data: OverlayResponse;
  nStates: number;
}) {
  const [mode, setMode] = useState<"pca" | "umap">("pca");
  const xKey = mode === "pca" ? "pca_x" : "umap_x";
  const yKey = mode === "pca" ? "pca_y" : "umap_y";

  const traces = useMemo(() => {
    const byState: Record<number, { x: number[]; y: number[]; text: string[] }> = {};
    for (let s = 0; s < nStates; s++) {
      byState[s] = { x: [], y: [], text: [] };
    }
    for (const pt of data.items) {
      const s = pt.viterbi_state;
      if (!byState[s]) byState[s] = { x: [], y: [], text: [] };
      byState[s].x.push(pt[xKey]);
      byState[s].y.push(pt[yKey]);
      byState[s].text.push(
        `S${s} p=${pt.max_state_probability.toFixed(2)} t=${pt.start_timestamp.toFixed(0)}s`,
      );
    }
    return Object.entries(byState).map(([s, d]) => ({
      x: d.x,
      y: d.y,
      text: d.text,
      mode: "markers" as const,
      type: "scattergl" as const,
      marker: {
        color: STATE_COLORS[Number(s) % STATE_COLORS.length],
        size: 4,
        opacity: 0.7,
      },
      name: `State ${s}`,
      hoverinfo: "text" as const,
    }));
  }, [data, xKey, yKey, nStates]);

  return (
    <div data-testid="hmm-pca-umap-scatter">
      <div className="flex gap-2 mb-2">
        <Button
          size="sm"
          variant={mode === "pca" ? "default" : "outline"}
          onClick={() => setMode("pca")}
        >
          PCA
        </Button>
        <Button
          size="sm"
          variant={mode === "umap" ? "default" : "outline"}
          onClick={() => setMode("umap")}
        >
          UMAP
        </Button>
      </div>
      <Plot
        data={traces}
        layout={{
          height: 400,
          margin: { l: 50, r: 20, t: 10, b: 40 },
          xaxis: { title: { text: `${mode.toUpperCase()} 1` } },
          yaxis: { title: { text: `${mode.toUpperCase()} 2` } },
          showlegend: true,
          legend: { orientation: "h", y: 1.15 },
        }}
        config={{ responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function LabelDistributionChart({
  data,
}: {
  data: LabelDistribution;
}) {
  const traces = useMemo(() => {
    const allLabels = new Set<string>();
    for (const counts of Object.values(data.states)) {
      for (const lbl of Object.keys(counts)) {
        allLabels.add(lbl);
      }
    }
    const sortedLabels = Array.from(allLabels).sort();
    const stateKeys = Array.from({ length: data.n_states }, (_, i) => String(i));
    const xLabels = stateKeys.map((s) => `S${s}`);

    return sortedLabels.map((lbl, i) => ({
      x: xLabels,
      y: stateKeys.map((s) => data.states[s]?.[lbl] ?? 0),
      name: lbl,
      type: "bar" as const,
      marker: {
        color:
          lbl === "unlabeled"
            ? "#d1d5db"
            : STATE_COLORS[i % STATE_COLORS.length],
      },
    }));
  }, [data]);

  return (
    <div data-testid="hmm-label-distribution">
      <Plot
        data={traces}
        layout={{
          barmode: "stack",
          height: 300,
          margin: { l: 50, r: 20, t: 10, b: 40 },
          xaxis: { title: { text: "State" } },
          yaxis: { title: { text: "Count" } },
          showlegend: true,
          legend: { orientation: "h", y: 1.15 },
        }}
        config={{ responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function TierCompositionStrip({
  composition,
}: {
  composition: StateTierComposition[];
}) {
  const traces = useMemo(() => {
    const states = composition.map((c) => `S${c.state}`);
    const tiers: { name: string; key: keyof StateTierComposition; color: string }[] =
      [
        { name: "event_core", key: "event_core", color: "#3b82f6" },
        { name: "near_event", key: "near_event", color: "#fbbf24" },
        { name: "background", key: "background", color: "#94a3b8" },
      ];
    return tiers.map((t) => ({
      x: states,
      y: composition.map((c) => Number(c[t.key])),
      type: "bar" as const,
      name: t.name,
      marker: { color: t.color },
    }));
  }, [composition]);

  return (
    <div data-testid="hmm-tier-composition-strip">
      <Plot
        data={traces}
        layout={{
          barmode: "stack",
          height: 220,
          margin: { l: 50, r: 20, t: 10, b: 40 },
          xaxis: { title: { text: "State" } },
          yaxis: {
            title: { text: "Tier composition" },
            tickformat: ".0%",
            range: [0, 1],
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


function ExemplarGallery({
  states,
  nStates,
}: {
  states: Record<string, ExemplarRecord[]>;
  nStates: number;
}) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set([0]));

  const toggle = (s: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  };

  const typeLabel: Record<string, string> = {
    high_confidence: "High Confidence",
    mean_nearest: "Nearest to Centroid",
    boundary: "Boundary (Low Confidence)",
  };

  return (
    <div data-testid="hmm-exemplar-gallery" className="space-y-2">
      {Array.from({ length: nStates }, (_, s) => {
        const records = states[String(s)] ?? [];
        const isOpen = expanded.has(s);
        const grouped: Record<string, ExemplarRecord[]> = {};
        for (const r of records) {
          (grouped[r.exemplar_type] ??= []).push(r);
        }

        return (
          <div key={s} className="border rounded-md">
            <button
              type="button"
              className="w-full text-left px-3 py-2 text-sm font-medium flex items-center gap-2"
              onClick={() => toggle(s)}
            >
              <span
                className="w-3 h-3 rounded-full inline-block"
                style={{
                  backgroundColor: STATE_COLORS[s % STATE_COLORS.length],
                }}
              />
              State {s}
              <span className="text-slate-400 ml-auto">
                {isOpen ? "▾" : "▸"}
              </span>
            </button>
            {isOpen && (
              <div className="px-3 pb-3 space-y-2">
                {records.length === 0 && (
                  <div className="text-xs text-slate-400">No exemplars</div>
                )}
                {["high_confidence", "mean_nearest", "boundary"].map(
                  (etype) => {
                    const items = grouped[etype];
                    if (!items?.length) return null;
                    return (
                      <div key={etype}>
                        <div className="text-xs font-medium text-slate-500 mb-1">
                          {typeLabel[etype] ?? etype}
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                          {items.map((ex, i) => (
                            <div
                              key={i}
                              className="border rounded p-2 text-xs space-y-0.5"
                            >
                              <div>
                                audio:{" "}
                                {ex.audio_file_id == null
                                  ? "hydrophone"
                                  : String(ex.audio_file_id).slice(0, 8)}
                              </div>
                              <div>
                                {ex.start_timestamp.toFixed(1)}s –{" "}
                                {ex.end_timestamp.toFixed(1)}s
                              </div>
                              <div>
                                p={ex.max_state_probability.toFixed(3)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  },
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function HMMTimelineBody({
  regionDetectionJobId,
  tileUrlBuilder,
  eventSpan,
  stateItems,
  nStates,
  scrollToCenter,
}: {
  regionDetectionJobId: string;
  tileUrlBuilder: (jid: string, zoomLevel: string, tileIndex: number, freqMin: number, freqMax: number) => string;
  eventSpan: SpanInfo;
  stateItems: ViterbiWindow[];
  nStates: number;
  scrollToCenter: { target: number; seq: number } | undefined;
}) {
  const ctx = useTimelineContext();

  const scrollSeqRef = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!scrollToCenter || scrollToCenter.seq === scrollSeqRef.current) return;
    scrollSeqRef.current = scrollToCenter.seq;
    ctx.seekTo(scrollToCenter.target);
  }, [scrollToCenter, ctx]);

  return (
    <>
      <div className="flex" style={{ height: 160 }}>
        <Spectrogram
          jobId={regionDetectionJobId}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
        >
          <RegionBoundaryMarkers
            startEpoch={eventSpan.startTimestamp}
            endEpoch={eventSpan.endTimestamp}
            dimOutside={false}
            lineColor="rgba(255, 255, 255, 0.95)"
            lineStyle="solid"
          />
        </Spectrogram>
      </div>
      <HMMStateBar
        items={stateItems}
        nStates={nStates}
        currentRegion={eventSpan}
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

export function HMMSequenceDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data, isLoading, error } = useHMMSequenceJob(jobId ?? null);
  const cancelMutation = useCancelHMMSequenceJob();

  const isComplete = data?.job.status === "complete";
  const cejId = data?.job.continuous_embedding_job_id ?? null;
  const { data: cejDetail } = useContinuousEmbeddingJob(isComplete ? cejId : null);

  const { data: statesData } = useHMMStates(
    jobId ?? null,
    0,
    50000,
    isComplete,
  );
  const { data: transData } = useHMMTransitions(jobId ?? null, isComplete);
  const { data: dwellData } = useHMMDwell(jobId ?? null, isComplete);
  const { data: overlayData } = useHMMOverlay(jobId ?? null, isComplete);
  const { data: labelDistData } = useHMMLabelDistribution(jobId ?? null, isComplete);
  const { data: exemplarsData } = useHMMExemplars(jobId ?? null, isComplete);
  const generateMutation = useGenerateInterpretations();

  const sourceKind = data?.source_kind ?? "surfperch";
  const isCrnnSource = sourceKind === "region_crnn";
  // CRNN states.parquet groups rows by region_id (string); SurfPerch by
  // merged_span_id (int). Every Viterbi-state filter must dispatch on this.
  const groupKeyCol = isCrnnSource ? "region_id" : "merged_span_id";

  const spanIds = useMemo<(string | number)[]>(() => {
    if (!statesData?.items) return [];
    // Track min start_timestamp per group key so CRNN region UUIDs sort
    // chronologically (timeline order), not lexicographically.
    const minStart = new Map<string | number, number>();
    for (const row of statesData.items) {
      const key = row[groupKeyCol] as string | number | undefined;
      if (key == null) continue;
      const t = row.start_timestamp as number;
      const existing = minStart.get(key);
      if (existing === undefined || t < existing) {
        minStart.set(key, t);
      }
    }
    const keys = Array.from(minStart.keys());
    if (isCrnnSource) {
      return keys.sort(
        (a, b) => (minStart.get(a) ?? 0) - (minStart.get(b) ?? 0),
      );
    }
    return (keys as number[]).sort((a, b) => a - b);
  }, [statesData, groupKeyCol, isCrnnSource]);

  const spans: SpanInfo[] = useMemo(() => {
    if (!statesData?.items || spanIds.length === 0) return [];
    const manifestSpans = cejDetail?.manifest?.spans ?? [];
    const manifestRegions = cejDetail?.manifest?.regions ?? [];
    return spanIds.map((id) => {
      let minT = Infinity;
      let maxT = -Infinity;
      for (const row of statesData.items) {
        if ((row[groupKeyCol] as string | number) !== id) continue;
        const s = row.start_timestamp as number;
        const e = row.end_timestamp as number;
        if (s < minT) minT = s;
        if (e > maxT) maxT = e;
      }
      if (isCrnnSource) {
        const mr = manifestRegions.find((r) => r.region_id === id);
        return {
          id,
          eventId: "",
          regionId: String(id),
          startTimestamp: mr?.start_timestamp ?? minT,
          endTimestamp: mr?.end_timestamp ?? maxT,
        };
      }
      const ms = manifestSpans.find((s) => s.merged_span_id === (id as number));
      return {
        id,
        eventId: ms?.event_id ?? "",
        regionId: ms?.region_id ?? "",
        startTimestamp: minT,
        endTimestamp: maxT,
      };
    });
  }, [statesData, spanIds, cejDetail, groupKeyCol, isCrnnSource]);

  const regions: RegionGroup[] = useMemo(() => {
    // CRNN: each span IS a region — region-level nav is redundant with
    // span-level nav, so suppress it.
    if (isCrnnSource || spans.length === 0) return [];
    const groups: RegionGroup[] = [];
    let currentRegion = spans[0].regionId;
    let startIdx = 0;
    for (let i = 1; i < spans.length; i++) {
      if (spans[i].regionId !== currentRegion) {
        groups.push({ regionId: currentRegion, startIndex: startIdx, endIndex: i - 1 });
        currentRegion = spans[i].regionId;
        startIdx = i;
      }
    }
    groups.push({ regionId: currentRegion, startIndex: startIdx, endIndex: spans.length - 1 });
    return groups;
  }, [spans, isCrnnSource]);

  const [selectedSpan, setSelectedSpan] = useState<string | number | null>(null);
  const [timelineZoom, setTimelineZoom] = useState<string | null>(null);
  const activeSpan = selectedSpan ?? spanIds[0] ?? 0;

  useEffect(() => {
    setTimelineZoom(null);
    setSelectedSpan(null);
  }, [jobId]);

  const activeSpanIndex = useMemo(
    () => Math.max(0, spanIds.indexOf(activeSpan)),
    [spanIds, activeSpan],
  );
  const activeTimelineSpan = spans[activeSpanIndex] ?? null;

  const activeRegionIndex = useMemo(() => {
    for (let i = 0; i < regions.length; i++) {
      if (activeSpanIndex >= regions[i].startIndex && activeSpanIndex <= regions[i].endIndex) {
        return i;
      }
    }
    return 0;
  }, [activeSpanIndex, regions]);

  const handlePrevEvent = useCallback(() => {
    const newIdx = Math.max(0, activeSpanIndex - 1);
    setSelectedSpan(spanIds[newIdx] ?? null);
  }, [activeSpanIndex, spanIds]);
  const handleNextEvent = useCallback(() => {
    const newIdx = Math.min(spanIds.length - 1, activeSpanIndex + 1);
    setSelectedSpan(spanIds[newIdx] ?? null);
  }, [activeSpanIndex, spanIds]);

  const handlePrevRegion = useCallback(() => {
    const newRegIdx = Math.max(0, activeRegionIndex - 1);
    const firstEventIdx = regions[newRegIdx]?.startIndex ?? 0;
    setSelectedSpan(spanIds[firstEventIdx] ?? null);
  }, [activeRegionIndex, regions, spanIds]);
  const handleNextRegion = useCallback(() => {
    const newRegIdx = Math.min(regions.length - 1, activeRegionIndex + 1);
    const firstEventIdx = regions[newRegIdx]?.startIndex ?? 0;
    setSelectedSpan(spanIds[firstEventIdx] ?? null);
  }, [activeRegionIndex, regions, spanIds]);

  const regionDetectionJobId = data?.region_detection_job_id ?? "";
  const regionStartTimestamp =
    data?.region_start_timestamp ?? activeTimelineSpan?.startTimestamp ?? 0;
  const regionEndTimestamp =
    data?.region_end_timestamp ?? activeTimelineSpan?.endTimestamp ?? 0;

  const defaultZoom = useMemo(() => {
    if (!activeTimelineSpan) return REVIEW_ZOOM[0].key;
    const duration =
      activeTimelineSpan.endTimestamp - activeTimelineSpan.startTimestamp;
    let best = REVIEW_ZOOM[0];
    for (const preset of REVIEW_ZOOM) {
      if (preset.span <= duration * 1.2) best = preset;
    }
    return best.key;
  }, [activeTimelineSpan]);

  const activeSpanRows = useMemo<Record<string, unknown>[]>(() => {
    if (!statesData?.items || !activeTimelineSpan) return [];
    return statesData.items.filter(
      (r) => (r[groupKeyCol] as string | number) === activeTimelineSpan.id,
    );
  }, [statesData, activeTimelineSpan, groupKeyCol]);

  const timelineStateItems: ViterbiWindow[] = useMemo(
    () =>
      (statesData?.items ?? []).map((r) => ({
        start_timestamp: r.start_timestamp as number,
        end_timestamp: r.end_timestamp as number,
        viterbi_state: r.viterbi_state as number,
        max_state_probability: r.max_state_probability as number,
      })),
    [statesData],
  );

  const hmmTileUrlBuilder = useCallback(
    (
      _jid: string,
      zoomLevel: string,
      tileIndex: number,
      _freqMin: number,
      _freqMax: number,
    ) => {
      return regionTileUrl(regionDetectionJobId, zoomLevel, tileIndex);
    },
    [regionDetectionJobId],
  );

  const scrollSeqRef = useRef(0);
  const [scrollToCenter, setScrollToCenter] = useState<
    { target: number; seq: number } | undefined
  >(undefined);

  const wrappedPrevEvent = useCallback(() => {
    handlePrevEvent();
  }, [handlePrevEvent]);
  const wrappedNextEvent = useCallback(() => {
    handleNextEvent();
  }, [handleNextEvent]);

  const activeTimelineSpanKey =
    activeTimelineSpan == null ? null : String(activeTimelineSpan.id);
  const activeTimelineSpanStart = activeTimelineSpan?.startTimestamp;
  const activeTimelineSpanEnd = activeTimelineSpan?.endTimestamp;
  useEffect(() => {
    if (
      activeTimelineSpanKey == null ||
      activeTimelineSpanStart == null ||
      activeTimelineSpanEnd == null
    )
      return;
    scrollSeqRef.current += 1;
    setScrollToCenter({
      target: (activeTimelineSpanStart + activeTimelineSpanEnd) / 2,
      seq: scrollSeqRef.current,
    });
  }, [activeTimelineSpanKey, activeTimelineSpanStart, activeTimelineSpanEnd]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      )
        return;
      if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        wrappedPrevEvent();
      } else if (e.key === "d" || e.key === "D") {
        e.preventDefault();
        wrappedNextEvent();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [wrappedPrevEvent, wrappedNextEvent]);

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
  const tierComposition = data.tier_composition ?? null;
  const itemLabel: "Event" | "Region" = isCrnnSource ? "Region" : "Event";
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
          <CardTitle data-testid="hmm-detail-id">
            {job.id}
            <span
              className="ml-3 text-xs uppercase tracking-wide text-slate-500"
              data-testid="hmm-detail-source-kind"
            >
              {isCrnnSource ? "CRNN" : "SurfPerch"}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div>
            <span className="font-medium">Status:</span>{" "}
            <span data-testid="hmm-detail-status">{job.status}</span>
          </div>
          <div className="grid grid-cols-5 gap-2">
            <div>
              <span className="font-medium text-slate-500">states</span>
              <div>{job.n_states}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">pca_dims</span>
              <div>{job.pca_dims}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">pca_whiten</span>
              <div>{job.pca_whiten ? "yes" : "no"}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">l2_normalize</span>
              <div>{job.l2_normalize ? "yes" : "no"}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">cov_type</span>
              <div>{job.covariance_type}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">n_iter</span>
              <div>{job.n_iter}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">tol</span>
              <div>{job.tol}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">seed</span>
              <div>{job.random_seed}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">min_seq_len</span>
              <div>{job.min_sequence_length_frames}</div>
            </div>
            <div>
              <span className="font-medium text-slate-500">library</span>
              <div>{job.library}</div>
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

      {isComplete && statesData && activeTimelineSpan && regionDetectionJobId && (
        <Card>
          <CardHeader>
            <CardTitle>HMM State Timeline Viewer</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2" data-testid="hmm-timeline-viewer">
            <SpanNavBar
              spans={spans}
              regions={regions}
              activeIndex={activeSpanIndex}
              activeRegionIndex={activeRegionIndex}
              onPrevEvent={wrappedPrevEvent}
              onNextEvent={wrappedNextEvent}
              onPrevRegion={handlePrevRegion}
              onNextRegion={handleNextRegion}
              itemLabel={itemLabel}
            />
            <TimelineProvider
              key={`hmm-timeline-${regionDetectionJobId}`}
              jobStart={regionStartTimestamp}
              jobEnd={regionEndTimestamp}
              zoomLevels={REVIEW_ZOOM}
              defaultZoom={timelineZoom ?? defaultZoom}
              onZoomChange={setTimelineZoom}
              playback="slice"
              audioUrlBuilder={(startEpoch, durationSec) =>
                regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec)
              }
            >
              <HMMTimelineBody
                regionDetectionJobId={regionDetectionJobId}
                tileUrlBuilder={hmmTileUrlBuilder}
                eventSpan={activeTimelineSpan}
                stateItems={timelineStateItems}
                nStates={job.n_states}
                scrollToCenter={scrollToCenter}
              />
            </TimelineProvider>
          </CardContent>
        </Card>
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
                  value={String(activeSpan)}
                  onChange={(e) => {
                    const raw = e.target.value;
                    setSelectedSpan(isCrnnSource ? raw : Number(raw));
                  }}
                >
                  {spanIds.map((id, idx) => (
                    <option key={String(id)} value={String(id)}>
                      {isCrnnSource
                        ? `Region ${idx + 1} (${String(id).slice(0, 8)})`
                        : `Span ${id}`}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <StateTimeline
              spanItems={activeSpanRows}
              nStates={job.n_states}
            />
          </CardContent>
        </Card>
      )}

      {isComplete && overlayData && (
        <Card>
          <CardHeader>
            <CardTitle>PCA / UMAP Overlay</CardTitle>
          </CardHeader>
          <CardContent>
            <PcaUmapScatter data={overlayData} nStates={job.n_states} />
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

      {isComplete && sourceKind === "region_crnn" && tierComposition && (
        <Card>
          <CardHeader>
            <CardTitle>Per-State Tier Composition</CardTitle>
          </CardHeader>
          <CardContent>
            <TierCompositionStrip composition={tierComposition} />
          </CardContent>
        </Card>
      )}

      {isComplete && (
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle>Label Distribution</CardTitle>
              <Button
                size="sm"
                variant="outline"
                disabled={generateMutation.isPending}
                onClick={() => generateMutation.mutate(job.id)}
                data-testid="hmm-generate-interpretations"
              >
                {generateMutation.isPending ? "Generating…" : "Refresh"}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {labelDistData ? (
              <LabelDistributionChart data={labelDistData} />
            ) : (
              <div className="text-sm text-slate-500">
                No label distribution available. Click Refresh to generate.
              </div>
            )}
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

      {isComplete && exemplarsData && (
        <Card>
          <CardHeader>
            <CardTitle>State Exemplars</CardTitle>
          </CardHeader>
          <CardContent>
            <ExemplarGallery
              states={exemplarsData.states}
              nStates={exemplarsData.n_states}
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
