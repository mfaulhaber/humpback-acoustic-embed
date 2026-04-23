import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useWindowClassificationJobs,
  useWindowScores,
  useWindowScoreCorrections,
  useUpsertWindowScoreCorrections,
  useRegionJobRegions,
} from "@/hooks/queries/useCallParsing";
import { useVocClassifierModel } from "@/hooks/queries/useVocalization";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { regionTileUrl, regionAudioSliceUrl } from "@/api/client";
import type {
  Region,
  WindowClassificationJob,
  WindowScoreRow,
  WindowScoreCorrection,
} from "@/api/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Square,
  Save,
  X,
  Plus,
} from "lucide-react";
import { toast } from "@/components/ui/use-toast";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import type { TimelinePlaybackHandle } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { useOverlayContext } from "@/components/timeline/overlays/OverlayContext";
import type { GradientStops } from "@/components/timeline/spectrogram/ConfidenceStrip";
import { typeColor } from "./TypePalette";

const WINDOW_SIZE_SEC = 5.0;
const HOP_SEC = 1.0;
const STRIP_HEIGHT = 26;

const HEATMAP_GRADIENT: GradientStops = [
  [0.0, "#0a2040"],
  [0.15, "#2060a0"],
  [0.3, "#20a060"],
  [0.45, "#40c040"],
  [0.6, "#a0d020"],
  [0.75, "#d0c020"],
  [0.85, "#d06020"],
  [0.95, "#c02020"],
  [1.0, "#801010"],
] as const;

interface PendingCorrection {
  time_sec: number;
  region_id: string;
  correction_type: "add" | "remove";
  type_name: string;
}

function correctionKey(c: {
  time_sec: number;
  region_id: string;
  type_name: string;
}) {
  return `${c.time_sec}:${c.region_id}:${c.type_name}`;
}

export function WindowClassifyReviewWorkspace({
  initialJobId,
}: {
  initialJobId?: string;
}) {
  const { data: wcJobs = [] } = useWindowClassificationJobs(0);
  const { data: hydrophones = [] } = useHydrophones();

  const completeJobs = useMemo(
    () => wcJobs.filter((j) => j.status === "complete"),
    [wcJobs],
  );

  const [selectedJobId, setSelectedJobId] = useState<string | null>(
    initialJobId ?? null,
  );

  useEffect(() => {
    if (initialJobId && completeJobs.some((j) => j.id === initialJobId)) {
      setSelectedJobId(initialJobId);
    }
  }, [initialJobId, completeJobs]);

  const selectedJob =
    completeJobs.find((j) => j.id === selectedJobId) ?? null;

  const regionDetectionJobId = selectedJob?.region_detection_job_id ?? null;
  const { data: regions = [] } = useRegionJobRegions(regionDetectionJobId);
  const { data: allScoreRows = [] } = useWindowScores(
    selectedJobId ?? undefined,
  );
  const { data: savedCorrections = [] } = useWindowScoreCorrections(
    selectedJobId ?? undefined,
  );

  const { data: vocModel } = useVocClassifierModel(
    selectedJob?.vocalization_model_id ?? null,
  );

  const vocabulary: string[] = useMemo(() => {
    if (selectedJob?.vocabulary_snapshot) {
      try {
        return JSON.parse(selectedJob.vocabulary_snapshot) as string[];
      } catch {
        /* ignore */
      }
    }
    return vocModel?.vocabulary_snapshot ?? [];
  }, [selectedJob?.vocabulary_snapshot, vocModel?.vocabulary_snapshot]);

  const perClassThresholds: Record<string, number> = useMemo(
    () => vocModel?.per_class_thresholds ?? {},
    [vocModel?.per_class_thresholds],
  );

  // Region navigation
  const [regionIndex, setRegionIndex] = useState(0);

  useEffect(() => {
    setRegionIndex(0);
    setSelectedWindowTime(null);
    setPendingCorrections(new Map());
  }, [selectedJobId]);

  const currentRegion = regions[regionIndex] ?? null;

  // Scores for current region
  const regionScores = useMemo(
    () =>
      currentRegion
        ? allScoreRows.filter(
            (s) => s.region_id === currentRegion.region_id,
          )
        : [],
    [allScoreRows, currentRegion],
  );

  // Type selector for confidence strip
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [thresholdOverride, setThresholdOverride] = useState<string>("");

  const effectiveThreshold = useMemo(() => {
    const parsed = parseFloat(thresholdOverride);
    if (!isNaN(parsed)) return parsed;
    if (selectedType && perClassThresholds[selectedType] != null) {
      return perClassThresholds[selectedType];
    }
    const thresholds = Object.values(perClassThresholds);
    if (thresholds.length > 0) {
      return thresholds.reduce((a, b) => a + b, 0) / thresholds.length;
    }
    return 0.5;
  }, [thresholdOverride, selectedType, perClassThresholds]);

  // Timeline extent covers all regions so the user can scroll freely.
  const timelineEnd = useMemo(() => {
    if (regions.length === 0) return 0;
    return Math.max(...regions.map((r) => r.padded_end_sec));
  }, [regions]);

  // Build confidence strip scores array covering the full timeline (jobStart=0
  // to timelineEnd) using ALL score rows so strips render for every region.
  const stripScores = useMemo(() => {
    if (allScoreRows.length === 0 || timelineEnd === 0) return [];
    const n = Math.ceil(timelineEnd / HOP_SEC);
    const arr: (number | null)[] = new Array(n).fill(null);

    const lookup = new Map<number, WindowScoreRow>();
    for (const row of allScoreRows) {
      lookup.set(Math.round(row.time_sec * 10) / 10, row);
    }

    for (let i = 0; i < n; i++) {
      const t = Math.round(i * HOP_SEC * 10) / 10;
      const row = lookup.get(t);
      if (!row) continue;
      if (selectedType) {
        arr[i] = row.scores[selectedType] ?? null;
      } else {
        const values = Object.values(row.scores);
        arr[i] = values.length > 0 ? Math.max(...values) : null;
      }
    }
    return arr;
  }, [allScoreRows, timelineEnd, selectedType]);

  // Selected window
  const [selectedWindowTime, setSelectedWindowTime] = useState<number | null>(
    null,
  );

  const selectedScoreRow = useMemo(
    () =>
      selectedWindowTime != null
        ? regionScores.find(
            (s) =>
              Math.abs(s.time_sec - selectedWindowTime) < HOP_SEC * 0.5,
          ) ?? null
        : null,
    [selectedWindowTime, regionScores],
  );

  // Pending corrections: Map<correctionKey, PendingCorrection>
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, PendingCorrection>
  >(new Map());

  const isDirty = pendingCorrections.size > 0;

  // Merged corrections (saved + pending, pending overrides)
  const mergedCorrections = useMemo(() => {
    const map = new Map<string, "add" | "remove">();
    for (const c of savedCorrections) {
      map.set(correctionKey(c), c.correction_type);
    }
    for (const [key, c] of pendingCorrections) {
      map.set(key, c.correction_type);
    }
    return map;
  }, [savedCorrections, pendingCorrections]);

  // Playback
  const playbackRef = useRef<TimelinePlaybackHandle>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [userZoom, setUserZoom] = useState("30s");

  // Region navigation handlers
  const goPrevRegion = useCallback(() => {
    setRegionIndex((i) => Math.max(0, i - 1));
    setSelectedWindowTime(null);
  }, []);
  const goNextRegion = useCallback(() => {
    setRegionIndex((i) => Math.min(regions.length - 1, i + 1));
    setSelectedWindowTime(null);
  }, [regions.length]);

  // Playback
  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      playbackRef.current?.pause();
    } else if (selectedWindowTime != null) {
      playbackRef.current?.play(selectedWindowTime, WINDOW_SIZE_SEC);
    }
  }, [isPlaying, selectedWindowTime]);

  // Badge click: toggle add/remove correction
  const handleBadgeClick = useCallback(
    (typeName: string) => {
      if (selectedScoreRow == null || !currentRegion) return;
      const key = correctionKey({
        time_sec: selectedScoreRow.time_sec,
        region_id: currentRegion.region_id,
        type_name: typeName,
      });
      const existing = mergedCorrections.get(key);
      const score = selectedScoreRow.scores[typeName] ?? 0;
      const threshold = perClassThresholds[typeName] ?? effectiveThreshold;
      const isAbove = score >= threshold;

      setPendingCorrections((prev) => {
        const next = new Map(prev);
        if (existing === "remove") {
          // Undo removal
          next.delete(key);
        } else if (isAbove && existing !== "add") {
          // Mark as removed
          next.set(key, {
            time_sec: selectedScoreRow.time_sec,
            region_id: currentRegion.region_id,
            correction_type: "remove",
            type_name: typeName,
          });
        } else {
          // Toggle off add if already added
          next.delete(key);
        }
        return next;
      });
    },
    [
      selectedScoreRow,
      currentRegion,
      mergedCorrections,
      perClassThresholds,
      effectiveThreshold,
    ],
  );

  // Add type via popover
  const handleAddType = useCallback(
    (typeName: string) => {
      if (selectedScoreRow == null || !currentRegion) return;
      const key = correctionKey({
        time_sec: selectedScoreRow.time_sec,
        region_id: currentRegion.region_id,
        type_name: typeName,
      });
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        next.set(key, {
          time_sec: selectedScoreRow.time_sec,
          region_id: currentRegion.region_id,
          correction_type: "add",
          type_name: typeName,
        });
        return next;
      });
    },
    [selectedScoreRow, currentRegion],
  );

  // Save
  const upsertMutation = useUpsertWindowScoreCorrections();

  const handleSave = useCallback(() => {
    if (!selectedJobId || !isDirty) return;
    const corrections = Array.from(pendingCorrections.values());
    upsertMutation.mutate(
      { jobId: selectedJobId, corrections },
      {
        onSuccess: () => {
          setPendingCorrections(new Map());
          toast({
            title: "Corrections saved",
            description: `${corrections.length} correction${corrections.length !== 1 ? "s" : ""} saved.`,
          });
        },
        onError: (err) => {
          toast({
            title: "Failed to save corrections",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  }, [selectedJobId, isDirty, pendingCorrections, upsertMutation]);

  const handleCancel = useCallback(() => {
    if (isDirty && !window.confirm("Discard unsaved corrections?")) return;
    setPendingCorrections(new Map());
  }, [isDirty]);

  // beforeunload warning
  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (
        el.tagName === "INPUT" ||
        el.tagName === "TEXTAREA" ||
        el.tagName === "SELECT"
      )
        return;

      switch (e.code) {
        case "BracketLeft":
        case "KeyA":
          e.preventDefault();
          goPrevRegion();
          break;
        case "BracketRight":
        case "KeyD":
          e.preventDefault();
          goNextRegion();
          break;
        case "Space":
          e.preventDefault();
          togglePlayback();
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goPrevRegion, goNextRegion, togglePlayback]);

  // Job label
  const jobLabel = useCallback(
    (job: WindowClassificationJob) => {
      const shortId = job.id.slice(0, 8);
      const wc = job.window_count ?? 0;
      // Try to find hydrophone name via upstream region job
      // We don't have the region job data per-job here, so use a simpler label
      return `${shortId} — ${wc} windows`;
    },
    [],
  );

  // Detail: compute effective labels for selected window
  const selectedWindowLabels = useMemo(() => {
    if (!selectedScoreRow || !currentRegion) return [];
    return vocabulary.map((typeName) => {
      const score = selectedScoreRow.scores[typeName] ?? 0;
      const threshold = perClassThresholds[typeName] ?? effectiveThreshold;
      const aboveThreshold = score >= threshold;
      const addKey = correctionKey({
        time_sec: selectedScoreRow.time_sec,
        region_id: currentRegion.region_id,
        type_name: typeName,
      });
      const correction = mergedCorrections.get(addKey);
      const isPendingAdd = pendingCorrections.has(addKey);
      return {
        typeName,
        score,
        threshold,
        aboveThreshold,
        correction,
        isPending: isPendingAdd,
      };
    });
  }, [
    selectedScoreRow,
    currentRegion,
    vocabulary,
    perClassThresholds,
    effectiveThreshold,
    mergedCorrections,
    pendingCorrections,
  ]);

  // Types not in vocabulary (for the plus popover)
  const [showAddPopover, setShowAddPopover] = useState(false);
  const addableTypes = useMemo(() => {
    if (!selectedScoreRow) return [];
    const present = new Set(vocabulary);
    return [] as string[]; // Vocabulary already covers all types from the model
  }, [selectedScoreRow, vocabulary]);

  return (
    <div className="space-y-4">
      {/* Job selector */}
      <div className="flex items-center gap-4">
        <label htmlFor="wc-review-job" className="text-sm font-medium">
          Job
        </label>
        <select
          id="wc-review-job"
          className="rounded-md border bg-background px-3 py-2 text-sm"
          value={selectedJobId ?? ""}
          onChange={(e) => setSelectedJobId(e.target.value || null)}
        >
          <option value="">Select a completed window classification job</option>
          {completeJobs.map((j) => (
            <option key={j.id} value={j.id}>
              {jobLabel(j)}
            </option>
          ))}
        </select>
        {isDirty && (
          <span className="text-xs text-yellow-500">
            {pendingCorrections.size} unsaved change
            {pendingCorrections.size !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Workspace */}
      {selectedJob ? (
        <div className="rounded-md border">
          {/* Toolbar */}
          <div className="flex items-center justify-between px-4 py-2 border-b">
            <div className="flex items-center gap-2">
              {/* Region navigator */}
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goPrevRegion}
                disabled={regionIndex === 0}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-xs text-muted-foreground tabular-nums min-w-[100px] text-center">
                Region {regions.length > 0 ? regionIndex + 1 : 0} of{" "}
                {regions.length}
                {currentRegion && (
                  <span className="ml-1.5 text-muted-foreground/70">
                    ({currentRegion.start_sec.toFixed(1)}s –{" "}
                    {currentRegion.end_sec.toFixed(1)}s)
                  </span>
                )}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goNextRegion}
                disabled={regionIndex >= regions.length - 1}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>

              <div className="w-px h-5 bg-border mx-1" />

              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={togglePlayback}
                disabled={selectedWindowTime == null}
              >
                {isPlaying ? (
                  <Square className="h-3.5 w-3.5" />
                ) : (
                  <Play className="h-3.5 w-3.5" />
                )}
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={handleCancel}
                disabled={!isDirty}
              >
                <X className="h-3 w-3 mr-1" />
                Cancel
              </Button>
              <Button
                size="sm"
                className="h-7 text-xs"
                onClick={handleSave}
                disabled={!isDirty || upsertMutation.isPending}
              >
                <Save className="h-3 w-3 mr-1" />
                {upsertMutation.isPending ? "Saving…" : "Save"}
              </Button>
            </div>
          </div>

          {/* Type selector + threshold controls */}
          <div className="flex items-center gap-3 px-4 py-1.5 border-b bg-muted/30">
            <label className="text-xs text-muted-foreground">Type</label>
            <select
              className="rounded border bg-background px-2 py-1 text-xs"
              value={selectedType ?? ""}
              onChange={(e) => setSelectedType(e.target.value || null)}
            >
              <option value="">All types (max)</option>
              {vocabulary.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>

            <label className="text-xs text-muted-foreground ml-2">
              Threshold
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="rounded border bg-background px-2 py-1 text-xs w-20"
              placeholder={effectiveThreshold.toFixed(2)}
              value={thresholdOverride}
              onChange={(e) => setThresholdOverride(e.target.value)}
            />
          </div>

          {/* Spectrogram + strip */}
          {currentRegion && regionDetectionJobId ? (
            <TimelineProvider
              ref={playbackRef}
              key={selectedJobId}
              jobStart={0}
              jobEnd={timelineEnd}
              zoomLevels={REVIEW_ZOOM}
              defaultZoom={userZoom}
              playback="slice"
              audioUrlBuilder={(startEpoch, durationSec) =>
                regionAudioSliceUrl(
                  regionDetectionJobId,
                  startEpoch,
                  durationSec,
                )
              }
              disableKeyboardShortcuts
              scrollOnPlayback={false}
              onZoomChange={setUserZoom}
              onPlayStateChange={setIsPlaying}
            >
              <WindowClassifyViewerBody
                regionDetectionJobId={regionDetectionJobId}
                region={currentRegion}
                allRegions={regions}
                stripScores={stripScores}
                selectedWindowTime={selectedWindowTime}
                onSelectWindow={setSelectedWindowTime}
                allScoreRows={allScoreRows}
              />
            </TimelineProvider>
          ) : (
            <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
              No regions to display
            </div>
          )}

          {/* Detail panel */}
          <WindowDetailPanel
            selectedScoreRow={selectedScoreRow}
            selectedWindowTime={selectedWindowTime}
            labels={selectedWindowLabels}
            region={currentRegion}
            onBadgeClick={handleBadgeClick}
            onAddType={handleAddType}
            vocabulary={vocabulary}
          />
        </div>
      ) : (
        <div className="py-8 text-center text-muted-foreground">
          Select a completed window classification job to begin reviewing
          scores.
        </div>
      )}
    </div>
  );
}

// ---- Viewer body (inside TimelineProvider) ----

interface WindowClassifyViewerBodyProps {
  regionDetectionJobId: string;
  region: Region;
  allRegions: Region[];
  stripScores: (number | null)[];
  selectedWindowTime: number | null;
  onSelectWindow: (time: number | null) => void;
  allScoreRows: WindowScoreRow[];
}

function WindowClassifyViewerBody({
  regionDetectionJobId,
  region,
  allRegions,
  stripScores,
  selectedWindowTime,
  onSelectWindow,
  allScoreRows,
}: WindowClassifyViewerBodyProps) {
  const ctx = useTimelineContext();

  // Re-center when region changes
  const prevRegionRef = useRef<string>(region.region_id);
  useEffect(() => {
    if (region.region_id !== prevRegionRef.current) {
      prevRegionRef.current = region.region_id;
      const dur = region.padded_end_sec - region.padded_start_sec;
      const span = ctx.activePreset.span;
      ctx.seekTo(region.padded_start_sec + Math.min(dur, span) / 2);
    }
  }, [region.region_id, region.padded_start_sec, region.padded_end_sec, ctx]);

  // Zoom/pan keyboard shortcuts (provider shortcuts disabled)
  const {
    zoomIn: ctxZoomIn,
    zoomOut: ctxZoomOut,
    pan: ctxPan,
    centerTimestamp: ctxCenter,
    viewportSpan: ctxSpan,
  } = ctx;
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (
        el.tagName === "INPUT" ||
        el.tagName === "TEXTAREA" ||
        el.tagName === "SELECT"
      )
        return;

      switch (e.key) {
        case "+":
        case "=":
          e.preventDefault();
          ctxZoomIn();
          break;
        case "-":
          e.preventDefault();
          ctxZoomOut();
          break;
        case "ArrowLeft":
          e.preventDefault();
          ctxPan(ctxCenter - ctxSpan * 0.1);
          break;
        case "ArrowRight":
          e.preventDefault();
          ctxPan(ctxCenter + ctxSpan * 0.1);
          break;
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [ctxZoomIn, ctxZoomOut, ctxPan, ctxCenter, ctxSpan]);

  const tileUrlBuilder = useCallback(
    (_jobId: string, zoomLevel: string, tileIndex: number) =>
      regionTileUrl(regionDetectionJobId, zoomLevel, tileIndex),
    [regionDetectionJobId],
  );

  // Click handler for window selection
  const handleSpectrogramClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const relX = e.clientX - rect.left;
      const clickEpoch =
        ctx.centerTimestamp + (relX - rect.width / 2) / ctx.pxPerSec;

      // Snap to nearest window by time_sec (search all regions)
      let nearest: WindowScoreRow | null = null;
      let minDist = Infinity;
      for (const row of allScoreRows) {
        const dist = Math.abs(row.time_sec + WINDOW_SIZE_SEC / 2 - clickEpoch);
        if (dist < minDist) {
          minDist = dist;
          nearest = row;
        }
      }
      if (nearest && minDist < WINDOW_SIZE_SEC) {
        onSelectWindow(nearest.time_sec);
      }
    },
    [ctx.centerTimestamp, ctx.pxPerSec, allScoreRows, onSelectWindow],
  );

  return (
    <div className="w-full select-none">
      <div
        className="flex flex-col"
        style={{ height: 200 }}
        onClick={handleSpectrogramClick}
      >
        <Spectrogram
          jobId={regionDetectionJobId}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
          scores={stripScores.length > 0 ? stripScores : undefined}
          windowSec={HOP_SEC}
          stripHeight={STRIP_HEIGHT}
          stripGradient={HEATMAP_GRADIENT}
        >
          <AllRegionLines
            regions={allRegions}
            activeRegionId={region.region_id}
          />
          {selectedWindowTime != null && (
            <WindowBandOverlay
              timeSec={selectedWindowTime}
              windowSize={WINDOW_SIZE_SEC}
            />
          )}
        </Spectrogram>
      </div>
      <div className="border-t border-border px-2 py-1">
        <ZoomSelector />
      </div>
    </div>
  );
}

// ---- Window band overlay ----

function WindowBandOverlay({
  timeSec,
  windowSize,
}: {
  timeSec: number;
  windowSize: number;
}) {
  const { epochToX, canvasHeight } = useOverlayContext();
  const x1 = epochToX(timeSec);
  const x2 = epochToX(timeSec + windowSize);
  const width = x2 - x1;

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: x1,
        width: Math.max(2, width),
        height: canvasHeight,
        background: "rgba(168, 130, 220, 0.35)",
        borderLeft: "2px solid rgba(168, 130, 220, 0.8)",
        borderRight: "2px solid rgba(168, 130, 220, 0.8)",
        pointerEvents: "none",
        zIndex: 5,
      }}
    />
  );
}

// ---- All-region boundary lines (no shading) ----

function AllRegionLines({
  regions,
  activeRegionId,
}: {
  regions: Region[];
  activeRegionId: string;
}) {
  const { epochToX, canvasWidth, canvasHeight } = useOverlayContext();

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      {regions.map((r) => {
        const isActive = r.region_id === activeRegionId;
        const startX = epochToX(r.start_sec);
        const endX = epochToX(r.end_sec);
        const color = isActive
          ? "rgba(59, 130, 246, 0.8)"
          : "rgba(59, 130, 246, 0.35)";
        const lineWidth = isActive ? "2px" : "1px";

        return (
          <div key={r.region_id}>
            {startX >= 0 && startX <= canvasWidth && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: startX,
                  width: 0,
                  height: canvasHeight,
                  borderLeft: `${lineWidth} dashed ${color}`,
                }}
              />
            )}
            {endX >= 0 && endX <= canvasWidth && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: endX,
                  width: 0,
                  height: canvasHeight,
                  borderLeft: `${lineWidth} dashed ${color}`,
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---- Detail panel ----

interface WindowLabel {
  typeName: string;
  score: number;
  threshold: number;
  aboveThreshold: boolean;
  correction: "add" | "remove" | undefined;
  isPending: boolean;
}

function WindowDetailPanel({
  selectedScoreRow,
  selectedWindowTime,
  labels,
  region,
  onBadgeClick,
  onAddType,
  vocabulary,
}: {
  selectedScoreRow: WindowScoreRow | null;
  selectedWindowTime: number | null;
  labels: WindowLabel[];
  region: Region | null;
  onBadgeClick: (typeName: string) => void;
  onAddType: (typeName: string) => void;
  vocabulary: string[];
}) {
  const [showAddPopover, setShowAddPopover] = useState(false);

  if (selectedWindowTime == null || !selectedScoreRow) {
    return (
      <div className="px-4 py-3 text-sm text-muted-foreground border-t">
        Click on a window in the spectrogram to inspect scores
      </div>
    );
  }

  const endSec = selectedWindowTime + WINDOW_SIZE_SEC;

  // Separate above/below threshold, then sort by score descending
  const aboveLabels = labels
    .filter(
      (l) =>
        (l.aboveThreshold && l.correction !== "remove") ||
        l.correction === "add",
    )
    .sort((a, b) => b.score - a.score);
  const belowLabels = labels
    .filter(
      (l) =>
        (!l.aboveThreshold || l.correction === "remove") &&
        l.correction !== "add",
    )
    .sort((a, b) => b.score - a.score);

  // Types not currently shown as "add" that could be added
  const addableTypes = vocabulary.filter(
    (t) => !aboveLabels.some((l) => l.typeName === t),
  );

  return (
    <div className="px-4 py-3 border-t space-y-2">
      {/* Time range + region */}
      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Window: {selectedWindowTime.toFixed(1)}s – {endSec.toFixed(1)}s
          {region && (
            <span className="ml-2">
              Region: {region.region_id.slice(0, 8)}
            </span>
          )}
        </div>
      </div>

      {/* Vocalization badges */}
      <div className="flex flex-wrap items-center gap-1.5">
        {/* Above-threshold badges */}
        {aboveLabels.map((l) => (
          <Badge
            key={l.typeName}
            className="cursor-pointer text-xs text-white select-none"
            style={{
              backgroundColor: typeColor(l.typeName),
              outline: l.isPending
                ? "2px solid rgba(250, 204, 21, 0.8)"
                : undefined,
              outlineOffset: l.isPending ? "1px" : undefined,
            }}
            onClick={() => onBadgeClick(l.typeName)}
          >
            {l.typeName} {(l.score * 100).toFixed(0)}%
            {l.correction === "add" && (
              <Plus className="h-2.5 w-2.5 ml-0.5 inline" />
            )}
          </Badge>
        ))}

        {/* Below-threshold badges (dimmed) */}
        {belowLabels.map((l) => (
          <Badge
            key={l.typeName}
            variant="outline"
            className="cursor-pointer text-xs select-none opacity-50 hover:opacity-75"
            style={{
              borderColor: typeColor(l.typeName),
              color: typeColor(l.typeName),
              outline: l.isPending
                ? "2px solid rgba(250, 204, 21, 0.8)"
                : undefined,
              outlineOffset: l.isPending ? "1px" : undefined,
            }}
            onClick={() => onBadgeClick(l.typeName)}
          >
            {l.typeName} {(l.score * 100).toFixed(0)}%
            {l.correction === "remove" && (
              <X className="h-2.5 w-2.5 ml-0.5 inline" />
            )}
          </Badge>
        ))}

        {/* Plus button for adding types */}
        <div className="relative">
          <button
            className="px-1.5 py-0.5 rounded text-xs border border-dashed border-muted-foreground/40 text-muted-foreground hover:border-muted-foreground hover:text-foreground"
            onClick={() => setShowAddPopover((v) => !v)}
          >
            <Plus className="h-3 w-3" />
          </button>
          {showAddPopover && addableTypes.length > 0 && (
            <div className="absolute z-50 bottom-full mb-1 left-0 bg-popover border rounded-md shadow-md py-1 min-w-[120px]">
              {addableTypes.map((t) => (
                <button
                  key={t}
                  className="w-full px-3 py-1 text-xs text-left hover:bg-accent flex items-center gap-1.5"
                  onClick={() => {
                    onAddType(t);
                    setShowAddPopover(false);
                  }}
                >
                  <span
                    className="w-2 h-2 rounded-full inline-block"
                    style={{ backgroundColor: typeColor(t) }}
                  />
                  {t}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* All scores grid */}
      {labels.length > 0 && (
        <div className="text-xs">
          <span className="text-muted-foreground font-medium">All scores:</span>
          <div className="mt-1 grid grid-cols-3 gap-x-4 gap-y-0.5">
            {labels
              .slice()
              .sort((a, b) => b.score - a.score)
              .map((l) => (
                <div key={l.typeName} className="flex items-center gap-1">
                  <span
                    className="w-2 h-2 rounded-full inline-block"
                    style={{ backgroundColor: typeColor(l.typeName) }}
                  />
                  <span
                    className={l.aboveThreshold ? "font-medium" : ""}
                  >
                    {l.typeName}
                  </span>
                  <span className="text-muted-foreground ml-auto">
                    {l.score.toFixed(3)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
