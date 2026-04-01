import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  ArrowUp,
  ArrowDown,
  Download,
  Save,
  PackageOpen,
  AlertTriangle,
  AlertCircle,
  Info,
  X,
  Folder,
  Globe,
  Search,
  Settings,
  Activity,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  useClassifierModels,
  useHydrophones,
  useHydrophoneDetectionJobs,
  useCreateHydrophoneDetectionJob,
  useCancelHydrophoneDetectionJob,
  usePauseHydrophoneDetectionJob,
  useResumeHydrophoneDetectionJob,
  useBulkDeleteDetectionJobs,
  useDetectionContent,
  useSaveDetectionLabels,
  useSaveDetectionRowState,
  useExtractLabeledSamples,
  useBrowseDirectories,
} from "@/hooks/queries/useClassifier";
import { usePrepareTimeline } from "@/hooks/queries/useTimeline";
import { detectionTsvUrl, detectionAudioSliceUrl, detectionSpectrogramUrl } from "@/api/client";
import { BulkDeleteDialog } from "./BulkDeleteDialog";
import { ExtractDialog } from "./ExtractDialog";
import { SpectrogramPopup } from "./SpectrogramPopup";
import { showMsg } from "@/components/shared/MessageToast";
import { DateRangePickerUtc } from "@/components/shared/DateRangePickerUtc";
import type { DetectionJob, DetectionRow, DetectionLabelRow, FlashAlert } from "@/api/types";

type SortKey = "time" | "duration_sec" | "avg_confidence";
type SortDir = "asc" | "desc";
type LabelField = "humpback" | "orca" | "ship" | "background";
type PlayClip = {
  startUtc: number;
  durationSec: number;
};
type SpectrogramMarkerBounds = {
  startSec: number;
  endSec: number;
};
type SpectrogramAdjustment = "start-earlier" | "start-later" | "end-earlier" | "end-later";
type SpectrogramPopupState = {
  imageUrl: string;
  position: { x: number; y: number };
  durationSec: number;
  row: HydratedDetectionRow;
  initialMarkerBounds: SpectrogramMarkerBounds | null;
  draftManualBounds: SpectrogramMarkerBounds | null;
  editable: boolean;
};
type HydratedDetectionRow = DetectionRow & {
  _clipDurationSec: number;
  _clipRange: string;
  _rawRange: string | null;
  _playKey: string;
};

function rowUtcKey(row: DetectionRow): string {
  return `${row.start_utc}:${row.end_utc}`;
}

function isLegacyMergedMode(
  detectionMode: DetectionJob["detection_mode"],
): boolean {
  return detectionMode !== "windowed";
}

const SELECTION_WINDOW_STEP_SEC = 5;

const statusColor: Record<string, string> = {
  queued: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  paused: "bg-amber-100 text-amber-800",
  complete: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  canceled: "bg-gray-100 text-gray-800",
};

function formatCompactUtcFromEpoch(epochSec: number): string {
  const d = new Date(epochSec * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}T${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}Z`;
}


function formatUtcDateTime(timestampSeconds: number): string {
  const date = new Date(timestampSeconds * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${date.getUTCFullYear()}-${p(date.getUTCMonth() + 1)}-${p(date.getUTCDate())} ${p(date.getUTCHours())}:${p(date.getUTCMinutes())} UTC`;
}

function buildPlaybackKey(
  jobId: string,
  startUtc: number,
  durationSec: number,
): string {
  return `${jobId}:${startUtc.toFixed(3)}:${durationSec.toFixed(3)}`;
}

function resolveClipTiming(
  row: DetectionRow,
): Omit<HydratedDetectionRow, keyof DetectionRow | "_playKey"> {
  const durationSec = Math.max(0, row.end_utc - row.start_utc);
  const clipRange = `${formatCompactUtcFromEpoch(row.start_utc)}_${formatCompactUtcFromEpoch(row.end_utc)}`;

  const rawStartUtc = typeof row.raw_start_utc === "number" ? row.raw_start_utc : row.start_utc;
  const rawEndUtc = typeof row.raw_end_utc === "number" ? row.raw_end_utc : row.end_utc;
  const rawRange =
    rawStartUtc !== row.start_utc || rawEndUtc !== row.end_utc
      ? `${formatCompactUtcFromEpoch(rawStartUtc)}_${formatCompactUtcFromEpoch(rawEndUtc)}`
      : null;

  return {
    _clipDurationSec: durationSec,
    _clipRange: clipRange,
    _rawRange: rawRange,
  };
}

function resolveMarkerBoundsFromAbsoluteUtc(
  row: HydratedDetectionRow,
  startUtc: number | null | undefined,
  endUtc: number | null | undefined,
): SpectrogramMarkerBounds | null {
  if (typeof startUtc !== "number" || typeof endUtc !== "number" || endUtc <= startUtc) {
    return null;
  }

  const epsilon = 1e-6;
  const relativeStartSec = startUtc - row.start_utc;
  const relativeEndSec = endUtc - row.start_utc;

  if (
    relativeStartSec < -epsilon ||
    relativeEndSec > row._clipDurationSec + epsilon ||
    relativeEndSec <= relativeStartSec
  ) {
    return null;
  }

  return {
    startSec: Math.max(0, relativeStartSec),
    endSec: Math.min(row._clipDurationSec, relativeEndSec),
  };
}

function resolvePositiveSelectionMarkerBounds(
  row: HydratedDetectionRow,
  options?: { includeAutoSelection?: boolean },
): SpectrogramMarkerBounds | null {
  const explicitSelection = resolveMarkerBoundsFromAbsoluteUtc(
    row,
    row.positive_selection_start_utc,
    row.positive_selection_end_utc,
  );
  if (explicitSelection) {
    return explicitSelection;
  }

  if (options?.includeAutoSelection) {
    const autoSelection = resolveMarkerBoundsFromAbsoluteUtc(
      row,
      row.auto_positive_selection_start_utc,
      row.auto_positive_selection_end_utc,
    );
    if (autoSelection) {
      return autoSelection;
    }
  }

  if (row._clipDurationSec <= 0) {
    return null;
  }

  return {
    startSec: 0,
    endSec: row._clipDurationSec,
  };
}

function resolveManualSelectionMarkerBounds(
  row: HydratedDetectionRow,
): SpectrogramMarkerBounds | null {
  return resolveMarkerBoundsFromAbsoluteUtc(
    row,
    row.manual_positive_selection_start_utc,
    row.manual_positive_selection_end_utc,
  );
}

function adjustSelectionBounds(
  bounds: SpectrogramMarkerBounds,
  adjustment: SpectrogramAdjustment,
  clipDurationSec: number,
): SpectrogramMarkerBounds | null {
  const epsilon = 1e-6;
  const minDurationSec = SELECTION_WINDOW_STEP_SEC;
  const next = { ...bounds };
  const currentDurationSec = bounds.endSec - bounds.startSec;

  if (adjustment === "start-earlier") {
    if (currentDurationSec >= clipDurationSec - epsilon) {
      return null;
    }
    if (currentDurationSec + SELECTION_WINDOW_STEP_SEC > clipDurationSec + epsilon) {
      return {
        startSec: 0,
        endSec: clipDurationSec,
      };
    }

    next.startSec = Math.max(0, bounds.startSec - SELECTION_WINDOW_STEP_SEC);
    const remainingGrowthSec =
      currentDurationSec + SELECTION_WINDOW_STEP_SEC - (bounds.endSec - next.startSec);
    if (remainingGrowthSec > epsilon) {
      next.endSec = Math.min(clipDurationSec, bounds.endSec + remainingGrowthSec);
    }
  } else if (adjustment === "start-later") {
    if (bounds.endSec - (bounds.startSec + SELECTION_WINDOW_STEP_SEC) < minDurationSec - epsilon) {
      return null;
    }
    next.startSec = bounds.startSec + SELECTION_WINDOW_STEP_SEC;
  } else if (adjustment === "end-earlier") {
    if ((bounds.endSec - SELECTION_WINDOW_STEP_SEC) - bounds.startSec < minDurationSec - epsilon) {
      return null;
    }
    next.endSec = bounds.endSec - SELECTION_WINDOW_STEP_SEC;
  } else {
    if (currentDurationSec >= clipDurationSec - epsilon) {
      return null;
    }
    if (currentDurationSec + SELECTION_WINDOW_STEP_SEC > clipDurationSec + epsilon) {
      return {
        startSec: 0,
        endSec: clipDurationSec,
      };
    }

    next.endSec = Math.min(clipDurationSec, bounds.endSec + SELECTION_WINDOW_STEP_SEC);
    const remainingGrowthSec =
      currentDurationSec + SELECTION_WINDOW_STEP_SEC - (next.endSec - bounds.startSec);
    if (remainingGrowthSec > epsilon) {
      next.startSec = Math.max(0, bounds.startSec - remainingGrowthSec);
    }
  }

  if (next.endSec <= next.startSec || next.endSec - next.startSec < minDurationSec - epsilon) {
    return null;
  }
  if (next.startSec < -epsilon || next.endSec > clipDurationSec + epsilon) {
    return null;
  }

  return {
    startSec: Math.max(0, next.startSec),
    endSec: Math.min(clipDurationSec, next.endSec),
  };
}

function formatLocalDateTime(isoString: string): string {
  const d = new Date(isoString.endsWith("Z") ? isoString : isoString + "Z");
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function formatUtcDateRange(startTs: number, endTs: number): string {
  return `${formatUtcDateTime(startTs)} — ${formatUtcDateTime(endTs)}`;
}

function formatDurationHM(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  return `${hours}h ${String(minutes).padStart(2, "0")}m`;
}

export function HydrophoneTab() {
  const { data: models = [] } = useClassifierModels();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: jobs = [] } = useHydrophoneDetectionJobs(3000);
  const createMutation = useCreateHydrophoneDetectionJob();
  const cancelMutation = useCancelHydrophoneDetectionJob();
  const pauseMutation = usePauseHydrophoneDetectionJob();
  const resumeMutation = useResumeHydrophoneDetectionJob();
  const bulkDeleteMutation = useBulkDeleteDetectionJobs();
  const saveLabelsMutation = useSaveDetectionLabels();
  const extractMutation = useExtractLabeledSamples();

  // Buffered label edits: jobId -> rowKey -> { humpback, orca, ship, background }
  const [labelEdits, setLabelEdits] = useState<
    Map<string, Map<string, Partial<Record<LabelField, number | null>>>>
  >(new Map());
  const [dirtyJobs, setDirtyJobs] = useState<Set<string>>(new Set());
  const [showExtractDialog, setShowExtractDialog] = useState(false);

  // Form state
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedHydrophoneId, setSelectedHydrophoneId] = useState("");
  const [startEpoch, setStartEpoch] = useState<number | null>(null);
  const [endEpoch, setEndEpoch] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(0.90);
  const [hopSeconds, setHopSeconds] = useState(1.0);
  const [highThreshold, setHighThreshold] = useState(0.80);
  const [lowThreshold, setLowThreshold] = useState(0.70);
  const [sourceType, setSourceType] = useState<"orcasound" | "noaa" | "local">("orcasound");
  const [localCachePath, setLocalCachePath] = useState("");
  const [browseRoot, setBrowseRoot] = useState<string | null>(null);
  const { data: browseData } = useBrowseDirectories(browseRoot);

  // Filtered hydrophones by source type
  const filteredHydrophones = useMemo(() => {
    if (sourceType === "local") return hydrophones;
    return hydrophones.filter((h) =>
      sourceType === "orcasound"
        ? h.provider_kind === "orcasound_hls"
        : h.provider_kind === "noaa_gcs",
    );
  }, [hydrophones, sourceType]);

  // Reset hydrophone selection when it falls outside the filtered list
  useEffect(() => {
    if (selectedHydrophoneId && !filteredHydrophones.some((h) => h.id === selectedHydrophoneId)) {
      setSelectedHydrophoneId("");
    }
  }, [filteredHydrophones, selectedHydrophoneId]);

  // Table state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);

  // Audio
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

  const activeJobs = jobs.filter((j) => j.status === "running" || j.status === "queued" || j.status === "paused");
  const previousJobs = jobs.filter((j) => j.status !== "running" && j.status !== "queued" && j.status !== "paused");

  // Previous Jobs preferences
  type PrevJobsColumnId = "status" | "created" | "hydrophone" | "date" | "threshold" | "results" | "download" | "extract" | "timeline" | "error";
  const ALL_PREV_COLUMNS: { id: PrevJobsColumnId; label: string }[] = [
    { id: "status", label: "Status" },
    { id: "created", label: "Created" },
    { id: "hydrophone", label: "Hydrophone" },
    { id: "date", label: "Date Range (UTC)" },
    { id: "threshold", label: "Threshold" },
    { id: "results", label: "Results" },
    { id: "download", label: "Download" },
    { id: "extract", label: "Extract" },
    { id: "timeline", label: "Timeline" },
    { id: "error", label: "Error" },
  ];
  const [prevJobsPageSize, setPrevJobsPageSize] = useState(20);
  const [prevJobsVisibleCols, setPrevJobsVisibleCols] = useState<Set<PrevJobsColumnId>>(
    () => new Set(ALL_PREV_COLUMNS.map((c) => c.id)),
  );
  const [showPrefsDialog, setShowPrefsDialog] = useState(false);
  // Staging state for the dialog (apply on Confirm)
  const [prefsPageSize, setPrefsPageSize] = useState(20);
  const [prefsVisibleCols, setPrefsVisibleCols] = useState<Set<PrevJobsColumnId>>(
    () => new Set(ALL_PREV_COLUMNS.map((c) => c.id)),
  );

  // Previous Jobs pagination, filter, and sort
  const [prevJobsPage, setPrevJobsPage] = useState(1);
  const [prevJobsFilterText, setPrevJobsFilterText] = useState("");

  type PrevJobsSortKey = "status" | "created" | "hydrophone" | "date" | "threshold" | "results";
  const [prevJobsSortKey, setPrevJobsSortKey] = useState<PrevJobsSortKey>("created");
  const [prevJobsSortDir, setPrevJobsSortDir] = useState<SortDir>("desc");

  const togglePrevJobsSort = useCallback(
    (key: PrevJobsSortKey) => {
      if (prevJobsSortKey === key) {
        setPrevJobsSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setPrevJobsSortKey(key);
        setPrevJobsSortDir("asc");
      }
    },
    [prevJobsSortKey],
  );

  const filteredPreviousJobs = useMemo(() => {
    if (!prevJobsFilterText) return previousJobs;
    const q = prevJobsFilterText.toLowerCase();
    return previousJobs.filter((j) => (j.hydrophone_name ?? "").toLowerCase().includes(q));
  }, [previousJobs, prevJobsFilterText]);

  const sortedPreviousJobs = useMemo(() => {
    const sorted = [...filteredPreviousJobs];
    const dir = prevJobsSortDir === "asc" ? 1 : -1;
    sorted.sort((a, b) => {
      switch (prevJobsSortKey) {
        case "status":
          return dir * (a.status ?? "").localeCompare(b.status ?? "");
        case "created":
          return dir * (new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
        case "hydrophone":
          return dir * (a.hydrophone_name ?? "").localeCompare(b.hydrophone_name ?? "");
        case "date":
          return dir * ((a.start_timestamp ?? 0) - (b.start_timestamp ?? 0));
        case "threshold":
          return dir * (a.high_threshold - b.high_threshold);
        case "results": {
          const aCount = (a.result_summary as Record<string, number> | null)?.n_spans ?? 0;
          const bCount = (b.result_summary as Record<string, number> | null)?.n_spans ?? 0;
          return dir * (aCount - bCount);
        }
        default:
          return 0;
      }
    });
    return sorted;
  }, [filteredPreviousJobs, prevJobsSortKey, prevJobsSortDir]);

  const totalPrevPages = Math.max(1, Math.ceil(sortedPreviousJobs.length / prevJobsPageSize));
  const effectivePrevPage = Math.min(prevJobsPage, totalPrevPages);

  const paginatedPreviousJobs = useMemo(() => {
    const start = (effectivePrevPage - 1) * prevJobsPageSize;
    return sortedPreviousJobs.slice(start, start + prevJobsPageSize);
  }, [sortedPreviousJobs, effectivePrevPage]);

  // Reset page when filter or sort changes
  useEffect(() => {
    setPrevJobsPage(1);
  }, [prevJobsFilterText, prevJobsSortKey, prevJobsSortDir]);

  const expandedJob = useMemo(
    () => [...activeJobs, ...previousJobs].find((j) => j.id === expandedJobId) ?? null,
    [activeJobs, previousJobs, expandedJobId],
  );
  const expandedContentJobId = useMemo(() => {
    if (!expandedJob) return null;
    if (expandedJob.status === "complete" || expandedJob.status === "canceled") return expandedJob.id;
    if (expandedJob.status === "paused" && expandedJob.output_tsv_path) return expandedJob.id;
    if (expandedJob.status === "running" && (expandedJob.segments_processed ?? 0) > 0) return expandedJob.id;
    return null;
  }, [expandedJob]);
  const { data: expandedRows = [] } = useDetectionContent(expandedContentJobId);
  const expandedJobIsLegacyMerged = useMemo(
    () => (expandedJob ? isLegacyMergedMode(expandedJob.detection_mode) : false),
    [expandedJob],
  );
  const expandedHasSavedLabels = useMemo(
    () => expandedRows.some((r) => r.humpback === 1 || r.orca === 1 || r.ship === 1 || r.background === 1),
    [expandedRows],
  );
  const extractTargetIds = useMemo(() => {
    if (!expandedJob || !expandedHasSavedLabels || expandedJobIsLegacyMerged) {
      return new Set<string>();
    }
    if (expandedJob.status === "paused" || expandedJob.status === "complete" || expandedJob.status === "canceled") {
      return new Set<string>([expandedJob.id]);
    }
    return new Set<string>();
  }, [expandedJob, expandedHasSavedLabels, expandedJobIsLegacyMerged]);

  const handleSubmit = () => {
    if (!selectedModelId || !startEpoch || !endEpoch) return;
    if (sourceType !== "local" && !selectedHydrophoneId) return;
    if (sourceType === "local" && !localCachePath) return;
    createMutation.mutate({
      classifier_model_id: selectedModelId,
      hydrophone_id: selectedHydrophoneId,
      start_timestamp: startEpoch,
      end_timestamp: endEpoch,
      confidence_threshold: threshold,
      hop_seconds: hopSeconds,
      high_threshold: highThreshold,
      low_threshold: lowThreshold,
      ...(sourceType === "local" ? { local_cache_path: localCachePath } : {}),
    });
  };

  // Selection helpers
  const toggleId = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handlePlay = useCallback(
    (jobId: string, row: DetectionRow, clip?: PlayClip) => {
      const startUtc = clip?.startUtc ?? row.start_utc;
      const requestedDurationSec = clip?.durationSec ?? row.end_utc - row.start_utc;
      const durationSec = Math.max(0.1, requestedDurationSec);
      const key = buildPlaybackKey(jobId, startUtc, durationSec);
      if (playingKey === key) {
        audioRef.current?.pause();
        setPlayingKey(null);
        return;
      }
      const url = detectionAudioSliceUrl(jobId, startUtc, durationSec);
      if (audioRef.current) {
        const audio = audioRef.current;
        audio.src = url;
        audio.load();
        setPlayingKey(key);
        audio.play().catch(() => setPlayingKey(null));
      }
    },
    [playingKey],
  );

  // Buffered label editing (save via Save Labels button)
  const handleLabelChange = useCallback(
    (jobId: string, rk: string, field: LabelField, value: number | null) => {
      setLabelEdits((prev) => {
        const next = new Map(prev);
        const jobEdits = new Map(next.get(jobId) ?? new Map());
        const rowEdits = { ...(jobEdits.get(rk) ?? {}) };
        rowEdits[field] = value;
        jobEdits.set(rk, rowEdits);
        next.set(jobId, jobEdits);
        return next;
      });
      setDirtyJobs((prev) => new Set(prev).add(jobId));
    },
    [],
  );

  const handleSaveLabels = useCallback(async () => {
    const promises: Promise<unknown>[] = [];
    for (const jobId of dirtyJobs) {
      const jobEdits = labelEdits.get(jobId);
      if (!jobEdits || jobEdits.size === 0) continue;
      const rows: DetectionLabelRow[] = [];
      for (const [rk, edits] of jobEdits) {
        const [startStr, endStr] = rk.split(":");
        rows.push({
          start_utc: parseFloat(startStr),
          end_utc: parseFloat(endStr),
          humpback: edits.humpback ?? null,
          orca: edits.orca ?? null,
          ship: edits.ship ?? null,
          background: edits.background ?? null,
        });
      }
      promises.push(saveLabelsMutation.mutateAsync({ jobId, rows }));
    }
    await Promise.all(promises);
    setLabelEdits(new Map());
    setDirtyJobs(new Set());
  }, [dirtyJobs, labelEdits, saveLabelsMutation]);

  const clearRowEdits = useCallback((jobId: string, rk: string) => {
    setLabelEdits((prev) => {
      const next = new Map(prev);
      const jobEdits = next.get(jobId);
      if (!jobEdits) {
        return prev;
      }
      const nextJobEdits = new Map(jobEdits);
      nextJobEdits.delete(rk);
      if (nextJobEdits.size === 0) {
        next.delete(jobId);
      } else {
        next.set(jobId, nextJobEdits);
      }
      return next;
    });
    setDirtyJobs((prev) => {
      const next = new Set(prev);
      const jobEdits = labelEdits.get(jobId);
      if (!jobEdits || (jobEdits.size <= 1 && jobEdits.has(rk))) {
        next.delete(jobId);
      }
      return next;
    });
  }, [labelEdits]);

  return (
    <div className="space-y-4">
      <audio ref={audioRef} onEnded={() => setPlayingKey(null)} />

      {/* Job Creation Form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Hydrophone Detection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Audio Source */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Audio Source</label>
            <div className="flex gap-2">
              <Button
                type="button"
                variant={sourceType === "orcasound" ? "default" : "outline"}
                size="sm"
                onClick={() => setSourceType("orcasound")}
              >
                <Globe className="h-3.5 w-3.5 mr-1.5" />
                Orcasound
              </Button>
              <Button
                type="button"
                variant={sourceType === "noaa" ? "default" : "outline"}
                size="sm"
                onClick={() => setSourceType("noaa")}
              >
                <Globe className="h-3.5 w-3.5 mr-1.5" />
                NOAA
              </Button>
              <Button
                type="button"
                variant={sourceType === "local" ? "default" : "outline"}
                size="sm"
                onClick={() => setSourceType("local")}
              >
                <Folder className="h-3.5 w-3.5 mr-1.5" />
                Local Cache
              </Button>
            </div>
          </div>

          <div className={sourceType !== "local" ? "grid grid-cols-2 gap-3" : ""}>
            {sourceType !== "local" && (
              <div>
                <label className="text-sm font-medium">Hydrophone</label>
                <select
                  className="w-full border rounded px-3 py-2 text-sm mt-1"
                  value={selectedHydrophoneId}
                  onChange={(e) => setSelectedHydrophoneId(e.target.value)}
                >
                  <option value="">Select a hydrophone…</option>
                  {filteredHydrophones.map((h) => (
                    <option key={h.id} value={h.id}>
                      {h.name} — {h.location}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <div>
              <label className="text-sm font-medium">Classifier Model</label>
              <select
                className="w-full border rounded px-3 py-2 text-sm mt-1"
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
              >
                <option value="">Select a model…</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.model_version})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {sourceType === "local" && (
            <div className="space-y-1.5">
              <div className="flex gap-2">
                <Input
                  placeholder="Local folder path…"
                  value={localCachePath}
                  onChange={(e) => {
                    setLocalCachePath(e.target.value);
                    setBrowseRoot(null);
                  }}
                  className="flex-1"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setBrowseRoot(localCachePath || "/")}
                >
                  Browse
                </Button>
              </div>
              {browseData && browseRoot !== null && (
                <div className="border rounded p-2 max-h-40 overflow-y-auto text-xs space-y-0.5">
                  {browseRoot !== "/" && (
                    <button
                      className="block w-full text-left px-2 py-1 hover:bg-muted rounded text-muted-foreground"
                      onClick={() => {
                        const parent = browseRoot.replace(/\/[^/]+\/?$/, "") || "/";
                        setBrowseRoot(parent);
                      }}
                    >
                      ../ (up)
                    </button>
                  )}
                  {browseData.subdirectories.map((d) => (
                    <button
                      key={d.path}
                      className="block w-full text-left px-2 py-1 hover:bg-muted rounded"
                      onClick={() => setBrowseRoot(d.path)}
                      onDoubleClick={() => {
                        setLocalCachePath(d.path);
                        setBrowseRoot(null);
                      }}
                    >
                      {d.name}/
                    </button>
                  ))}
                  {browseData.subdirectories.length === 0 && (
                    <p className="text-muted-foreground px-2 py-1">No subdirectories</p>
                  )}
                  <div className="pt-1 border-t mt-1">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="w-full text-xs"
                      onClick={() => {
                        setLocalCachePath(browseData.path);
                        setBrowseRoot(null);
                      }}
                    >
                      Select: {browseData.path}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}

          <div>
            <label className="text-sm font-medium">Date Range (UTC)</label>
            <DateRangePickerUtc
              value={{ startEpoch, endEpoch }}
              onChange={({ startEpoch: s, endEpoch: e }) => {
                setStartEpoch(s);
                setEndEpoch(e);
              }}
              placeholder="Select date range (UTC)"
            />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-sm font-medium">
                Confidence Threshold: {threshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full mt-1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">
                Start Threshold: {highThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={highThreshold}
                onChange={(e) => setHighThreshold(parseFloat(e.target.value))}
                className="w-full mt-1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">
                Continue Threshold: {lowThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={lowThreshold}
                onChange={(e) => setLowThreshold(parseFloat(e.target.value))}
                className="w-full mt-1"
              />
            </div>
          </div>
          <div className="w-1/3">
            <label className="text-sm font-medium">Hop Size (s)</label>
            <Input
              type="number"
              min={0.1}
              max={10}
              step={0.1}
              value={hopSeconds}
              onChange={(e) => setHopSeconds(parseFloat(e.target.value) || 1.0)}
              className="mt-1"
            />
          </div>
          <p className="text-xs text-muted-foreground">
            New jobs use fixed 5-second windowed detections.
          </p>
          <Button
            onClick={handleSubmit}
            disabled={
              !selectedModelId ||
              (sourceType !== "local" && !selectedHydrophoneId) ||
              !startEpoch ||
              !endEpoch ||
              (sourceType === "local" && !localCachePath) ||
              createMutation.isPending
            }
          >
            {createMutation.isPending ? "Creating…" : "Start Detection"}
          </Button>
          {createMutation.isError && (
            <p className="text-sm text-red-600">
              {(createMutation.error as Error).message}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Active Jobs Panel */}
      {activeJobs.length > 0 && (
        <div className="border rounded-md">
          <div className="flex items-center justify-between px-4 py-3 border-b">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">Active Jobs</h3>
              <Badge variant="secondary">{activeJobs.length}</Badge>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={
                  dirtyJobs.size === 0 ||
                  saveLabelsMutation.isPending ||
                  (expandedJobId != null &&
                    jobs.find((j) => j.id === expandedJobId)?.status === "running")
                }
                onClick={handleSaveLabels}
              >
                <Save className="h-3.5 w-3.5 mr-1" />
                {saveLabelsMutation.isPending ? "Saving…" : "Save Labels"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={extractTargetIds.size === 0}
                onClick={() => setShowExtractDialog(true)}
              >
                <PackageOpen className="h-3.5 w-3.5 mr-1" />
                Extract Labeled Samples
              </Button>
            </div>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="w-8 px-1 py-2" />
                <th className="px-3 py-2 text-left font-medium">Status</th>
                <th className="px-3 py-2 text-left font-medium">Created</th>
                <th className="px-3 py-2 text-left font-medium">Hydrophone</th>
                <th className="px-3 py-2 text-left font-medium">Date Range (UTC)</th>
                <th className="px-3 py-2 text-left font-medium">Threshold</th>
                <th className="px-3 py-2 text-left font-medium">Progress</th>
                <th className="px-3 py-2 text-left font-medium">Actions</th>
                <th className="px-3 py-2 text-left font-medium">Download</th>
              </tr>
            </thead>
            <tbody>
              {activeJobs.map((job) => (
                <HydrophoneJobRow
                  key={job.id}
                  job={job}
                  isActive
                  expanded={expandedJobId === job.id}
                  onExpand={() =>
                    setExpandedJobId(expandedJobId === job.id ? null : job.id)
                  }
                  playingKey={playingKey}
                  onPlay={handlePlay}
                  onLabelChange={handleLabelChange}
                  onClearRowEdit={clearRowEdits}
                  labelEdits={labelEdits.get(job.id) ?? null}
                  onPause={(id) => pauseMutation.mutate(id)}
                  onResume={(id) => resumeMutation.mutate(id)}
                  onCancel={(id) => cancelMutation.mutate(id)}
                  pausePending={pauseMutation.isPending}
                  resumePending={resumeMutation.isPending}
                  cancelPending={cancelMutation.isPending}
                />
              ))}
            </tbody>
          </table>
          {activeJobs.some((j) => j.alerts && j.alerts.length > 0) && (
            <div className="px-4 py-2 border-t">
              {activeJobs.map((j) =>
                j.alerts && j.alerts.length > 0 ? (
                  <AlertsPanel key={j.id} alerts={j.alerts} />
                ) : null,
              )}
            </div>
          )}
        </div>
      )}

      {/* Previous Jobs */}
      {previousJobs.length > 0 && (
        <div className="border rounded-md">
          {/* Title + actions row */}
          <div className="flex items-center justify-between px-4 py-3 border-b">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">Previous Jobs</h3>
              <Badge variant="secondary">{filteredPreviousJobs.length}</Badge>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={
                  dirtyJobs.size === 0 ||
                  saveLabelsMutation.isPending ||
                  (expandedJobId != null &&
                    jobs.find((j) => j.id === expandedJobId)?.status === "running")
                }
                onClick={handleSaveLabels}
              >
                <Save className="h-3.5 w-3.5 mr-1" />
                {saveLabelsMutation.isPending ? "Saving…" : "Save Labels"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={extractTargetIds.size === 0}
                onClick={() => setShowExtractDialog(true)}
              >
                <PackageOpen className="h-3.5 w-3.5 mr-1" />
                Extract Labeled Samples
              </Button>
              <Button
                variant="destructive"
                size="sm"
                disabled={selectedIds.size === 0}
                onClick={() => setShowDeleteDialog(true)}
              >
                Delete ({selectedIds.size})
              </Button>
            </div>
          </div>
          {/* Filter + pagination + preferences bar */}
          <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/30">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Filter by hydrophone…"
                value={prevJobsFilterText}
                onChange={(e) => setPrevJobsFilterText(e.target.value)}
                className="h-8 w-64 pl-8 text-xs"
                autoComplete="off"
                data-lpignore="true"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs">
                {sortedPreviousJobs.length > 0
                  ? `${(effectivePrevPage - 1) * prevJobsPageSize + 1}–${Math.min(effectivePrevPage * prevJobsPageSize, sortedPreviousJobs.length)} of ${sortedPreviousJobs.length}`
                  : "0 items"}
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2"
                disabled={effectivePrevPage <= 1}
                onClick={() => setPrevJobsPage((p) => Math.max(1, p - 1))}
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2"
                disabled={effectivePrevPage >= totalPrevPages}
                onClick={() => setPrevJobsPage((p) => Math.min(totalPrevPages, p + 1))}
              >
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => {
                  setPrefsPageSize(prevJobsPageSize);
                  setPrefsVisibleCols(new Set(prevJobsVisibleCols));
                  setShowPrefsDialog(true);
                }}
              >
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="w-10 px-3 py-2">
                  <Checkbox
                    checked={
                      paginatedPreviousJobs.length > 0 &&
                      paginatedPreviousJobs.every((j) => selectedIds.has(j.id))
                        ? true
                        : paginatedPreviousJobs.some((j) => selectedIds.has(j.id))
                          ? "indeterminate"
                          : false
                    }
                    onCheckedChange={() => {
                      const allSel = paginatedPreviousJobs.every((j) => selectedIds.has(j.id));
                      if (allSel) {
                        setSelectedIds((prev) => {
                          const next = new Set(prev);
                          paginatedPreviousJobs.forEach((j) => next.delete(j.id));
                          return next;
                        });
                      } else {
                        setSelectedIds((prev) => {
                          const next = new Set(prev);
                          paginatedPreviousJobs.forEach((j) => next.add(j.id));
                          return next;
                        });
                      }
                    }}
                  />
                </th>
                <th className="w-8 px-1 py-2" />
                {prevJobsVisibleCols.has("status") && (
                  <th
                    className="px-3 py-2 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
                    onClick={() => togglePrevJobsSort("status")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Status
                      {prevJobsSortKey === "status" &&
                        (prevJobsSortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : (
                          <ArrowDown className="h-3 w-3" />
                        ))}
                    </span>
                  </th>
                )}
                {prevJobsVisibleCols.has("created") && (
                  <th
                    className="px-3 py-2 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
                    onClick={() => togglePrevJobsSort("created")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Created
                      {prevJobsSortKey === "created" &&
                        (prevJobsSortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : (
                          <ArrowDown className="h-3 w-3" />
                        ))}
                    </span>
                  </th>
                )}
                {prevJobsVisibleCols.has("hydrophone") && (
                  <th
                    className="px-3 py-2 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
                    onClick={() => togglePrevJobsSort("hydrophone")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Hydrophone
                      {prevJobsSortKey === "hydrophone" &&
                        (prevJobsSortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : (
                          <ArrowDown className="h-3 w-3" />
                        ))}
                    </span>
                  </th>
                )}
                {prevJobsVisibleCols.has("date") && (
                  <th
                    className="px-3 py-2 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
                    onClick={() => togglePrevJobsSort("date")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Date Range (UTC)
                      {prevJobsSortKey === "date" &&
                        (prevJobsSortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : (
                          <ArrowDown className="h-3 w-3" />
                        ))}
                    </span>
                  </th>
                )}
                {prevJobsVisibleCols.has("threshold") && (
                  <th
                    className="px-3 py-2 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
                    onClick={() => togglePrevJobsSort("threshold")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Threshold
                      {prevJobsSortKey === "threshold" &&
                        (prevJobsSortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : (
                          <ArrowDown className="h-3 w-3" />
                        ))}
                    </span>
                  </th>
                )}
                {prevJobsVisibleCols.has("results") && (
                  <th
                    className="px-3 py-2 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
                    onClick={() => togglePrevJobsSort("results")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Results
                      {prevJobsSortKey === "results" &&
                        (prevJobsSortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : (
                          <ArrowDown className="h-3 w-3" />
                        ))}
                    </span>
                  </th>
                )}
                {prevJobsVisibleCols.has("download") && (
                  <th className="px-3 py-2 text-left font-medium">Download</th>
                )}
                {prevJobsVisibleCols.has("extract") && (
                  <th className="px-3 py-2 text-left font-medium">Extract</th>
                )}
                {prevJobsVisibleCols.has("timeline") && (
                  <th className="px-3 py-2 text-left font-medium">Timeline</th>
                )}
                {prevJobsVisibleCols.has("error") && (
                  <th className="px-3 py-2 text-left font-medium">Error</th>
                )}
              </tr>
            </thead>
            <tbody>
              {paginatedPreviousJobs.map((job) => (
                <HydrophoneJobRow
                  key={job.id}
                  job={job}
                  checked={selectedIds.has(job.id)}
                  onToggle={() => toggleId(job.id)}
                  expanded={expandedJobId === job.id}
                  onExpand={() =>
                    setExpandedJobId(expandedJobId === job.id ? null : job.id)
                  }
                  playingKey={playingKey}
                  onPlay={handlePlay}
                  onLabelChange={handleLabelChange}
                  onClearRowEdit={clearRowEdits}
                  labelEdits={labelEdits.get(job.id) ?? null}
                  visibleColumns={prevJobsVisibleCols}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <BulkDeleteDialog
        open={showDeleteDialog}
        onOpenChange={setShowDeleteDialog}
        count={selectedIds.size}
        entityName="hydrophone detection job"
        onConfirm={() => {
          bulkDeleteMutation.mutate([...selectedIds], {
            onSuccess: () => {
              setSelectedIds(new Set());
              setShowDeleteDialog(false);
            },
          });
        }}
        isPending={bulkDeleteMutation.isPending}
      />

      <ExtractDialog
        open={showExtractDialog}
        onOpenChange={setShowExtractDialog}
        selectedIds={extractTargetIds}
        extractMutation={extractMutation}
        onSuccess={() => undefined}
      />

      {/* Preferences dialog */}
      <Dialog open={showPrefsDialog} onOpenChange={setShowPrefsDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Preferences</DialogTitle>
          </DialogHeader>
          <div className="space-y-5 py-2">
            {/* Page size */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Page size</label>
              <div className="flex gap-3">
                {[10, 20, 50, 100].map((size) => (
                  <label key={size} className="flex items-center gap-1.5 text-sm cursor-pointer">
                    <input
                      type="radio"
                      name="pageSize"
                      checked={prefsPageSize === size}
                      onChange={() => setPrefsPageSize(size)}
                      className="accent-primary"
                    />
                    {size}
                  </label>
                ))}
              </div>
            </div>
            {/* Visible columns */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Visible columns</label>
              <div className="grid grid-cols-2 gap-2">
                {ALL_PREV_COLUMNS.map((col) => (
                  <label key={col.id} className="flex items-center gap-2 text-sm cursor-pointer">
                    <Checkbox
                      checked={prefsVisibleCols.has(col.id)}
                      onCheckedChange={(checked) => {
                        setPrefsVisibleCols((prev) => {
                          const next = new Set(prev);
                          if (checked) next.add(col.id);
                          else next.delete(col.id);
                          return next;
                        });
                      }}
                    />
                    {col.label}
                  </label>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setShowPrefsDialog(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() => {
                setPrevJobsPageSize(prefsPageSize);
                setPrevJobsVisibleCols(new Set(prefsVisibleCols));
                setPrevJobsPage(1);
                setShowPrefsDialog(false);
              }}
            >
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function AlertsPanel({ alerts }: { alerts: FlashAlert[] }) {
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());

  const visible = alerts.filter((_, i) => !dismissed.has(i));
  if (visible.length === 0) return null;

  const dismiss = (idx: number) => {
    setDismissed((prev) => new Set(prev).add(idx));
  };

  const alertStyles: Record<string, string> = {
    error: "bg-red-50 border-red-200 text-red-800 dark:bg-red-950/30 dark:border-red-800 dark:text-red-300",
    warning: "bg-amber-50 border-amber-200 text-amber-800 dark:bg-amber-950/30 dark:border-amber-800 dark:text-amber-300",
    info: "bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-950/30 dark:border-blue-800 dark:text-blue-300",
  };

  const alertIcons: Record<string, typeof AlertTriangle> = {
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
  };

  return (
    <div className="space-y-1.5 max-h-40 overflow-y-auto">
      {alerts.map((alert, i) => {
        if (dismissed.has(i)) return null;
        const Icon = alertIcons[alert.type] ?? Info;
        return (
          <div
            key={i}
            className={`flex items-start gap-2 px-3 py-2 text-xs border rounded ${alertStyles[alert.type] ?? alertStyles.info}`}
          >
            <Icon className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
            <span className="flex-1">{alert.message}</span>
            <button
              className="p-0.5 hover:bg-black/10 rounded flex-shrink-0"
              onClick={() => dismiss(i)}
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

function HydrophoneJobRow({
  job,
  isActive,
  checked,
  onToggle,
  expanded,
  onExpand,
  playingKey,
  onPlay,
  onLabelChange,
  onClearRowEdit,
  labelEdits,
  onPause,
  onResume,
  onCancel,
  pausePending,
  resumePending,
  cancelPending,
  visibleColumns,
}: {
  job: DetectionJob;
  isActive?: boolean;
  checked?: boolean;
  onToggle?: () => void;
  expanded: boolean;
  onExpand: () => void;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow, clip?: PlayClip) => void;
  onLabelChange: (jobId: string, rk: string, field: LabelField, value: number | null) => void;
  onClearRowEdit: (jobId: string, rk: string) => void;
  labelEdits: Map<string, Partial<Record<LabelField, number | null>>> | null;
  onPause?: (id: string) => void;
  onResume?: (id: string) => void;
  onCancel?: (id: string) => void;
  pausePending?: boolean;
  resumePending?: boolean;
  cancelPending?: boolean;
  visibleColumns?: Set<string>;
}) {
  const navigate = useNavigate();
  const prepareTimeline = usePrepareTimeline();
  const summary = job.result_summary as Record<string, unknown> | null;
  const isRunning = job.status === "running";
  const isLegacyMerged = isLegacyMergedMode(job.detection_mode);
  const canExpand =
    job.status === "complete" || job.status === "canceled" ||
    job.status === "paused" ||
    (isRunning && (job.segments_processed ?? 0) > 0);

  // For active rows, show all columns; for previous rows, respect visibleColumns
  const showCol = (id: string) => isActive || !visibleColumns || visibleColumns.has(id);
  const colSpan = isActive ? 9 : 2 + (visibleColumns?.size ?? 9);

  return (
    <>
      <tr className="border-b hover:bg-muted/30">
        {!isActive && (
          <td className="px-3 py-2">
            <Checkbox checked={checked} onCheckedChange={onToggle} />
          </td>
        )}
        <td className="px-1 py-2">
          {canExpand && (
            <button className="p-0.5 hover:bg-muted rounded" onClick={onExpand}>
              {expanded ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
          )}
        </td>
        {showCol("status") && (
          <td className="px-3 py-2">
            <Badge className={statusColor[job.status] ?? ""}>{job.status}</Badge>
            {isLegacyMerged && (
              <Badge variant="outline" className="ml-1.5 border-amber-300 text-amber-800">
                Legacy merged
              </Badge>
            )}
            {job.has_positive_labels && (
              <Badge variant="outline" className="ml-1.5 text-[10px] py-0">
                Whale
              </Badge>
            )}
          </td>
        )}
        {showCol("created") && (
          <td className="px-3 py-2 text-muted-foreground text-xs">
            {job.created_at
              ? formatLocalDateTime(job.created_at)
              : "\u2014"}
          </td>
        )}
        {showCol("hydrophone") && (
          <td className="px-3 py-2 text-muted-foreground">
            {job.hydrophone_name}
            {job.local_cache_path && (
              <Badge variant="outline" className="ml-1.5 text-[10px] py-0">local</Badge>
            )}
          </td>
        )}
        {showCol("date") && (
          <td className="px-3 py-2 text-muted-foreground text-xs">
            {job.start_timestamp != null && job.end_timestamp != null
              ? formatUtcDateRange(job.start_timestamp, job.end_timestamp)
              : "\u2014"}
          </td>
        )}
        {showCol("threshold") && (
          <td className="px-3 py-2 text-muted-foreground">
            {job.high_threshold}/{job.low_threshold}
          </td>
        )}
        {isActive ? (
          <>
            <td className="px-3 py-2 text-muted-foreground">
              {job.status === "queued" ? (
                "\u2014"
              ) : (
                <>
                  {job.segments_processed ?? 0}/{job.segments_total ?? "?"}
                  {job.time_covered_sec != null && (
                    <span className="text-xs ml-1">
                      ({formatDurationHM(job.time_covered_sec)} processed audio)
                    </span>
                  )}
                </>
              )}
            </td>
            <td className="px-3 py-2">
              <div className="flex gap-1">
                {job.status === "running" && onPause && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    disabled={pausePending}
                    onClick={() => onPause(job.id)}
                  >
                    <Pause className="h-3 w-3 mr-1" />
                    Pause
                  </Button>
                )}
                {job.status === "paused" && onResume && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    disabled={resumePending}
                    onClick={() => onResume(job.id)}
                  >
                    <Play className="h-3 w-3 mr-1" />
                    Resume
                  </Button>
                )}
                {onCancel && (
                  <Button
                    variant="destructive"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    disabled={cancelPending}
                    onClick={() => onCancel(job.id)}
                  >
                    <X className="h-3 w-3 mr-1" />
                    Cancel
                  </Button>
                )}
              </div>
            </td>
            <td className="px-3 py-2">
              {(job.status === "paused" || job.status === "running") && job.output_tsv_path && (
                <a
                  href={detectionTsvUrl(job.id)}
                  download
                  className="text-blue-600 hover:underline text-xs inline-flex items-center gap-1"
                >
                  <Download className="h-3 w-3" />
                  TSV
                </a>
              )}
            </td>
          </>
        ) : (
          <>
            {showCol("results") && (
              <td className="px-3 py-2 text-muted-foreground">
                {summary
                  ? `${summary.n_spans} span(s)`
                  : "\u2014"}
                {job.time_covered_sec != null && (
                  <span className="text-xs ml-1">
                    ({formatDurationHM(job.time_covered_sec)} processed audio)
                  </span>
                )}
              </td>
            )}
            {showCol("download") && (
              <td className="px-3 py-2">
                {(job.status === "complete" || job.status === "canceled") && job.output_tsv_path && (
                  <a
                    href={detectionTsvUrl(job.id)}
                    download
                    className="text-blue-600 hover:underline text-xs inline-flex items-center gap-1"
                  >
                    <Download className="h-3 w-3" />
                    TSV
                  </a>
                )}
              </td>
            )}
            {showCol("extract") && (
              <td className="px-3 py-2">
                {job.extract_status ? (
                  <Badge className={statusColor[job.extract_status] ?? ""}>
                    {job.extract_status}
                  </Badge>
                ) : (
                  <span className="text-muted-foreground">&mdash;</span>
                )}
              </td>
            )}
            {showCol("timeline") && (
              <td className="px-3 py-2">
                {job.status === "complete" && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 px-2 text-xs"
                    disabled={prepareTimeline.isPending}
                    onClick={async () => {
                      try {
                        const start = job.start_timestamp ?? 0;
                        const end = job.end_timestamp ?? start;
                        await prepareTimeline.mutateAsync({
                          jobId: job.id,
                          request: {
                            scope: "startup",
                            zoomLevel: "1h",
                            centerTimestamp: start + (end - start) / 2,
                          },
                        });
                        navigate(`/app/classifier/timeline/${job.id}`, {
                          state: { prepareRequested: true },
                        });
                      } catch {
                        showMsg("warning", "Could not start timeline cache prep; opening Timeline view anyway.");
                        navigate(`/app/classifier/timeline/${job.id}`);
                      }
                    }}
                  >
                    <Activity className="h-3 w-3 mr-1" />
                    Timeline
                  </Button>
                )}
              </td>
            )}
            {showCol("error") && (
              <td className="px-3 py-2">
                {job.error_message && (
                  <span className="text-red-600 text-xs truncate block max-w-48">
                    {job.error_message}
                  </span>
                )}
              </td>
            )}
          </>
        )}
      </tr>
      {expanded && canExpand && (
        <tr>
          <td colSpan={colSpan} className="p-0">
            <HydrophoneContentTable
              jobId={job.id}
              isRunning={isRunning}
              detectionMode={job.detection_mode}
              playingKey={playingKey}
              onPlay={onPlay}
              onLabelChange={onLabelChange}
              onClearRowEdit={onClearRowEdit}
              labelEdits={labelEdits}
            />
          </td>
        </tr>
      )}
    </>
  );
}

function HydrophoneContentTable({
  jobId,
  isRunning,
  detectionMode,
  playingKey,
  onPlay,
  onLabelChange,
  onClearRowEdit,
  labelEdits,
}: {
  jobId: string;
  isRunning: boolean;
  detectionMode: "merged" | "windowed" | null;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow, clip?: PlayClip) => void;
  onLabelChange: (jobId: string, rk: string, field: LabelField, value: number | null) => void;
  onClearRowEdit: (jobId: string, rk: string) => void;
  labelEdits: Map<string, Partial<Record<LabelField, number | null>>> | null;
}) {
  const navigate = useNavigate();
  const { data: rows = [], isLoading } = useDetectionContent(
    jobId,
    isRunning ? 3000 : undefined,
  );
  const isLegacyMerged = isLegacyMergedMode(detectionMode);
  const saveRowStateMutation = useSaveDetectionRowState();
  const [sortKey, setSortKey] = useState<SortKey>(isRunning ? "time" : "avg_confidence");
  const [sortDir, setSortDir] = useState<SortDir>(isRunning ? "asc" : "desc");
  const prevRunning = useRef(isRunning);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const [spectrogramPopup, setSpectrogramPopup] = useState<SpectrogramPopupState | null>(null);
  const tableRef = useRef<HTMLDivElement>(null);

  const hydratedRows = useMemo<HydratedDetectionRow[]>(
    () =>
      rows.map((row) => {
        const resolved = resolveClipTiming(row);
        return {
          ...row,
          ...resolved,
          _playKey: buildPlaybackKey(
            jobId,
            row.start_utc,
            Math.max(0.1, resolved._clipDurationSec),
          ),
        };
      }),
    [rows, jobId],
  );

  // Switch to confidence desc when job completes
  useEffect(() => {
    if (prevRunning.current && !isRunning) {
      setSortKey("avg_confidence" as SortKey);
      setSortDir("desc");
    }
    prevRunning.current = isRunning;
  }, [isRunning]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sorted = useMemo(() => {
    return [...hydratedRows].sort((a, b) => {
      if (sortKey === "duration_sec") {
        return sortDir === "asc"
          ? a._clipDurationSec - b._clipDurationSec
          : b._clipDurationSec - a._clipDurationSec;
      }
      if (sortKey === "time") {
        return sortDir === "asc"
          ? a.start_utc - b.start_utc
          : b.start_utc - a.start_utc;
      }
      const av = a.avg_confidence ?? 0;
      const bv = b.avg_confidence ?? 0;
      return sortDir === "asc" ? av - bv : bv - av;
    });
  }, [hydratedRows, sortKey, sortDir]);

  const getEffectiveLabel = useCallback(
    (row: DetectionRow, field: LabelField): number | null => {
      const rk = rowUtcKey(row);
      const edit = labelEdits?.get(rk);
      if (edit && field in edit) {
        return edit[field] ?? null;
      }
      return row[field];
    },
    [labelEdits],
  );

  const isPositiveRow = useCallback(
    (row: DetectionRow) =>
      getEffectiveLabel(row, "humpback") === 1 || getEffectiveLabel(row, "orca") === 1,
    [getEffectiveLabel],
  );

  const openSpectrogramPopup = useCallback(
    (row: HydratedDetectionRow, position: { x: number; y: number }) => {
      const initialMarkerBounds = isPositiveRow(row)
        ? resolvePositiveSelectionMarkerBounds(row, { includeAutoSelection: true })
        : null;
      setSpectrogramPopup({
        imageUrl: detectionSpectrogramUrl(
          jobId,
          row.start_utc,
          Math.max(0.1, row._clipDurationSec),
        ),
        position,
        durationSec: Math.max(0.1, row._clipDurationSec),
        row,
        initialMarkerBounds,
        draftManualBounds: resolveManualSelectionMarkerBounds(row),
        editable: false,
      });
    },
    [isPositiveRow, jobId],
  );

  const popupMarkerBounds =
    spectrogramPopup?.draftManualBounds ?? spectrogramPopup?.initialMarkerBounds ?? null;

  const adjustPopupBounds = useCallback((adjustment: SpectrogramAdjustment) => {
    setSpectrogramPopup((prev) => {
      if (!prev) {
        return prev;
      }
      const currentBounds = prev.draftManualBounds ?? prev.initialMarkerBounds;
      if (!currentBounds) {
        return prev;
      }
      const nextBounds = adjustSelectionBounds(
        currentBounds,
        adjustment,
        prev.row._clipDurationSec,
      );
      if (!nextBounds) {
        return prev;
      }
      return {
        ...prev,
        draftManualBounds: nextBounds,
      };
    });
  }, []);

  const handleApplySpectrogramEdit = useCallback(async () => {
    if (!spectrogramPopup?.editable) {
      return;
    }

    const { row, draftManualBounds } = spectrogramPopup;
    try {
      await saveRowStateMutation.mutateAsync({
        jobId,
        body: {
          start_utc: row.start_utc,
          end_utc: row.end_utc,
          humpback: getEffectiveLabel(row, "humpback"),
          orca: getEffectiveLabel(row, "orca"),
          ship: getEffectiveLabel(row, "ship"),
          background: getEffectiveLabel(row, "background"),
          manual_positive_selection_start_utc:
            draftManualBounds !== null
              ? row.start_utc + draftManualBounds.startSec
              : null,
          manual_positive_selection_end_utc:
            draftManualBounds !== null
              ? row.start_utc + draftManualBounds.endSec
              : null,
        },
      });
      onClearRowEdit(jobId, rowUtcKey(row));
      setSpectrogramPopup(null);
    } catch (error) {
      showMsg(
        "error",
        error instanceof Error ? error.message : "Failed to update detection row state",
      );
    }
  }, [
    getEffectiveLabel,
    jobId,
    onClearRowEdit,
    saveRowStateMutation,
    spectrogramPopup,
  ]);

  const spectrogramEditor =
    spectrogramPopup?.editable && popupMarkerBounds
      ? {
          selectionDurationSec: popupMarkerBounds.endSec - popupMarkerBounds.startSec,
          canMoveStartEarlier: adjustSelectionBounds(
            popupMarkerBounds,
            "start-earlier",
            spectrogramPopup.row._clipDurationSec,
          ) !== null,
          canMoveStartLater: adjustSelectionBounds(
            popupMarkerBounds,
            "start-later",
            spectrogramPopup.row._clipDurationSec,
          ) !== null,
          canMoveEndEarlier: adjustSelectionBounds(
            popupMarkerBounds,
            "end-earlier",
            spectrogramPopup.row._clipDurationSec,
          ) !== null,
          canMoveEndLater: adjustSelectionBounds(
            popupMarkerBounds,
            "end-later",
            spectrogramPopup.row._clipDurationSec,
          ) !== null,
          isApplying: saveRowStateMutation.isPending,
          onMoveStartEarlier: () => adjustPopupBounds("start-earlier"),
          onMoveStartLater: () => adjustPopupBounds("start-later"),
          onMoveEndEarlier: () => adjustPopupBounds("end-earlier"),
          onMoveEndLater: () => adjustPopupBounds("end-later"),
          onApply: () => {
            void handleApplySpectrogramEdit();
          },
          onCancel: () => setSpectrogramPopup(null),
        }
      : null;

  const handleCheckboxClick = useCallback(
    (row: DetectionRow, field: LabelField) => {
      if (isLegacyMerged) {
        return;
      }
      const current = getEffectiveLabel(row, field);
      const next = current === 1 ? 0 : 1;
      onLabelChange(jobId, rowUtcKey(row), field, next);
    },
    [isLegacyMerged, jobId, getEffectiveLabel, onLabelChange],
  );

  // Keyboard shortcuts
  const keyMap: Record<string, LabelField> = { h: "humpback", o: "orca", s: "ship", b: "background" };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

      // Any key closes the spectrogram popup
      const hadPopup = !!spectrogramPopup;
      if (hadPopup) {
        setSpectrogramPopup(null);
      }

      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        setFocusedIndex((prev) => {
          if (prev === null) return 0;
          return Math.min(prev + 1, sorted.length - 1);
        });
        return;
      }
      if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        setFocusedIndex((prev) => {
          if (prev === null) return 0;
          return Math.max(prev - 1, 0);
        });
        return;
      }

      if (e.key === " " && focusedIndex !== null && focusedIndex < sorted.length) {
        e.preventDefault();
        const focused = sorted[focusedIndex];
        onPlay(jobId, focused, {
          startUtc: focused.start_utc,
          durationSec: focused._clipDurationSec,
        });
        // Open spectrogram popup on the left side of the viewport (only on play, not stop)
        if (!hadPopup) {
          const cx = window.innerWidth * 0.15;
          const cy = window.innerHeight / 2;
          openSpectrogramPopup(focused, { x: cx, y: cy });
        }
        return;
      }
      const field = keyMap[e.key.toLowerCase()];
      if (field && focusedIndex !== null && focusedIndex < sorted.length) {
        e.preventDefault();
        handleCheckboxClick(sorted[focusedIndex], field);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [sorted, focusedIndex, handleCheckboxClick, onPlay, jobId, spectrogramPopup, openSpectrogramPopup]);

  useEffect(() => {
    if (focusedIndex === null || !tableRef.current) return;
    const row = tableRef.current.querySelector(`tbody tr:nth-child(${focusedIndex + 1})`);
    row?.scrollIntoView({ block: "nearest" });
  }, [focusedIndex]);

  if (isLoading) {
    return <div className="p-4 text-sm text-muted-foreground">Loading detections…</div>;
  }
  if (hydratedRows.length === 0) {
    return <div className="p-4 text-sm text-muted-foreground">No detections</div>;
  }

  const SortHeader = ({ label, field }: { label: string; field: SortKey }) => (
    <th
      className="px-3 py-1.5 text-left font-medium cursor-pointer hover:bg-muted/50 select-none"
      onClick={() => handleSort(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === field &&
          (sortDir === "asc" ? (
            <ArrowUp className="h-3 w-3" />
          ) : (
            <ArrowDown className="h-3 w-3" />
          ))}
      </span>
    </th>
  );

  return (
    <div className="bg-muted/20 border-t" ref={tableRef}>
      {isLegacyMerged && (
        <div className="flex items-start gap-2 border-b bg-amber-50 px-3 py-2 text-[11px] text-amber-900">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            Legacy merged-mode job. Labels, extraction, and marker edits are disabled
            here; rerun this job in windowed mode to continue.
          </span>
        </div>
      )}
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b">
            <th className="w-8 px-3 py-1.5" />
            <SortHeader label="Detection Range" field="time" />
            <SortHeader label="Duration (s)" field="duration_sec" />
            <SortHeader label="Confidence" field="avg_confidence" />
            <th className="px-3 py-1.5 text-center font-medium">Humpback</th>
            <th className="px-3 py-1.5 text-center font-medium">Orca</th>
            <th className="px-3 py-1.5 text-center font-medium">Ship</th>
            <th className="px-3 py-1.5 text-center font-medium">Background</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const isPlaying = playingKey === row._playKey;
            const isFocused = focusedIndex === i;
            return (
              <tr
                key={i}
                className={`border-b last:border-0 cursor-pointer ${
                  isFocused ? "bg-blue-100 dark:bg-blue-900/30" : "hover:bg-muted/30"
                }`}
                onClick={(e) => {
                  if (e.altKey) {
                    e.preventDefault();
                    openSpectrogramPopup(row, { x: e.clientX, y: e.clientY });
                    return;
                  }
                  setFocusedIndex(i);
                }}
              >
                <td className="px-3 py-1.5">
                  <button
                    className="p-0.5 hover:bg-muted rounded"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (!isPlaying) {
                        const rect = e.currentTarget.getBoundingClientRect();
                        openSpectrogramPopup(row, {
                          x: rect.right,
                          y: rect.top + rect.height / 2,
                        });
                      }
                      onPlay(jobId, row, {
                        startUtc: row.start_utc,
                        durationSec: row._clipDurationSec,
                      });
                    }}
                    title={isPlaying ? "Pause" : "Play"}
                  >
                    {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate("/app/search", {
                        state: {
                          source: "detection",
                          detectionJobId: jobId,
                          startUtc: row.start_utc,
                          endUtc: row.end_utc,
                          clipDuration: row._clipDurationSec,
                        },
                      });
                    }}
                    className="p-0.5 rounded hover:bg-blue-100"
                    title="Search similar"
                  >
                    <Search className="h-3.5 w-3.5 text-blue-600" />
                  </button>
                </td>
                <td
                  className="px-3 py-1.5 max-w-80"
                  title={
                    row._rawRange
                      ? `${row._clipRange}\nRaw: ${row._rawRange}`
                      : row._clipRange
                  }
                >
                  <div className="font-mono clip-range">{row._clipRange}</div>
                </td>
                <td className="px-3 py-1.5 clip-duration">{row._clipDurationSec.toFixed(1)}</td>
                <td className="px-3 py-1.5">{row.avg_confidence != null ? row.avg_confidence.toFixed(3) : "—"}</td>
                <td className="px-3 py-1.5 text-center">
                  <input
                    type="checkbox"
                    className={`h-3.5 w-3.5 ${isLegacyMerged ? "cursor-not-allowed" : "cursor-pointer"}`}
                    checked={getEffectiveLabel(row, "humpback") === 1}
                    disabled={isLegacyMerged}
                    onChange={() => handleCheckboxClick(row, "humpback")}
                  />
                </td>
                <td className="px-3 py-1.5 text-center">
                  <input
                    type="checkbox"
                    className={`h-3.5 w-3.5 ${isLegacyMerged ? "cursor-not-allowed" : "cursor-pointer"}`}
                    checked={getEffectiveLabel(row, "orca") === 1}
                    disabled={isLegacyMerged}
                    onChange={() => handleCheckboxClick(row, "orca")}
                  />
                </td>
                <td className="px-3 py-1.5 text-center">
                  <input
                    type="checkbox"
                    className={`h-3.5 w-3.5 ${isLegacyMerged ? "cursor-not-allowed" : "cursor-pointer"}`}
                    checked={getEffectiveLabel(row, "ship") === 1}
                    disabled={isLegacyMerged}
                    onChange={() => handleCheckboxClick(row, "ship")}
                  />
                </td>
                <td className="px-3 py-1.5 text-center">
                  <input
                    type="checkbox"
                    className={`h-3.5 w-3.5 ${isLegacyMerged ? "cursor-not-allowed" : "cursor-pointer"}`}
                    checked={getEffectiveLabel(row, "background") === 1}
                    disabled={isLegacyMerged}
                    onChange={() => handleCheckboxClick(row, "background")}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {spectrogramPopup && (
        <SpectrogramPopup
          imageUrl={spectrogramPopup.imageUrl}
          position={spectrogramPopup.position}
          markerBounds={detectionMode === "windowed" ? null : popupMarkerBounds}
          durationSec={spectrogramPopup.durationSec}
          editor={spectrogramEditor}
          onClose={() => setSpectrogramPopup(null)}
        />
      )}
    </div>
  );
}
