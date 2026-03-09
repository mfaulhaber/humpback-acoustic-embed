import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ChevronDown,
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
} from "lucide-react";
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
  useExtractLabeledSamples,
  useBrowseDirectories,
} from "@/hooks/queries/useClassifier";
import { detectionTsvUrl, detectionAudioSliceUrl } from "@/api/client";
import { BulkDeleteDialog } from "./BulkDeleteDialog";
import { ExtractDialog } from "./ExtractDialog";
import { DateRangePickerUtc } from "@/components/shared/DateRangePickerUtc";
import type { DetectionJob, DetectionRow, DetectionLabelRow, FlashAlert } from "@/api/types";

type SortKey = "filename" | "duration_sec" | "avg_confidence";
type SortDir = "asc" | "desc";
type LabelField = "humpback" | "ship" | "background";
type ClipSource = "extract_filename" | "snapped" | "raw";
type PlayClip = {
  startSec: number;
  durationSec: number;
};
type HydratedDetectionRow = DetectionRow & {
  _extractFilename: string | null;
  _clipSource: ClipSource;
  _clipStartSec: number;
  _clipEndSec: number;
  _clipDurationSec: number;
  _clipRange: string;
  _rawRange: string;
  _playKey: string;
};

function rowKey(row: { filename: string; start_sec: number; end_sec: number }): string {
  return `${row.filename}:${row.start_sec}:${row.end_sec}`;
}

const statusColor: Record<string, string> = {
  queued: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  paused: "bg-amber-100 text-amber-800",
  complete: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  canceled: "bg-gray-100 text-gray-800",
};

function formatCompactUtc(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}T${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}Z`;
}


function formatUtcDateTime(timestampSeconds: number): string {
  const date = new Date(timestampSeconds * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${date.getUTCFullYear()}-${p(date.getUTCMonth() + 1)}-${p(date.getUTCDate())} ${p(date.getUTCHours())}:${p(date.getUTCMinutes())} UTC`;
}

function computeUtcRange(filename: string, startSec: number, endSec: number): string {
  const basename = filename.replace(".wav", "");
  const match = basename.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  if (!match) return filename;
  const chunkMs = Date.UTC(+match[1], +match[2] - 1, +match[3], +match[4], +match[5], +match[6]);
  return `${formatCompactUtc(chunkMs + startSec * 1000)}_${formatCompactUtc(chunkMs + endSec * 1000)}`;
}

function parseCompactUtcMs(value: string): number | null {
  const match = value.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  if (!match) return null;
  return Date.UTC(
    Number(match[1]),
    Number(match[2]) - 1,
    Number(match[3]),
    Number(match[4]),
    Number(match[5]),
    Number(match[6]),
  );
}

function computeSnappedExtractFilename(
  filename: string,
  startSec: number,
  endSec: number,
  windowSizeSeconds: number | null,
): string | null {
  const baseTs = parseCompactUtcMs(filename.replace(".wav", ""));
  if (baseTs === null) return null;
  if (windowSizeSeconds == null || windowSizeSeconds <= 0) return null;

  const snappedStartSec = Math.floor(startSec / windowSizeSeconds) * windowSizeSeconds;
  const snappedEndSec = Math.ceil(endSec / windowSizeSeconds) * windowSizeSeconds;
  return `${formatCompactUtc(baseTs + snappedStartSec * 1000)}_${formatCompactUtc(baseTs + snappedEndSec * 1000)}.wav`;
}

function parseExtractFilenameRange(extractFilename: string): { startMs: number; endMs: number } | null {
  const base = extractFilename.replace(".wav", "");
  const parts = base.split("_");
  if (parts.length !== 2) return null;
  const startMs = parseCompactUtcMs(parts[0]);
  const endMs = parseCompactUtcMs(parts[1]);
  if (startMs === null || endMs === null || endMs <= startMs) return null;
  return { startMs, endMs };
}

function buildPlaybackKey(
  jobId: string,
  filename: string,
  startSec: number,
  durationSec: number,
): string {
  return `${jobId}:${filename}:${startSec.toFixed(3)}:${durationSec.toFixed(3)}`;
}

function resolveClipTiming(
  row: DetectionRow,
  windowSizeSeconds: number | null,
): Omit<HydratedDetectionRow, keyof DetectionRow | "_playKey"> {
  const rawStartSec = row.start_sec;
  const rawEndSec = row.end_sec;
  const rawDurationSec = Math.max(0, rawEndSec - rawStartSec);
  const rawRange = computeUtcRange(row.filename, rawStartSec, rawEndSec);

  const directExtractFilename =
    typeof row.extract_filename === "string" && row.extract_filename.trim()
      ? row.extract_filename.trim()
      : null;
  const snappedExtractFilename = computeSnappedExtractFilename(
    row.filename,
    row.start_sec,
    row.end_sec,
    windowSizeSeconds,
  );
  const extractFilename = directExtractFilename ?? snappedExtractFilename ?? null;

  const baseTs = parseCompactUtcMs(row.filename.replace(".wav", ""));
  if (baseTs !== null && directExtractFilename) {
    const extracted = parseExtractFilenameRange(directExtractFilename);
    if (extracted) {
      const clipStartSec = (extracted.startMs - baseTs) / 1000;
      const clipEndSec = (extracted.endMs - baseTs) / 1000;
      if (clipEndSec > clipStartSec) {
        return {
          _extractFilename: extractFilename,
          _clipSource: "extract_filename",
          _clipStartSec: clipStartSec,
          _clipEndSec: clipEndSec,
          _clipDurationSec: clipEndSec - clipStartSec,
          _clipRange: `${formatCompactUtc(extracted.startMs)}_${formatCompactUtc(extracted.endMs)}`,
          _rawRange: rawRange,
        };
      }
    }
  }

  if (windowSizeSeconds != null && windowSizeSeconds > 0) {
    const snappedStartSec = Math.floor(rawStartSec / windowSizeSeconds) * windowSizeSeconds;
    const snappedEndSec = Math.ceil(rawEndSec / windowSizeSeconds) * windowSizeSeconds;
    if (snappedEndSec > snappedStartSec) {
      const clipRange =
        baseTs !== null
          ? `${formatCompactUtc(baseTs + snappedStartSec * 1000)}_${formatCompactUtc(baseTs + snappedEndSec * 1000)}`
          : rawRange;
      return {
        _extractFilename: extractFilename,
        _clipSource: "snapped",
        _clipStartSec: snappedStartSec,
        _clipEndSec: snappedEndSec,
        _clipDurationSec: snappedEndSec - snappedStartSec,
        _clipRange: clipRange,
        _rawRange: rawRange,
      };
    }
  }

  return {
    _extractFilename: extractFilename,
    _clipSource: "raw",
    _clipStartSec: rawStartSec,
    _clipEndSec: rawEndSec,
    _clipDurationSec: rawDurationSec,
    _clipRange: rawRange,
    _rawRange: rawRange,
  };
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

  // Buffered label edits: jobId -> rowKey -> { humpback, ship, background }
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
  const [threshold, setThreshold] = useState(0.5);
  const [hopSeconds, setHopSeconds] = useState(1.0);
  const [highThreshold, setHighThreshold] = useState(0.70);
  const [lowThreshold, setLowThreshold] = useState(0.45);
  const [sourceType, setSourceType] = useState<"s3" | "local">("s3");
  const [localCachePath, setLocalCachePath] = useState("");
  const [browseRoot, setBrowseRoot] = useState<string | null>(null);
  const { data: browseData } = useBrowseDirectories(browseRoot);

  // Table state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);

  // Audio
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const modelWindowSizeById = useMemo(
    () => new Map(models.map((model) => [model.id, model.window_size_seconds])),
    [models],
  );

  const activeJob = jobs.find((j) => j.status === "running" || j.status === "queued" || j.status === "paused");
  const previousJobs = jobs.filter((j) => j.status !== "running" && j.status !== "queued" && j.status !== "paused");
  const expandedJob = useMemo(
    () => previousJobs.find((j) => j.id === expandedJobId) ?? null,
    [previousJobs, expandedJobId],
  );
  const expandedCompletedJobId =
    expandedJob && (expandedJob.status === "complete" || expandedJob.status === "canceled") ? expandedJob.id : null;
  const { data: expandedRows = [] } = useDetectionContent(expandedCompletedJobId);
  const expandedHasSavedLabels = useMemo(
    () => expandedRows.some((r) => r.humpback === 1 || r.ship === 1 || r.background === 1),
    [expandedRows],
  );
  const extractTargetIds = useMemo(() => {
    if (!expandedCompletedJobId || !expandedHasSavedLabels) return new Set<string>();
    return new Set<string>([expandedCompletedJobId]);
  }, [expandedCompletedJobId, expandedHasSavedLabels]);

  const handleSubmit = () => {
    if (!selectedModelId || !selectedHydrophoneId || !startEpoch || !endEpoch) return;
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
      const startSec = clip?.startSec ?? row.start_sec;
      const requestedDurationSec = clip?.durationSec ?? row.end_sec - row.start_sec;
      const durationSec = Math.max(0.1, requestedDurationSec);
      const key = buildPlaybackKey(jobId, row.filename, startSec, durationSec);
      if (playingKey === key) {
        audioRef.current?.pause();
        setPlayingKey(null);
        return;
      }
      const url = detectionAudioSliceUrl(jobId, row.filename, startSec, durationSec);
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
        const [filename, startStr, endStr] = rk.split(":");
        rows.push({
          filename,
          start_sec: parseFloat(startStr),
          end_sec: parseFloat(endStr),
          humpback: edits.humpback ?? null,
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

  return (
    <div className="space-y-4">
      <audio ref={audioRef} onEnded={() => setPlayingKey(null)} />

      {/* Job Creation Form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Hydrophone Detection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-sm font-medium">Hydrophone</label>
              <select
                className="w-full border rounded px-3 py-2 text-sm mt-1"
                value={selectedHydrophoneId}
                onChange={(e) => setSelectedHydrophoneId(e.target.value)}
              >
                <option value="">Select a hydrophone…</option>
                {hydrophones.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.name} — {h.location}
                  </option>
                ))}
              </select>
            </div>
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

          {/* Audio Source */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Audio Source</label>
            <div className="flex gap-2">
              <Button
                type="button"
                variant={sourceType === "s3" ? "default" : "outline"}
                size="sm"
                onClick={() => setSourceType("s3")}
              >
                <Globe className="h-3.5 w-3.5 mr-1.5" />
                S3 (Orcasound)
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
            {sourceType === "local" && (
              <div className="space-y-1.5">
                <div className="flex gap-2">
                  <Input
                    placeholder="Path to local HLS cache folder…"
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
                <p className="text-xs text-muted-foreground">
                  Expects S3-mirrored structure: <code>{"{path}"}/audio-orcasound-net/{"{hydrophone}"}/hls/…</code>
                </p>
              </div>
            )}
          </div>

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
          <div>
            <label className="text-sm font-medium">
              Confidence Threshold (summary): {threshold.toFixed(2)}
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
          <div className="grid grid-cols-3 gap-3">
            <div>
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
          <Button
            onClick={handleSubmit}
            disabled={
              !selectedModelId ||
              !selectedHydrophoneId ||
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

      {/* Active Job Panel */}
      {activeJob && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">
                Active Job — {activeJob.hydrophone_name}
                {activeJob.local_cache_path && (
                  <Badge variant="outline" className="ml-2 text-[10px] py-0 align-middle">local</Badge>
                )}
              </CardTitle>
              <div className="flex gap-2">
                {activeJob.status === "running" && (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={pauseMutation.isPending}
                    onClick={() => pauseMutation.mutate(activeJob.id)}
                  >
                    <Pause className="h-3.5 w-3.5 mr-1" />
                    Pause
                  </Button>
                )}
                {activeJob.status === "paused" && (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={resumeMutation.isPending}
                    onClick={() => resumeMutation.mutate(activeJob.id)}
                  >
                    <Play className="h-3.5 w-3.5 mr-1" />
                    Resume
                  </Button>
                )}
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={
                    (activeJob.status !== "running" && activeJob.status !== "paused") ||
                    cancelMutation.isPending
                  }
                  onClick={() => cancelMutation.mutate(activeJob.id)}
                >
                  <X className="h-3.5 w-3.5 mr-1" />
                  Cancel
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {activeJob.start_timestamp != null && activeJob.end_timestamp != null && (
              <p className="text-sm text-muted-foreground">
                {formatUtcDateRange(activeJob.start_timestamp, activeJob.end_timestamp)}
              </p>
            )}
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>
                  Processed {activeJob.segments_processed ?? 0}/
                  {activeJob.segments_total ?? "?"} segments
                  {activeJob.time_covered_sec != null && (
                    <span className="text-muted-foreground">
                      {" "}({formatDurationHM(activeJob.time_covered_sec)} audio)
                    </span>
                  )}
                </span>
                <Badge className={statusColor[activeJob.status] ?? ""}>
                  {activeJob.status}
                </Badge>
              </div>
              {activeJob.segments_total != null && activeJob.segments_total > 0 && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{
                      width: `${Math.min(
                        100,
                        ((activeJob.segments_processed ?? 0) / activeJob.segments_total) * 100,
                      )}%`,
                    }}
                  />
                </div>
              )}
            </div>

            {/* Flash Alerts */}
            {activeJob.alerts && activeJob.alerts.length > 0 && (
              <AlertsPanel alerts={activeJob.alerts} />
            )}

            {/* Live detection content */}
            {(activeJob.segments_processed ?? 0) > 0 && activeJob.output_tsv_path && (
              <HydrophoneContentTable
                jobId={activeJob.id}
                isRunning={true}
                windowSizeSeconds={modelWindowSizeById.get(activeJob.classifier_model_id) ?? null}
                playingKey={playingKey}
                onPlay={handlePlay}
                onLabelChange={handleLabelChange}
                labelEdits={labelEdits.get(activeJob.id) ?? null}
              />
            )}
          </CardContent>
        </Card>
      )}

      {/* Previous Jobs */}
      {previousJobs.length > 0 && (
        <div className="border rounded-md">
          <div className="flex items-center justify-between px-4 py-3 border-b">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">Previous Jobs</h3>
              <Badge variant="secondary">{previousJobs.length}</Badge>
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
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="w-10 px-3 py-2">
                  <Checkbox
                    checked={
                      previousJobs.length > 0 &&
                      previousJobs.every((j) => selectedIds.has(j.id))
                        ? true
                        : previousJobs.some((j) => selectedIds.has(j.id))
                          ? "indeterminate"
                          : false
                    }
                    onCheckedChange={() => {
                      const allSel = previousJobs.every((j) => selectedIds.has(j.id));
                      if (allSel) setSelectedIds(new Set());
                      else setSelectedIds(new Set(previousJobs.map((j) => j.id)));
                    }}
                  />
                </th>
                <th className="w-8 px-1 py-2" />
                <th className="px-3 py-2 text-left font-medium">Status</th>
                <th className="px-3 py-2 text-left font-medium">Hydrophone</th>
                <th className="px-3 py-2 text-left font-medium">Date Range (UTC)</th>
                <th className="px-3 py-2 text-left font-medium">Threshold</th>
                <th className="px-3 py-2 text-left font-medium">Results</th>
                <th className="px-3 py-2 text-left font-medium">Download</th>
                <th className="px-3 py-2 text-left font-medium">Extract</th>
                <th className="px-3 py-2 text-left font-medium">Error</th>
              </tr>
            </thead>
            <tbody>
              {previousJobs.map((job) => (
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
                  labelEdits={labelEdits.get(job.id) ?? null}
                  windowSizeSeconds={modelWindowSizeById.get(job.classifier_model_id) ?? null}
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
  checked,
  onToggle,
  expanded,
  onExpand,
  playingKey,
  onPlay,
  onLabelChange,
  labelEdits,
  windowSizeSeconds,
}: {
  job: DetectionJob;
  checked: boolean;
  onToggle: () => void;
  expanded: boolean;
  onExpand: () => void;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow, clip?: PlayClip) => void;
  onLabelChange: (jobId: string, rk: string, field: LabelField, value: number | null) => void;
  labelEdits: Map<string, Partial<Record<LabelField, number | null>>> | null;
  windowSizeSeconds: number | null;
}) {
  const summary = job.result_summary as Record<string, unknown> | null;
  const isRunning = job.status === "running";
  const canExpand =
    (job.status === "complete" || job.status === "canceled" ||
     (isRunning && (job.segments_processed ?? 0) > 0)) &&
    !!job.output_tsv_path;

  return (
    <>
      <tr className="border-b hover:bg-muted/30">
        <td className="px-3 py-2">
          <Checkbox checked={checked} onCheckedChange={onToggle} />
        </td>
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
        <td className="px-3 py-2">
          <Badge className={statusColor[job.status] ?? ""}>{job.status}</Badge>
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {job.hydrophone_name}
          {job.local_cache_path && (
            <Badge variant="outline" className="ml-1.5 text-[10px] py-0">local</Badge>
          )}
        </td>
        <td className="px-3 py-2 text-muted-foreground text-xs">
          {job.start_timestamp != null && job.end_timestamp != null
            ? formatUtcDateRange(job.start_timestamp, job.end_timestamp)
            : "\u2014"}
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {job.high_threshold}/{job.low_threshold}
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {summary
            ? `${summary.n_spans} span(s)`
            : "\u2014"}
          {job.time_covered_sec != null && (
            <span className="text-xs ml-1">
              ({formatDurationHM(job.time_covered_sec)})
            </span>
          )}
        </td>
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
        <td className="px-3 py-2">
          {job.extract_status ? (
            <Badge className={statusColor[job.extract_status] ?? ""}>
              {job.extract_status}
            </Badge>
          ) : (
            <span className="text-muted-foreground">&mdash;</span>
          )}
        </td>
        <td className="px-3 py-2">
          {job.error_message && (
            <span className="text-red-600 text-xs truncate block max-w-48">
              {job.error_message}
            </span>
          )}
        </td>
      </tr>
      {expanded && canExpand && (
        <tr>
          <td colSpan={10} className="p-0">
            <HydrophoneContentTable
              jobId={job.id}
              isRunning={isRunning}
              windowSizeSeconds={windowSizeSeconds}
              playingKey={playingKey}
              onPlay={onPlay}
              onLabelChange={onLabelChange}
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
  windowSizeSeconds,
  playingKey,
  onPlay,
  onLabelChange,
  labelEdits,
}: {
  jobId: string;
  isRunning: boolean;
  windowSizeSeconds: number | null;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow, clip?: PlayClip) => void;
  onLabelChange: (jobId: string, rk: string, field: LabelField, value: number | null) => void;
  labelEdits: Map<string, Partial<Record<LabelField, number | null>>> | null;
}) {
  const { data: rows = [], isLoading } = useDetectionContent(
    jobId,
    isRunning ? 3000 : undefined,
  );
  const [sortKey, setSortKey] = useState<SortKey>(isRunning ? "filename" : "avg_confidence");
  const [sortDir, setSortDir] = useState<SortDir>(isRunning ? "asc" : "desc");
  const prevRunning = useRef(isRunning);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const tableRef = useRef<HTMLDivElement>(null);

  const hydratedRows = useMemo<HydratedDetectionRow[]>(
    () =>
      rows.map((row) => {
        const resolved = resolveClipTiming(row, windowSizeSeconds);
        return {
          ...row,
          ...resolved,
          _playKey: buildPlaybackKey(
            jobId,
            row.filename,
            resolved._clipStartSec,
            Math.max(0.1, resolved._clipDurationSec),
          ),
        };
      }),
    [rows, windowSizeSeconds, jobId],
  );

  // Switch to confidence desc when job completes
  useEffect(() => {
    if (prevRunning.current && !isRunning) {
      setSortKey("avg_confidence");
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
      const av = sortKey === "avg_confidence" ? a.avg_confidence : a.filename;
      const bv = sortKey === "avg_confidence" ? b.avg_confidence : b.filename;
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      const sa = String(av);
      const sb = String(bv);
      return sortDir === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
    });
  }, [hydratedRows, sortKey, sortDir]);

  const getEffectiveLabel = useCallback(
    (row: DetectionRow, field: LabelField): number | null => {
      const rk = rowKey(row);
      const edit = labelEdits?.get(rk);
      if (edit && field in edit) {
        return edit[field] ?? null;
      }
      return row[field];
    },
    [labelEdits],
  );

  const handleCheckboxClick = useCallback(
    (row: DetectionRow, field: LabelField) => {
      const current = getEffectiveLabel(row, field);
      const next = current === 1 ? 0 : 1;
      onLabelChange(jobId, rowKey(row), field, next);
    },
    [jobId, getEffectiveLabel, onLabelChange],
  );

  // Keyboard shortcuts
  const keyMap: Record<string, LabelField> = { h: "humpback", s: "ship", b: "background" };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

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
          startSec: focused._clipStartSec,
          durationSec: focused._clipDurationSec,
        });
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
  }, [sorted, focusedIndex, handleCheckboxClick, onPlay, jobId]);

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
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b">
            <th className="w-8 px-3 py-1.5" />
            <SortHeader label="Detection Range" field="filename" />
            <SortHeader label="Duration (s)" field="duration_sec" />
            <SortHeader label="Confidence" field="avg_confidence" />
            <th className="px-3 py-1.5 text-center font-medium">Humpback</th>
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
                onClick={() => setFocusedIndex(i)}
              >
                <td className="px-3 py-1.5">
                  <button
                    className="p-0.5 hover:bg-muted rounded"
                    onClick={(e) => {
                      e.stopPropagation();
                      onPlay(jobId, row, {
                        startSec: row._clipStartSec,
                        durationSec: row._clipDurationSec,
                      });
                    }}
                    title={isPlaying ? "Pause" : "Play"}
                  >
                    {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                  </button>
                </td>
                <td
                  className="px-3 py-1.5 max-w-80"
                  title={`Extract range: ${row._clipRange}\nRaw detection: ${row._rawRange}${row._extractFilename ? `\nExtract filename: ${row._extractFilename}` : ""}`}
                >
                  <div className="space-y-0.5 leading-tight">
                    <div className="font-mono clip-range">{row._clipRange}</div>
                    <div className="text-[10px] text-muted-foreground raw-range">
                      raw: {row._rawRange}
                    </div>
                  </div>
                </td>
                <td className="px-3 py-1.5 clip-duration">{row._clipDurationSec.toFixed(1)}</td>
                <td className="px-3 py-1.5">{row.avg_confidence.toFixed(3)}</td>
                <td className="px-3 py-1.5 text-center">
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 cursor-pointer"
                    checked={getEffectiveLabel(row, "humpback") === 1}
                    onChange={() => handleCheckboxClick(row, "humpback")}
                  />
                </td>
                <td className="px-3 py-1.5 text-center">
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 cursor-pointer"
                    checked={getEffectiveLabel(row, "ship") === 1}
                    onChange={() => handleCheckboxClick(row, "ship")}
                  />
                </td>
                <td className="px-3 py-1.5 text-center">
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 cursor-pointer"
                    checked={getEffectiveLabel(row, "background") === 1}
                    onChange={() => handleCheckboxClick(row, "background")}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
