import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ChevronDown,
  ChevronRight,
  FolderOpen,
  Play,
  Pause,
  ArrowUp,
  ArrowDown,
  Download,
  Save,
  PackageOpen,
} from "lucide-react";
import {
  useClassifierModels,
  useDetectionJobs,
  useCreateDetectionJob,
  useBulkDeleteDetectionJobs,
  useDetectionContent,
  useSaveDetectionLabels,
  useExtractionSettings,
  useExtractLabeledSamples,
} from "@/hooks/queries/useClassifier";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { detectionTsvUrl, detectionAudioSliceUrl } from "@/api/client";
import { FolderBrowser } from "@/components/shared/FolderBrowser";
import { BulkDeleteDialog } from "./BulkDeleteDialog";
import type { DetectionJob, DetectionRow, DetectionLabelRow } from "@/api/types";

type SortKey = "filename" | "start_sec" | "end_sec" | "avg_confidence";
type SortDir = "asc" | "desc";

type LabelField = "humpback" | "ship" | "background";

// Key for identifying a detection row: "filename:start_sec:end_sec"
function rowKey(row: { filename: string; start_sec: number; end_sec: number }): string {
  return `${row.filename}:${row.start_sec}:${row.end_sec}`;
}

export function DetectionTab() {
  const { data: models = [] } = useClassifierModels();
  const { data: detectionJobs = [] } = useDetectionJobs(3000);
  const createMutation = useCreateDetectionJob();
  const bulkDeleteMutation = useBulkDeleteDetectionJobs();
  const saveLabelsMutation = useSaveDetectionLabels();
  const extractMutation = useExtractLabeledSamples();

  const [selectedModelId, setSelectedModelId] = useState("");
  const [audioFolder, setAudioFolder] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [folderBrowserOpen, setFolderBrowserOpen] = useState(false);

  // Table selection
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showExtractDialog, setShowExtractDialog] = useState(false);

  // Expandable rows
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);

  // Audio playback
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

  // Label edits: jobId -> rowKey -> { humpback, ship, background }
  const [labelEdits, setLabelEdits] = useState<
    Map<string, Map<string, Partial<Record<LabelField, number | null>>>>
  >(new Map());
  const [dirtyJobs, setDirtyJobs] = useState<Set<string>>(new Set());

  const hasActiveJobs = detectionJobs.some(
    (j) => j.status === "queued" || j.status === "running",
  );

  const handleSubmit = () => {
    if (!selectedModelId || !audioFolder) return;
    createMutation.mutate(
      {
        classifier_model_id: selectedModelId,
        audio_folder: audioFolder,
        confidence_threshold: threshold,
      },
      {
        onSuccess: () => {
          setAudioFolder("");
        },
      },
    );
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

  const allSelected =
    detectionJobs.length > 0 &&
    detectionJobs.every((j) => selectedIds.has(j.id));
  const someSelected = detectionJobs.some((j) => selectedIds.has(j.id));

  const toggleAll = () => {
    if (allSelected) setSelectedIds(new Set());
    else setSelectedIds(new Set(detectionJobs.map((j) => j.id)));
  };

  const handlePlay = useCallback(
    (jobId: string, row: DetectionRow) => {
      const key = `${jobId}:${row.filename}:${row.start_sec}`;
      if (playingKey === key) {
        audioRef.current?.pause();
        setPlayingKey(null);
        return;
      }
      // Use the full window duration (5 s) as minimum to ensure audible playback,
      // since span end_sec marks the start of the next window boundary.
      const spanDuration = row.end_sec - row.start_sec;
      const duration = Math.max(spanDuration, 5);
      const url = detectionAudioSliceUrl(
        jobId,
        row.filename,
        row.start_sec,
        duration,
      );
      if (audioRef.current) {
        const audio = audioRef.current;
        audio.src = url;
        audio.load();
        setPlayingKey(key);
        audio.play().catch(() => {
          setPlayingKey(null);
        });
      }
    },
    [playingKey],
  );

  // Label editing
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

      promises.push(
        saveLabelsMutation.mutateAsync({ jobId, rows }),
      );
    }
    await Promise.all(promises);
    setLabelEdits(new Map());
    setDirtyJobs(new Set());
  }, [dirtyJobs, labelEdits, saveLabelsMutation]);

  return (
    <div className="space-y-4">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        onEnded={() => setPlayingKey(null)}
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Run Detection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
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
            {models.length === 0 && (
              <p className="text-xs text-muted-foreground mt-1">
                No trained models. Train a classifier first.
              </p>
            )}
          </div>
          <div>
            <label className="text-sm font-medium">Audio Folder Path</label>
            <div className="flex gap-2">
              <Input
                value={audioFolder}
                onChange={(e) => setAudioFolder(e.target.value)}
                placeholder="/path/to/hydrophone/recordings"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => setFolderBrowserOpen(true)}
                title="Browse folders"
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
            </div>
          </div>
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
          <Button
            onClick={handleSubmit}
            disabled={
              !selectedModelId || !audioFolder || createMutation.isPending
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

      {/* Detection Jobs Table */}
      {detectionJobs.length > 0 && (
        <div className="border rounded-md">
          <div className="flex items-center justify-between px-4 py-3 border-b">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">Detection Jobs</h3>
              <Badge variant="secondary">{detectionJobs.length}</Badge>
              {hasActiveJobs && (
                <span className="text-xs text-muted-foreground">
                  (polling…)
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={dirtyJobs.size === 0 || saveLabelsMutation.isPending}
                onClick={handleSaveLabels}
              >
                <Save className="h-3.5 w-3.5 mr-1" />
                {saveLabelsMutation.isPending ? "Saving…" : "Save Labels"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={selectedIds.size === 0}
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
                      allSelected
                        ? true
                        : someSelected
                          ? "indeterminate"
                          : false
                    }
                    onCheckedChange={toggleAll}
                  />
                </th>
                <th className="w-8 px-1 py-2" />
                <th className="px-3 py-2 text-left font-medium">Status</th>
                <th className="px-3 py-2 text-left font-medium">
                  Audio Folder
                </th>
                <th className="px-3 py-2 text-left font-medium">Threshold</th>
                <th className="px-3 py-2 text-left font-medium">Results</th>
                <th className="px-3 py-2 text-left font-medium">Download</th>
                <th className="px-3 py-2 text-left font-medium">Extract</th>
                <th className="px-3 py-2 text-left font-medium">Error</th>
              </tr>
            </thead>
            <tbody>
              {detectionJobs.map((job) => (
                <DetectionJobTableRow
                  key={job.id}
                  job={job}
                  checked={selectedIds.has(job.id)}
                  onToggle={() => toggleId(job.id)}
                  expanded={expandedJobId === job.id}
                  onExpand={() =>
                    setExpandedJobId(
                      expandedJobId === job.id ? null : job.id,
                    )
                  }
                  playingKey={playingKey}
                  onPlay={handlePlay}
                  labelEdits={labelEdits.get(job.id) ?? null}
                  onLabelChange={handleLabelChange}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Dialogs */}
      <FolderBrowser
        open={folderBrowserOpen}
        onOpenChange={setFolderBrowserOpen}
        onSelect={setAudioFolder}
        initialPath={audioFolder || "/"}
      />

      <BulkDeleteDialog
        open={showDeleteDialog}
        onOpenChange={setShowDeleteDialog}
        count={selectedIds.size}
        entityName="detection job"
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
        selectedIds={selectedIds}
        extractMutation={extractMutation}
        onSuccess={() => setSelectedIds(new Set())}
      />
    </div>
  );
}

const statusColor: Record<string, string> = {
  queued: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  complete: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  canceled: "bg-gray-100 text-gray-800",
};

function DetectionJobTableRow({
  job,
  checked,
  onToggle,
  expanded,
  onExpand,
  playingKey,
  onPlay,
  labelEdits,
  onLabelChange,
}: {
  job: DetectionJob;
  checked: boolean;
  onToggle: () => void;
  expanded: boolean;
  onExpand: () => void;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow) => void;
  labelEdits: Map<string, Partial<Record<LabelField, number | null>>> | null;
  onLabelChange: (jobId: string, rk: string, field: LabelField, value: number | null) => void;
}) {
  const summary = job.result_summary as Record<string, number> | null;
  const canExpand = job.status === "complete" && job.output_tsv_path;

  return (
    <>
      <tr className="border-b hover:bg-muted/30">
        <td className="px-3 py-2">
          <Checkbox checked={checked} onCheckedChange={onToggle} />
        </td>
        <td className="px-1 py-2">
          {canExpand && (
            <button
              className="p-0.5 hover:bg-muted rounded"
              onClick={onExpand}
            >
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
        <td
          className="px-3 py-2 text-muted-foreground truncate max-w-48"
          title={job.audio_folder}
        >
          {job.audio_folder}
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {job.confidence_threshold}
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {summary
            ? `${summary.n_spans} span(s) in ${summary.n_files} file(s)`
            : "\u2014"}
        </td>
        <td className="px-3 py-2">
          {canExpand && (
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
          <td colSpan={9} className="p-0">
            <DetectionContentTable
              jobId={job.id}
              playingKey={playingKey}
              onPlay={onPlay}
              labelEdits={labelEdits}
              onLabelChange={onLabelChange}
            />
          </td>
        </tr>
      )}
    </>
  );
}

function DetectionContentTable({
  jobId,
  playingKey,
  onPlay,
  labelEdits,
  onLabelChange,
}: {
  jobId: string;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow) => void;
  labelEdits: Map<string, Partial<Record<LabelField, number | null>>> | null;
  onLabelChange: (jobId: string, rk: string, field: LabelField, value: number | null) => void;
}) {
  const { data: rows = [], isLoading } = useDetectionContent(jobId);
  const [sortKey, setSortKey] = useState<SortKey>("avg_confidence");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const tableRef = useRef<HTMLDivElement>(null);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      const sa = String(av);
      const sb = String(bv);
      return sortDir === "asc"
        ? sa.localeCompare(sb)
        : sb.localeCompare(sa);
    });
  }, [rows, sortKey, sortDir]);

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

  // Keyboard shortcuts: h/s/b toggle labels on focused row, arrow keys navigate
  const keyMap: Record<string, LabelField> = { h: "humpback", s: "ship", b: "background" };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Skip when typing in an input, select, or textarea
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
        onPlay(jobId, sorted[focusedIndex]);
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

  // Scroll focused row into view
  useEffect(() => {
    if (focusedIndex === null || !tableRef.current) return;
    const row = tableRef.current.querySelector(`tbody tr:nth-child(${focusedIndex + 1})`);
    row?.scrollIntoView({ block: "nearest" });
  }, [focusedIndex]);

  if (isLoading) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Loading detections…
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="p-4 text-sm text-muted-foreground">No detections</div>
    );
  }

  const SortHeader = ({
    label,
    field,
  }: {
    label: string;
    field: SortKey;
  }) => (
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
            <SortHeader label="File" field="filename" />
            <SortHeader label="Start (s)" field="start_sec" />
            <SortHeader label="End (s)" field="end_sec" />
            <SortHeader label="Confidence" field="avg_confidence" />
            <th className="px-3 py-1.5 text-center font-medium">Humpback</th>
            <th className="px-3 py-1.5 text-center font-medium">Ship</th>
            <th className="px-3 py-1.5 text-center font-medium">Background</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const key = `${jobId}:${row.filename}:${row.start_sec}`;
            const isPlaying = playingKey === key;
            const isFocused = focusedIndex === i;
            return (
              <tr
                key={i}
                className={`border-b last:border-0 cursor-pointer ${
                  isFocused
                    ? "bg-blue-100 dark:bg-blue-900/30"
                    : "hover:bg-muted/30"
                }`}
                onClick={() => setFocusedIndex(i)}
              >
                <td className="px-3 py-1.5">
                  <button
                    className="p-0.5 hover:bg-muted rounded"
                    onClick={(e) => {
                      e.stopPropagation();
                      onPlay(jobId, row);
                    }}
                    title={isPlaying ? "Pause" : "Play"}
                  >
                    {isPlaying ? (
                      <Pause className="h-3.5 w-3.5" />
                    ) : (
                      <Play className="h-3.5 w-3.5" />
                    )}
                  </button>
                </td>
                <td
                  className="px-3 py-1.5 truncate max-w-40"
                  title={row.filename}
                >
                  {row.filename}
                </td>
                <td className="px-3 py-1.5">{row.start_sec.toFixed(1)}</td>
                <td className="px-3 py-1.5">{row.end_sec.toFixed(1)}</td>
                <td className="px-3 py-1.5">
                  {row.avg_confidence.toFixed(3)}
                </td>
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

function ExtractDialog({
  open,
  onOpenChange,
  selectedIds,
  extractMutation,
  onSuccess,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedIds: Set<string>;
  extractMutation: ReturnType<typeof useExtractLabeledSamples>;
  onSuccess: () => void;
}) {
  const { data: defaults } = useExtractionSettings();
  const [posPath, setPosPath] = useState("");
  const [negPath, setNegPath] = useState("");
  const [posBrowserOpen, setPosBrowserOpen] = useState(false);
  const [negBrowserOpen, setNegBrowserOpen] = useState(false);

  // Initialize paths from defaults when loaded
  useEffect(() => {
    if (defaults && !posPath) setPosPath(defaults.positive_output_path);
    if (defaults && !negPath) setNegPath(defaults.negative_output_path);
  }, [defaults]);

  const handleConfirm = () => {
    extractMutation.mutate(
      {
        jobIds: [...selectedIds],
        positiveOutputPath: posPath || undefined,
        negativeOutputPath: negPath || undefined,
      },
      {
        onSuccess: () => {
          onOpenChange(false);
          onSuccess();
        },
      },
    );
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Extract Labeled Samples</DialogTitle>
            <DialogDescription>
              Extract labeled audio segments from {selectedIds.size} selected detection job
              {selectedIds.size !== 1 ? "s" : ""} as WAV files.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <label className="text-sm font-medium">Positive Output Path (humpback)</label>
              <div className="flex gap-2 mt-1">
                <Input
                  value={posPath}
                  onChange={(e) => setPosPath(e.target.value)}
                  placeholder="data/samples/positive"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setPosBrowserOpen(true)}
                  title="Browse folders"
                >
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div>
              <label className="text-sm font-medium">Negative Output Path (ship/background)</label>
              <div className="flex gap-2 mt-1">
                <Input
                  value={negPath}
                  onChange={(e) => setNegPath(e.target.value)}
                  placeholder="data/samples/negative"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setNegBrowserOpen(true)}
                  title="Browse folders"
                >
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={extractMutation.isPending}
            >
              Cancel
            </Button>
            <Button onClick={handleConfirm} disabled={extractMutation.isPending}>
              {extractMutation.isPending ? "Queuing…" : "Extract"}
            </Button>
          </DialogFooter>
          {extractMutation.isError && (
            <p className="text-sm text-red-600">
              {(extractMutation.error as Error).message}
            </p>
          )}
        </DialogContent>
      </Dialog>

      <FolderBrowser
        open={posBrowserOpen}
        onOpenChange={setPosBrowserOpen}
        onSelect={setPosPath}
        initialPath={posPath || "/"}
      />
      <FolderBrowser
        open={negBrowserOpen}
        onOpenChange={setNegBrowserOpen}
        onSelect={setNegPath}
        initialPath={negPath || "/"}
      />
    </>
  );
}
