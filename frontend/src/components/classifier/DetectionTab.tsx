import { useState, useMemo, useRef, useCallback } from "react";
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
} from "lucide-react";
import {
  useClassifierModels,
  useDetectionJobs,
  useCreateDetectionJob,
  useBulkDeleteDetectionJobs,
  useDetectionContent,
} from "@/hooks/queries/useClassifier";
import { detectionTsvUrl, detectionAudioSliceUrl } from "@/api/client";
import { FolderBrowser } from "@/components/shared/FolderBrowser";
import { BulkDeleteDialog } from "./BulkDeleteDialog";
import type { DetectionJob, DetectionRow } from "@/api/types";

type SortKey = keyof DetectionRow;
type SortDir = "asc" | "desc";

export function DetectionTab() {
  const { data: models = [] } = useClassifierModels();
  const { data: detectionJobs = [] } = useDetectionJobs(3000);
  const createMutation = useCreateDetectionJob();
  const bulkDeleteMutation = useBulkDeleteDetectionJobs();

  const [selectedModelId, setSelectedModelId] = useState("");
  const [audioFolder, setAudioFolder] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [folderBrowserOpen, setFolderBrowserOpen] = useState(false);

  // Table selection
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  // Expandable rows
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);

  // Audio playback
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

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
          // Browser may reject play if the source isn't ready yet
          setPlayingKey(null);
        });
      }
    },
    [playingKey],
  );

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
            {selectedIds.size > 0 && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => setShowDeleteDialog(true)}
              >
                Delete ({selectedIds.size})
              </Button>
            )}
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
}: {
  job: DetectionJob;
  checked: boolean;
  onToggle: () => void;
  expanded: boolean;
  onExpand: () => void;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow) => void;
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
            : "—"}
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
          {job.error_message && (
            <span className="text-red-600 text-xs truncate block max-w-48">
              {job.error_message}
            </span>
          )}
        </td>
      </tr>
      {expanded && canExpand && (
        <tr>
          <td colSpan={8} className="p-0">
            <DetectionContentTable
              jobId={job.id}
              playingKey={playingKey}
              onPlay={onPlay}
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
}: {
  jobId: string;
  playingKey: string | null;
  onPlay: (jobId: string, row: DetectionRow) => void;
}) {
  const { data: rows = [], isLoading } = useDetectionContent(jobId);
  const [sortKey, setSortKey] = useState<SortKey>("avg_confidence");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

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
    <div className="bg-muted/20 border-t">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b">
            <th className="w-8 px-3 py-1.5" />
            <SortHeader label="File" field="filename" />
            <SortHeader label="Start (s)" field="start_sec" />
            <SortHeader label="End (s)" field="end_sec" />
            <SortHeader label="Avg Confidence" field="avg_confidence" />
            <SortHeader label="Peak Confidence" field="peak_confidence" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const key = `${jobId}:${row.filename}:${row.start_sec}`;
            const isPlaying = playingKey === key;
            return (
              <tr key={i} className="border-b last:border-0 hover:bg-muted/30">
                <td className="px-3 py-1.5">
                  <button
                    className="p-0.5 hover:bg-muted rounded"
                    onClick={() => onPlay(jobId, row)}
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
                <td className="px-3 py-1.5">
                  {row.peak_confidence.toFixed(3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
