import { useState, useRef, useCallback, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  Plus,
  X,
} from "lucide-react";
import {
  useVocClassifierInferenceResults,
  useVocClassifierInferenceJob,
  useVocClassifierModel,
} from "@/hooks/queries/useVocalization";
import {
  useVocalizationLabels,
  useAddVocalizationLabel,
  useDeleteVocalizationLabel,
  useLabelVocabulary,
} from "@/hooks/queries/useLabeling";
import {
  detectionSpectrogramUrl,
  detectionAudioSliceUrl,
} from "@/api/client";
import type {
  VocClassifierPredictionRow,
  VocalizationLabel,
} from "@/api/types";

const PAGE_SIZE = 50;

const LABEL_BADGES: Record<string, string> = {
  humpback: "bg-blue-100 text-blue-800",
  orca: "bg-purple-100 text-purple-800",
  ship: "bg-orange-100 text-orange-800",
  background: "bg-gray-100 text-gray-800",
};

const TYPE_COLORS = [
  "bg-emerald-100 text-emerald-800 border-emerald-200",
  "bg-cyan-100 text-cyan-800 border-cyan-200",
  "bg-violet-100 text-violet-800 border-violet-200",
  "bg-amber-100 text-amber-800 border-amber-200",
  "bg-rose-100 text-rose-800 border-rose-200",
  "bg-teal-100 text-teal-800 border-teal-200",
  "bg-indigo-100 text-indigo-800 border-indigo-200",
  "bg-fuchsia-100 text-fuchsia-800 border-fuchsia-200",
];

type SortMode = "uncertainty" | "score_desc" | "chronological";

interface Props {
  inferenceJobId: string;
  detectionJobId: string;
  onLabelCountChange: (count: number) => void;
}

export function LabelingWorkspace({
  inferenceJobId,
  detectionJobId,
  onLabelCountChange,
}: Props) {
  const { data: job } = useVocClassifierInferenceJob(inferenceJobId);
  const { data: model } = useVocClassifierModel(
    job?.vocalization_model_id ?? null,
  );
  const [page, setPage] = useState(0);
  const [sortMode, setSortMode] = useState<SortMode>("uncertainty");
  const [labelCount, setLabelCount] = useState(0);

  const vocabulary = model?.vocabulary_snapshot ?? [];
  const thresholds = model?.per_class_thresholds ?? {};

  // Build type→color map
  const typeColorMap = useMemo(() => {
    const m = new Map<string, string>();
    vocabulary.forEach((t, i) => {
      m.set(t, TYPE_COLORS[i % TYPE_COLORS.length]);
    });
    return m;
  }, [vocabulary]);

  // Fetch results (API max is 1000 per page)
  const { data: allRows = [], isLoading } = useVocClassifierInferenceResults(
    job?.status === "complete" ? inferenceJobId : null,
    { offset: 0, limit: 1000 },
  );

  // Sort rows
  const sortedRows = useMemo(() => {
    const midpoint =
      vocabulary.length > 0
        ? vocabulary.reduce(
            (sum, t) => sum + (thresholds[t] ?? 0.5),
            0,
          ) / vocabulary.length
        : 0.5;

    const rows = [...allRows];
    switch (sortMode) {
      case "uncertainty":
        rows.sort((a, b) => {
          const aMax = Math.max(...Object.values(a.scores), 0);
          const bMax = Math.max(...Object.values(b.scores), 0);
          return Math.abs(aMax - midpoint) - Math.abs(bMax - midpoint);
        });
        break;
      case "score_desc":
        rows.sort((a, b) => {
          const aMax = Math.max(...Object.values(a.scores), 0);
          const bMax = Math.max(...Object.values(b.scores), 0);
          return bMax - aMax;
        });
        break;
      case "chronological":
        rows.sort(
          (a, b) => (a.start_utc ?? a.start_sec) - (b.start_utc ?? b.start_sec),
        );
        break;
    }
    return rows;
  }, [allRows, sortMode, vocabulary, thresholds]);

  const pageRows = sortedRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / PAGE_SIZE));

  const handleLabelChange = useCallback(
    (delta: number) => {
      setLabelCount((c) => {
        const next = c + delta;
        onLabelCountChange(next);
        return next;
      });
    },
    [onLabelCountChange],
  );

  useEffect(() => {
    setPage(0);
  }, [sortMode]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Loading results...
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <CardTitle className="text-base">
          Labeling ({sortedRows.length} rows)
        </CardTitle>
        <div className="flex items-center gap-2">
          <label className="text-xs text-muted-foreground">Sort:</label>
          <Select
            value={sortMode}
            onValueChange={(v) => setSortMode(v as SortMode)}
          >
            <SelectTrigger className="h-7 w-36 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="uncertainty">Uncertainty</SelectItem>
              <SelectItem value="score_desc">Score (high first)</SelectItem>
              <SelectItem value="chronological">Chronological</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="border rounded-md divide-y">
          {pageRows.map((row, i) => (
            <LabelingRow
              key={`${row.start_utc ?? row.start_sec}-${i}`}
              row={row}
              detectionJobId={detectionJobId}
              vocabulary={vocabulary}
              typeColorMap={typeColorMap}
              thresholds={thresholds}
              onLabelChange={handleLabelChange}
            />
          ))}
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between pt-2">
          <Button
            size="sm"
            variant="outline"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-3.5 w-3.5 mr-1" />
            Prev
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {page + 1} of {totalPages}
          </span>
          <Button
            size="sm"
            variant="outline"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
            <ChevronRight className="h-3.5 w-3.5 ml-1" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function LabelingRow({
  row,
  detectionJobId,
  vocabulary,
  typeColorMap,
  thresholds,
  onLabelChange,
}: {
  row: VocClassifierPredictionRow;
  detectionJobId: string;
  vocabulary: string[];
  typeColorMap: Map<string, string>;
  thresholds: Record<string, number>;
  onLabelChange: (delta: number) => void;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);

  const hasUtc = row.start_utc != null && row.end_utc != null;
  const duration = row.end_sec - row.start_sec;

  // Fetch existing vocalization labels for this row
  const { data: existingLabels = [] } = useVocalizationLabels(
    detectionJobId,
    row.start_utc,
    row.end_utc,
  );

  const addLabel = useAddVocalizationLabel();
  const deleteLabel = useDeleteVocalizationLabel();
  const { data: labelVocab = [] } = useLabelVocabulary();

  // Merge vocabulary sources
  const allTypes = useMemo(() => {
    const set = new Set([...vocabulary, ...labelVocab]);
    return Array.from(set).sort();
  }, [vocabulary, labelVocab]);

  const existingTypeNames = new Set(existingLabels.map((l) => l.label));

  // Inference-predicted tags above threshold (suggestions)
  const predictedTags = useMemo(() => {
    return Object.entries(row.scores)
      .filter(([type, score]) => score >= (thresholds[type] ?? 0.5))
      .sort(([, a], [, b]) => b - a);
  }, [row.scores, thresholds]);

  const predictedTypeNames = new Set(predictedTags.map(([t]) => t));

  // Available types to add (not already assigned or predicted)
  const availableTypes = allTypes.filter(
    (t) => !existingTypeNames.has(t),
  );

  // Max score for display
  const maxScore = Math.max(...Object.values(row.scores), 0);

  // Binary label (read-only) — check for detection row labels
  // The inference results don't carry binary labels, so we show "—" as default
  // A full implementation would fetch from the detection row store
  const binaryLabel = null as string | null;

  const spectrogramSrc = hasUtc
    ? detectionSpectrogramUrl(detectionJobId, row.start_utc!, duration)
    : null;
  const audioSrc = hasUtc
    ? detectionAudioSliceUrl(detectionJobId, row.start_utc!, duration)
    : null;

  function handleAdd(type: string) {
    if (!row.start_utc || !row.end_utc) return;
    addLabel.mutate({
      detectionJobId,
      startUtc: row.start_utc,
      endUtc: row.end_utc,
      label: type,
      source: "manual",
    });
    onLabelChange(1);
  }

  function handleRemove(label: VocalizationLabel) {
    deleteLabel.mutate({
      labelId: label.id,
      detectionJobId,
      startUtc: label.start_utc,
      endUtc: label.end_utc,
    });
    onLabelChange(-1);
  }

  function formatUtcShort(epoch: number): string {
    return new Date(epoch * 1000)
      .toISOString()
      .replace("T", " ")
      .slice(0, 19)
      + " UTC";
  }

  return (
    <div className="flex items-start gap-3 px-3 py-2.5">
      {/* Spectrogram */}
      {spectrogramSrc ? (
        <img
          src={spectrogramSrc}
          alt="spectrogram"
          className="w-28 h-16 object-cover rounded border shrink-0"
          loading="lazy"
        />
      ) : (
        <div className="w-28 h-16 bg-muted rounded border flex items-center justify-center text-xs text-muted-foreground shrink-0">
          no preview
        </div>
      )}

      {/* Info + labels */}
      <div className="flex-1 min-w-0 space-y-1.5">
        <div className="flex items-center gap-2 text-sm">
          {audioSrc && (
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6 shrink-0"
              onClick={() => {
                if (!audioRef.current) return;
                if (playing) audioRef.current.pause();
                else audioRef.current.play();
                setPlaying(!playing);
              }}
            >
              {playing ? (
                <Pause className="h-3 w-3" />
              ) : (
                <Play className="h-3 w-3" />
              )}
            </Button>
          )}
          <span className="text-xs text-muted-foreground">
            Score: {(maxScore * 100).toFixed(0)}%
          </span>
          {audioSrc && (
            <audio
              ref={audioRef}
              src={audioSrc}
              onEnded={() => setPlaying(false)}
              preload="none"
            />
          )}
        </div>

        {/* Binary label + vocalization labels */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-muted-foreground">Binary:</span>
          {binaryLabel ? (
            <Badge
              variant="outline"
              className={`text-xs ${LABEL_BADGES[binaryLabel] ?? ""}`}
            >
              {binaryLabel}
            </Badge>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}

          <span className="text-xs text-muted-foreground ml-2">Voc:</span>
          {predictedTags.map(([type, score]) => (
            <Badge
              key={`pred-${type}`}
              variant="outline"
              className={`text-xs ${typeColorMap.get(type) ?? ""} ${
                existingTypeNames.has(type) ? "ring-1 ring-offset-1" : "opacity-70"
              } cursor-pointer`}
              title={`Predicted: ${(score * 100).toFixed(0)}% — click to add as label`}
              onClick={() => {
                if (!existingTypeNames.has(type)) handleAdd(type);
              }}
            >
              {type} {(score * 100).toFixed(0)}%
            </Badge>
          ))}
          {existingLabels
            .filter((lbl) => !predictedTypeNames.has(lbl.label))
            .map((lbl) => (
            <Badge
              key={lbl.id}
              variant="outline"
              className={`text-xs ${typeColorMap.get(lbl.label) ?? ""} cursor-pointer`}
              onClick={() => handleRemove(lbl)}
            >
              {lbl.label}
              <X className="h-2.5 w-2.5 ml-0.5" />
            </Badge>
          ))}
          {availableTypes.length > 0 && (
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="sm" className="h-5 px-1.5">
                  <Plus className="h-3 w-3" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-40 p-1" align="start">
                {availableTypes.map((t) => (
                  <button
                    key={t}
                    className="w-full text-left text-xs px-2 py-1 hover:bg-muted rounded"
                    onClick={() => handleAdd(t)}
                  >
                    {t}
                  </button>
                ))}
              </PopoverContent>
            </Popover>
          )}
        </div>

        {/* Time range */}
        {hasUtc && (
          <div className="text-xs text-muted-foreground">
            {formatUtcShort(row.start_utc!)} — {formatUtcShort(row.end_utc!)}
          </div>
        )}
      </div>
    </div>
  );
}
