import { useState, useCallback, useMemo, useRef } from "react";
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
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  Plus,
  X,
  Save,
  Undo2,
  RotateCcw,
} from "lucide-react";
import {
  useVocClassifierModels,
  useTrainingDataset,
  useTrainingDatasetRows,
  useCreateVocClassifierTrainingJob,
} from "@/hooks/queries/useVocalization";
import {
  createTrainingDatasetLabel,
  deleteTrainingDatasetLabel,
  trainingDatasetSpectrogramUrl,
  trainingDatasetAudioSliceUrl,
} from "@/api/client";
import { useQueryClient } from "@tanstack/react-query";
import type { TrainingDatasetRow } from "@/api/types";

const PAGE_SIZE = 50;
const NEGATIVE_LABEL = "(Negative)";
const NEGATIVE_COLOR = "bg-red-100 text-red-800 border-red-200";

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

export function TrainingDataView() {
  const { data: models = [] } = useVocClassifierModels();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Only show models with training datasets
  const modelsWithDataset = useMemo(
    () => models.filter((m) => m.training_dataset_id),
    [models],
  );

  const selectedModel = modelsWithDataset.find((m) => m.id === selectedModelId);
  const datasetId = selectedModel?.training_dataset_id ?? null;

  const { data: dataset } = useTrainingDataset(datasetId);

  // Type filter
  const vocabulary = dataset?.vocabulary ?? [];
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [group, setGroup] = useState<"positive" | "negative">("positive");
  const [sourceType, setSourceType] = useState<"detection_job" | null>(null);
  const [page, setPage] = useState(0);

  // Pending label state
  const [pendingAdds, setPendingAdds] = useState<Map<number, Set<string>>>(
    () => new Map(),
  );
  const [pendingRemovals, setPendingRemovals] = useState<
    Map<number, Set<string>>
  >(() => new Map());
  const [saving, setSaving] = useState(false);

  const isDirty = useMemo(() => {
    for (const s of pendingAdds.values()) if (s.size > 0) return true;
    for (const s of pendingRemovals.values()) if (s.size > 0) return true;
    return false;
  }, [pendingAdds, pendingRemovals]);

  const pendingChangeCount = useMemo(() => {
    let count = 0;
    for (const s of pendingAdds.values()) count += s.size;
    for (const s of pendingRemovals.values()) count += s.size;
    return count;
  }, [pendingAdds, pendingRemovals]);

  const typeColorMap = useMemo(() => {
    const m = new Map<string, string>();
    vocabulary.forEach((t, i) => {
      m.set(t, TYPE_COLORS[i % TYPE_COLORS.length]);
    });
    m.set(NEGATIVE_LABEL, NEGATIVE_COLOR);
    return m;
  }, [vocabulary]);

  // Fetch rows
  const { data: rowsResponse, isLoading } = useTrainingDatasetRows(
    datasetId,
    {
      type: selectedType ?? undefined,
      group: selectedType ? group : undefined,
      source_type: sourceType ?? undefined,
      offset: page * PAGE_SIZE,
      limit: PAGE_SIZE,
    },
  );

  const rows = rowsResponse?.rows ?? [];
  const totalRows = rowsResponse?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));

  // Pending label helpers
  const addPending = useCallback((rowIndex: number, label: string) => {
    setPendingAdds((prev) => {
      const next = new Map(prev);
      const s = new Set(next.get(rowIndex) ?? []);
      s.add(label);
      next.set(rowIndex, s);
      return next;
    });
  }, []);

  const removePendingAdd = useCallback((rowIndex: number, label: string) => {
    setPendingAdds((prev) => {
      const next = new Map(prev);
      const s = new Set(next.get(rowIndex) ?? []);
      s.delete(label);
      if (s.size === 0) next.delete(rowIndex);
      else next.set(rowIndex, s);
      return next;
    });
  }, []);

  const addPendingRemoval = useCallback((rowIndex: number, label: string) => {
    setPendingRemovals((prev) => {
      const next = new Map(prev);
      const s = new Set(next.get(rowIndex) ?? []);
      s.add(label);
      next.set(rowIndex, s);
      return next;
    });
  }, []);

  const removePendingRemoval = useCallback(
    (rowIndex: number, label: string) => {
      setPendingRemovals((prev) => {
        const next = new Map(prev);
        const s = new Set(next.get(rowIndex) ?? []);
        s.delete(label);
        if (s.size === 0) next.delete(rowIndex);
        else next.set(rowIndex, s);
        return next;
      });
    },
    [],
  );

  const qc = useQueryClient();
  const retrainMutation = useCreateVocClassifierTrainingJob();

  const handleSave = useCallback(async () => {
    if (!datasetId) return;
    setSaving(true);
    try {
      const addPromises: Promise<unknown>[] = [];
      for (const [rowIndex, labels] of pendingAdds) {
        for (const label of labels) {
          addPromises.push(
            createTrainingDatasetLabel(datasetId, {
              row_index: rowIndex,
              label,
            }),
          );
        }
      }

      const removePromises: Promise<unknown>[] = [];
      for (const [, labelIds] of pendingRemovals) {
        for (const labelId of labelIds) {
          removePromises.push(
            deleteTrainingDatasetLabel(datasetId, labelId),
          );
        }
      }

      await Promise.all([...addPromises, ...removePromises]);
      setPendingAdds(new Map());
      setPendingRemovals(new Map());
      qc.invalidateQueries({
        queryKey: ["vocalization", "training-dataset-rows", datasetId],
      });
    } finally {
      setSaving(false);
    }
  }, [datasetId, pendingAdds, pendingRemovals, qc]);

  const handleCancel = useCallback(() => {
    setPendingAdds(new Map());
    setPendingRemovals(new Map());
  }, []);

  const handleRetrain = useCallback(() => {
    if (!datasetId) return;
    retrainMutation.mutate({ training_dataset_id: datasetId });
  }, [datasetId, retrainMutation]);

  return (
    <div className="space-y-4">
      {/* Model selector */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Training Data Review</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground whitespace-nowrap">
              Model:
            </label>
            <Select
              value={selectedModelId ?? ""}
              onValueChange={(v) => {
                setSelectedModelId(v || null);
                setSelectedType(null);
                setPage(0);
                handleCancel();
              }}
            >
              <SelectTrigger className="w-80">
                <SelectValue placeholder="Select a trained model..." />
              </SelectTrigger>
              <SelectContent>
                {modelsWithDataset.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.name}
                    {m.is_active && " (active)"}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {dataset && (
              <span className="text-sm text-muted-foreground">
                {dataset.total_rows} rows &middot;{" "}
                {dataset.vocabulary.length} types
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {dataset && (
        <>
          {/* Filter bar */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Source type filter */}
            <div className="flex items-center gap-1 border rounded-md p-0.5 mr-2">
              <Button
                size="sm"
                variant={sourceType === null ? "default" : "ghost"}
                className="h-7 text-xs"
                onClick={() => {
                  setSourceType(null);
                  setPage(0);
                }}
              >
                All Sources
              </Button>
              <Button
                size="sm"
                variant={sourceType === "detection_job" ? "default" : "ghost"}
                className="h-7 text-xs"
                onClick={() => {
                  setSourceType("detection_job");
                  setPage(0);
                }}
              >
                Detection
              </Button>
            </div>

            {/* Type filter */}
            <Button
              size="sm"
              variant={selectedType === null ? "default" : "outline"}
              onClick={() => {
                setSelectedType(null);
                setPage(0);
              }}
            >
              All
            </Button>
            {vocabulary.map((t) => (
              <Button
                key={t}
                size="sm"
                variant={selectedType === t ? "default" : "outline"}
                onClick={() => {
                  setSelectedType(t);
                  setGroup("positive");
                  setPage(0);
                }}
              >
                {t}
              </Button>
            ))}

            {/* Positive/Negative toggle */}
            {selectedType && (
              <div className="ml-4 flex items-center gap-1 border rounded-md p-0.5">
                <Button
                  size="sm"
                  variant={group === "positive" ? "default" : "ghost"}
                  className="h-7 text-xs"
                  onClick={() => {
                    setGroup("positive");
                    setPage(0);
                  }}
                >
                  Positive
                </Button>
                <Button
                  size="sm"
                  variant={group === "negative" ? "default" : "ghost"}
                  className="h-7 text-xs"
                  onClick={() => {
                    setGroup("negative");
                    setPage(0);
                  }}
                >
                  Negative
                </Button>
              </div>
            )}
          </div>

          {/* Save/Cancel bar */}
          {isDirty && (
            <div className="sticky top-0 z-10 flex items-center justify-between gap-3 rounded-md border bg-muted/80 backdrop-blur px-3 py-2">
              <span className="text-sm text-muted-foreground">
                <span className="font-medium text-foreground">
                  {pendingChangeCount}
                </span>{" "}
                unsaved change{pendingChangeCount !== 1 ? "s" : ""}
              </span>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={handleCancel}
                  disabled={saving}
                >
                  <Undo2 className="h-3.5 w-3.5 mr-1" />
                  Cancel
                </Button>
                <Button size="sm" onClick={handleSave} disabled={saving}>
                  <Save className="h-3.5 w-3.5 mr-1" />
                  {saving ? "Saving..." : "Save"}
                </Button>
              </div>
            </div>
          )}

          {/* Row list */}
          {isLoading ? (
            <Card>
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                Loading rows...
              </CardContent>
            </Card>
          ) : (
            <div className="border rounded-md divide-y">
              {rows.map((row) => (
                <TrainingDataRow
                  key={row.row_index}
                  row={row}
                  datasetId={datasetId!}
                  vocabulary={vocabulary}
                  typeColorMap={typeColorMap}
                  pendingAdds={pendingAdds.get(row.row_index)}
                  pendingRemovals={pendingRemovals.get(row.row_index)}
                  onAddPending={(label) => addPending(row.row_index, label)}
                  onRemovePendingAdd={(label) =>
                    removePendingAdd(row.row_index, label)
                  }
                  onAddPendingRemoval={(label) =>
                    addPendingRemoval(row.row_index, label)
                  }
                  onRemovePendingRemoval={(label) =>
                    removePendingRemoval(row.row_index, label)
                  }
                />
              ))}
              {rows.length === 0 && (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  No rows match the current filter.
                </div>
              )}
            </div>
          )}

          {/* Pagination */}
          <div className="flex items-center justify-between">
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
              Page {page + 1} of {totalPages} ({totalRows} rows)
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

          {/* Retrain footer */}
          <Card>
            <CardContent className="py-3 flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Retrain using this dataset&apos;s current labels
              </span>
              <Button
                size="sm"
                disabled={isDirty || retrainMutation.isPending}
                onClick={handleRetrain}
              >
                <RotateCcw className="h-3.5 w-3.5 mr-1" />
                {retrainMutation.isPending ? "Retraining..." : "Retrain"}
              </Button>
            </CardContent>
          </Card>
          {retrainMutation.isSuccess && (
            <div className="text-sm text-green-600">
              Training job created. Check the Training tab for progress.
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────────────────── */
/*  TrainingDataRow                                          */
/* ────────────────────────────────────────────────────────── */

function TrainingDataRow({
  row,
  datasetId,
  vocabulary,
  typeColorMap,
  pendingAdds,
  pendingRemovals,
  onAddPending,
  onRemovePendingAdd,
  onAddPendingRemoval,
  onRemovePendingRemoval,
}: {
  row: TrainingDatasetRow;
  datasetId: string;
  vocabulary: string[];
  typeColorMap: Map<string, string>;
  pendingAdds: Set<string> | undefined;
  pendingRemovals: Set<string> | undefined;
  onAddPending: (label: string) => void;
  onRemovePendingAdd: (label: string) => void;
  onAddPendingRemoval: (label: string) => void;
  onRemovePendingRemoval: (label: string) => void;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [spectrogramOpen, setSpectrogramOpen] = useState(false);

  const duration = row.end_sec - row.start_sec;
  const pendingAddSet = pendingAdds ?? new Set<string>();
  // pendingRemovals now stores label IDs (UUIDs), not label names
  const pendingRemovalIdSet = pendingRemovals ?? new Set<string>();

  // Derive name-based sets from label objects for display/filter logic
  const existingLabelNames = useMemo(
    () => new Set(row.labels.map((l) => l.label)),
    [row.labels],
  );
  // Set of label names whose IDs are pending removal
  const pendingRemovalNames = useMemo(() => {
    const names = new Set<string>();
    for (const lbl of row.labels) {
      if (pendingRemovalIdSet.has(lbl.id)) names.add(lbl.label);
    }
    return names;
  }, [row.labels, pendingRemovalIdSet]);

  const allTypes = useMemo(() => {
    const set = new Set([...vocabulary]);
    set.delete(NEGATIVE_LABEL);
    return Array.from(set).sort();
  }, [vocabulary]);

  const hasNegative =
    (existingLabelNames.has(NEGATIVE_LABEL) &&
      !pendingRemovalNames.has(NEGATIVE_LABEL)) ||
    pendingAddSet.has(NEGATIVE_LABEL);

  // Available types to add (exclude existing non-removed and pending adds)
  const availableTypes = allTypes.filter(
    (t) =>
      (!existingLabelNames.has(t) || pendingRemovalNames.has(t)) &&
      !pendingAddSet.has(t),
  );
  const showNegativeOption = !hasNegative;

  function handleAddWithExclusivity(label: string) {
    if (label === NEGATIVE_LABEL) {
      for (const t of pendingAddSet) {
        if (t !== NEGATIVE_LABEL) onRemovePendingAdd(t);
      }
      // Mark existing type labels for removal by their IDs
      for (const lbl of row.labels) {
        if (lbl.label !== NEGATIVE_LABEL && !pendingRemovalIdSet.has(lbl.id)) {
          onAddPendingRemoval(lbl.id);
        }
      }
      onAddPending(NEGATIVE_LABEL);
    } else {
      if (pendingAddSet.has(NEGATIVE_LABEL)) {
        onRemovePendingAdd(NEGATIVE_LABEL);
      }
      // Mark existing (Negative) label for removal by its ID
      for (const lbl of row.labels) {
        if (lbl.label === NEGATIVE_LABEL && !pendingRemovalIdSet.has(lbl.id)) {
          onAddPendingRemoval(lbl.id);
        }
      }
      onAddPending(label);
    }
  }

  const spectrogramSrc = trainingDatasetSpectrogramUrl(
    datasetId,
    row.row_index,
  );
  const audioSrc = trainingDatasetAudioSliceUrl(datasetId, row.row_index);

  function togglePlayback() {
    if (!audioRef.current) {
      const a = new Audio(audioSrc);
      a.addEventListener("ended", () => setPlaying(false));
      audioRef.current = a;
    }
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.play();
      setPlaying(true);
    }
  }

  return (
    <div className="flex gap-3 p-3 items-start">
      {/* Large inline spectrogram */}
      <img
        src={spectrogramSrc}
        alt={`Spectrogram row ${row.row_index}`}
        className="rounded border cursor-pointer hover:ring-2 hover:ring-primary/50 shrink-0"
        style={{ width: 400, height: 120 }}
        loading="lazy"
        onClick={() => setSpectrogramOpen(true)}
      />

      {/* Row info + labels */}
      <div className="flex-1 min-w-0 space-y-2">
        {/* Header: filename, times, source */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="font-mono truncate">{row.filename}</span>
          <span>
            {row.start_sec.toFixed(1)}–{row.end_sec.toFixed(1)}s
          </span>
          {row.confidence != null && (
            <Badge variant="outline" className="text-[10px] h-4">
              conf {row.confidence.toFixed(2)}
            </Badge>
          )}
          <Badge variant="secondary" className="text-[10px] h-4">
            {row.source_type === "detection_job" ? "detection" : "legacy"}
          </Badge>
        </div>

        {/* Label tags */}
        <div className="flex flex-wrap gap-1.5 items-center">
          {/* Saved labels */}
          {row.labels.map((labelObj) => {
            const isPendingRemoval = pendingRemovalIdSet.has(labelObj.id);
            const color =
              typeColorMap.get(labelObj.label) ?? TYPE_COLORS[0];
            return (
              <Badge
                key={labelObj.id}
                variant="outline"
                className={`cursor-pointer select-none text-xs ${color} ${
                  isPendingRemoval ? "opacity-40 line-through" : ""
                }`}
                onClick={() => {
                  if (isPendingRemoval) {
                    onRemovePendingRemoval(labelObj.id);
                  } else {
                    onAddPendingRemoval(labelObj.id);
                  }
                }}
              >
                {labelObj.label}
                {isPendingRemoval && (
                  <X className="h-3 w-3 ml-0.5" />
                )}
              </Badge>
            );
          })}

          {/* Pending adds */}
          {Array.from(pendingAddSet).map((label) => (
            <Badge
              key={`add-${label}`}
              variant="outline"
              className={`cursor-pointer select-none text-xs ring-2 ring-primary/50 ${
                typeColorMap.get(label) ?? TYPE_COLORS[0]
              }`}
              onClick={() => onRemovePendingAdd(label)}
            >
              + {label}
              <X className="h-3 w-3 ml-0.5" />
            </Badge>
          ))}

          {/* Add button */}
          {(availableTypes.length > 0 || showNegativeOption) && (
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-5 w-5 p-0 rounded-full"
                >
                  <Plus className="h-3.5 w-3.5" />
                </Button>
              </PopoverTrigger>
              <PopoverContent
                className="w-auto p-1"
                side="right"
                align="start"
              >
                <div className="flex flex-col gap-0.5">
                  {availableTypes.map((t) => (
                    <Button
                      key={t}
                      variant="ghost"
                      size="sm"
                      className="justify-start h-7 text-xs px-2"
                      onClick={() => handleAddWithExclusivity(t)}
                    >
                      {t}
                    </Button>
                  ))}
                  {showNegativeOption && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="justify-start h-7 text-xs px-2 text-red-600"
                      onClick={() =>
                        handleAddWithExclusivity(NEGATIVE_LABEL)
                      }
                    >
                      {NEGATIVE_LABEL}
                    </Button>
                  )}
                </div>
              </PopoverContent>
            </Popover>
          )}
        </div>

        {/* Playback */}
        <Button
          variant="ghost"
          size="sm"
          className="h-6 text-xs"
          onClick={togglePlayback}
        >
          {playing ? (
            <Pause className="h-3 w-3 mr-1" />
          ) : (
            <Play className="h-3 w-3 mr-1" />
          )}
          {duration.toFixed(1)}s
        </Button>
      </div>

      {/* Spectrogram dialog */}
      <Dialog open={spectrogramOpen} onOpenChange={setSpectrogramOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle className="text-sm font-mono">
              {row.filename} ({row.start_sec.toFixed(1)}–
              {row.end_sec.toFixed(1)}s)
            </DialogTitle>
          </DialogHeader>
          <img
            src={spectrogramSrc}
            alt="Spectrogram"
            className="w-full rounded"
          />
          <div className="flex items-center gap-2 mt-2">
            <Button size="sm" variant="outline" onClick={togglePlayback}>
              {playing ? (
                <Pause className="h-3.5 w-3.5 mr-1" />
              ) : (
                <Play className="h-3.5 w-3.5 mr-1" />
              )}
              {playing ? "Pause" : "Play"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
