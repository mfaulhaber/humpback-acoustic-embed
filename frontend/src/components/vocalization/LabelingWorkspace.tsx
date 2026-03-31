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
  Eye,
} from "lucide-react";
import {
  useVocClassifierInferenceResults,
  useVocClassifierInferenceJob,
  useVocClassifierModel,
} from "@/hooks/queries/useVocalization";
import {
  useVocalizationLabels,
  useLabelVocabulary,
} from "@/hooks/queries/useLabeling";
import {
  createVocalizationLabel,
  deleteVocalizationLabel,
  detectionSpectrogramUrl,
  detectionAudioSliceUrl,
} from "@/api/client";
import { useQueryClient } from "@tanstack/react-query";
import type {
  VocClassifierPredictionRow,
  VocalizationLabel,
  LabelingSource,
} from "@/api/types";

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

type SortMode = "uncertainty" | "score_desc" | "chronological" | "confidence_desc";

/** Row key for pending label maps */
function rowKey(startUtc: number | null, endUtc: number | null): string {
  return `${startUtc ?? 0}_${endUtc ?? 0}`;
}

interface Props {
  inferenceJobId: string;
  source: LabelingSource;
  readonly: boolean;
  onLabelCountChange: (count: number) => void;
}

export function LabelingWorkspace({
  inferenceJobId,
  source,
  readonly,
  onLabelCountChange,
}: Props) {
  const { data: job } = useVocClassifierInferenceJob(inferenceJobId);
  const { data: model } = useVocClassifierModel(
    job?.vocalization_model_id ?? null,
  );
  const [page, setPage] = useState(0);
  const [sortMode, setSortMode] = useState<SortMode>(
    source.type === "detection_job" ? "confidence_desc" : "score_desc",
  );
  const [saving, setSaving] = useState(false);

  // Pending label state (local accumulation)
  const [pendingAdds, setPendingAdds] = useState<Map<string, Set<string>>>(
    () => new Map(),
  );
  const [pendingRemovals, setPendingRemovals] = useState<
    Map<string, Set<string>>
  >(() => new Map());

  const vocabulary = model?.vocabulary_snapshot ?? [];
  const thresholds = model?.per_class_thresholds ?? {};

  // Derive dirty state
  const isDirty = useMemo(() => {
    for (const s of pendingAdds.values()) if (s.size > 0) return true;
    for (const s of pendingRemovals.values()) if (s.size > 0) return true;
    return false;
  }, [pendingAdds, pendingRemovals]);

  // Count total pending changes
  const pendingChangeCount = useMemo(() => {
    let count = 0;
    for (const s of pendingAdds.values()) count += s.size;
    for (const s of pendingRemovals.values()) count += s.size;
    return count;
  }, [pendingAdds, pendingRemovals]);

  // Build type→color map
  const typeColorMap = useMemo(() => {
    const m = new Map<string, string>();
    vocabulary.forEach((t, i) => {
      m.set(t, TYPE_COLORS[i % TYPE_COLORS.length]);
    });
    m.set(NEGATIVE_LABEL, NEGATIVE_COLOR);
    return m;
  }, [vocabulary]);

  // Fetch results (API max is 1000 per page)
  const { data: allRows = [], isLoading } = useVocClassifierInferenceResults(
    job?.status === "complete" ? inferenceJobId : null,
    {
      offset: 0,
      limit: 1000,
      sort: sortMode === "confidence_desc" ? "confidence_desc" : undefined,
    },
  );

  // Sort rows (confidence_desc is server-side sorted, skip client re-sort)
  const sortedRows = useMemo(() => {
    if (sortMode === "confidence_desc") return allRows;

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
          (a, b) =>
            (a.start_utc ?? a.start_sec) - (b.start_utc ?? b.start_sec),
        );
        break;
    }
    return rows;
  }, [allRows, sortMode, vocabulary, thresholds]);

  const hasConfidence = allRows.length > 0 && allRows[0].confidence != null;

  const pageRows = sortedRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / PAGE_SIZE));

  // Pending label helpers
  const addPending = useCallback(
    (key: string, label: string) => {
      setPendingAdds((prev) => {
        const next = new Map(prev);
        const s = new Set(next.get(key) ?? []);
        s.add(label);
        next.set(key, s);
        return next;
      });
    },
    [],
  );

  const removePendingAdd = useCallback(
    (key: string, label: string) => {
      setPendingAdds((prev) => {
        const next = new Map(prev);
        const s = new Set(next.get(key) ?? []);
        s.delete(label);
        if (s.size === 0) next.delete(key);
        else next.set(key, s);
        return next;
      });
    },
    [],
  );

  const addPendingRemoval = useCallback(
    (key: string, labelId: string) => {
      setPendingRemovals((prev) => {
        const next = new Map(prev);
        const s = new Set(next.get(key) ?? []);
        s.add(labelId);
        next.set(key, s);
        return next;
      });
    },
    [],
  );

  const removePendingRemoval = useCallback(
    (key: string, labelId: string) => {
      setPendingRemovals((prev) => {
        const next = new Map(prev);
        const s = new Set(next.get(key) ?? []);
        s.delete(labelId);
        if (s.size === 0) next.delete(key);
        else next.set(key, s);
        return next;
      });
    },
    [],
  );

  // Save / Cancel — use raw API calls to avoid per-mutation query invalidation
  const qc = useQueryClient();

  const detectionJobId =
    source.type === "detection_job" ? source.jobId : null;

  const handleSave = useCallback(async () => {
    if (!detectionJobId) return;
    setSaving(true);
    let netNew = 0;

    try {
      // Process all adds via raw API (no per-call invalidation)
      const addPromises: Promise<unknown>[] = [];
      for (const [key, labels] of pendingAdds) {
        const [startStr, endStr] = key.split("_");
        const startUtc = parseFloat(startStr);
        const endUtc = parseFloat(endStr);
        for (const label of labels) {
          netNew++;
          addPromises.push(
            createVocalizationLabel(detectionJobId, startUtc, endUtc, {
              label,
              source: "manual",
            }),
          );
        }
      }

      // Process all removals via raw API
      const removePromises: Promise<unknown>[] = [];
      for (const [, labelIds] of pendingRemovals) {
        for (const labelId of labelIds) {
          removePromises.push(deleteVocalizationLabel(labelId));
        }
      }

      await Promise.all([...addPromises, ...removePromises]);

      // Clear pending state
      setPendingAdds(new Map());
      setPendingRemovals(new Map());

      // Single batch invalidation
      qc.invalidateQueries({ queryKey: ["labeling"] });

      // Update parent label count
      onLabelCountChange(netNew);
    } finally {
      setSaving(false);
    }
  }, [
    detectionJobId,
    pendingAdds,
    pendingRemovals,
    qc,
    onLabelCountChange,
  ]);

  const handleCancel = useCallback(() => {
    setPendingAdds(new Map());
    setPendingRemovals(new Map());
  }, []);

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
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">
            Labeling ({sortedRows.length} rows)
          </CardTitle>
          {readonly && (
            <Badge variant="secondary" className="text-xs">
              <Eye className="h-3 w-3 mr-1" />
              View only
            </Badge>
          )}
        </div>
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
              {hasConfidence && (
                <SelectItem value="confidence_desc">
                  Detection Confidence
                </SelectItem>
              )}
              <SelectItem value="score_desc">Score (high first)</SelectItem>
              <SelectItem value="uncertainty">Uncertainty</SelectItem>
              <SelectItem value="chronological">Chronological</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {/* Save / Cancel bar */}
        {!readonly && isDirty && (
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

        <div className="border rounded-md divide-y">
          {pageRows.map((row, i) => (
            <LabelingRow
              key={`${row.start_utc ?? row.start_sec}-${i}`}
              row={row}
              detectionJobId={detectionJobId}
              vocabulary={vocabulary}
              typeColorMap={typeColorMap}
              thresholds={thresholds}
              readonly={readonly}
              pendingAdds={pendingAdds.get(rowKey(row.start_utc, row.end_utc))}
              pendingRemovals={pendingRemovals.get(
                rowKey(row.start_utc, row.end_utc),
              )}
              onAddPending={(label) =>
                addPending(rowKey(row.start_utc, row.end_utc), label)
              }
              onRemovePendingAdd={(label) =>
                removePendingAdd(rowKey(row.start_utc, row.end_utc), label)
              }
              onAddPendingRemoval={(labelId) =>
                addPendingRemoval(
                  rowKey(row.start_utc, row.end_utc),
                  labelId,
                )
              }
              onRemovePendingRemoval={(labelId) =>
                removePendingRemoval(
                  rowKey(row.start_utc, row.end_utc),
                  labelId,
                )
              }
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

/* ────────────────────────────────────────────────────────── */
/*  LabelingRow                                               */
/* ────────────────────────────────────────────────────────── */

function LabelingRow({
  row,
  detectionJobId,
  vocabulary,
  typeColorMap,
  thresholds,
  readonly,
  pendingAdds,
  pendingRemovals,
  onAddPending,
  onRemovePendingAdd,
  onAddPendingRemoval,
  onRemovePendingRemoval,
}: {
  row: VocClassifierPredictionRow;
  detectionJobId: string | null;
  vocabulary: string[];
  typeColorMap: Map<string, string>;
  thresholds: Record<string, number>;
  readonly: boolean;
  pendingAdds: Set<string> | undefined;
  pendingRemovals: Set<string> | undefined;
  onAddPending: (label: string) => void;
  onRemovePendingAdd: (label: string) => void;
  onAddPendingRemoval: (labelId: string) => void;
  onRemovePendingRemoval: (labelId: string) => void;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [spectrogramOpen, setSpectrogramOpen] = useState(false);

  const hasUtc = row.start_utc != null && row.end_utc != null;
  const duration = row.end_sec - row.start_sec;

  // Fetch existing vocalization labels for this row
  const { data: existingLabels = [] } = useVocalizationLabels(
    detectionJobId,
    row.start_utc,
    row.end_utc,
  );

  const { data: labelVocab = [] } = useLabelVocabulary();

  // Merge vocabulary sources (exclude reserved labels)
  const allTypes = useMemo(() => {
    const set = new Set([...vocabulary, ...labelVocab]);
    set.delete(NEGATIVE_LABEL);
    return Array.from(set).sort();
  }, [vocabulary, labelVocab]);

  const existingTypeNames = new Set(existingLabels.map((l) => l.label));
  const pendingAddSet = pendingAdds ?? new Set<string>();
  const pendingRemovalSet = pendingRemovals ?? new Set<string>();

  const hasNegative =
    existingTypeNames.has(NEGATIVE_LABEL) || pendingAddSet.has(NEGATIVE_LABEL);
  const showNegativeOption =
    !hasNegative ||
    (existingTypeNames.has(NEGATIVE_LABEL) &&
      existingLabels.some(
        (l) => l.label === NEGATIVE_LABEL && pendingRemovalSet.has(l.id),
      ));

  // Inference-predicted tags above threshold (suggestions)
  const predictedTags = useMemo(() => {
    return Object.entries(row.scores)
      .filter(([type, score]) => score >= (thresholds[type] ?? 0.5))
      .sort(([, a], [, b]) => b - a);
  }, [row.scores, thresholds]);

  const predictedTypeNames = new Set(predictedTags.map(([t]) => t));

  // Available types to add (not already assigned or pending-add)
  const availableTypes = allTypes.filter(
    (t) =>
      !existingTypeNames.has(t) &&
      !pendingAddSet.has(t),
  );

  /** Add a label with mutual exclusivity enforcement */
  function handleAddWithExclusivity(label: string) {
    if (readonly) return;
    if (label === NEGATIVE_LABEL) {
      // Clear all pending type adds
      for (const t of pendingAddSet) {
        if (t !== NEGATIVE_LABEL) onRemovePendingAdd(t);
      }
      // Mark existing type labels for removal
      for (const lbl of existingLabels) {
        if (lbl.label !== NEGATIVE_LABEL && !pendingRemovalSet.has(lbl.id)) {
          onAddPendingRemoval(lbl.id);
        }
      }
      onAddPending(NEGATIVE_LABEL);
    } else {
      // Clear pending "(Negative)" add
      if (pendingAddSet.has(NEGATIVE_LABEL)) {
        onRemovePendingAdd(NEGATIVE_LABEL);
      }
      // Mark existing "(Negative)" for removal
      for (const lbl of existingLabels) {
        if (lbl.label === NEGATIVE_LABEL && !pendingRemovalSet.has(lbl.id)) {
          onAddPendingRemoval(lbl.id);
        }
      }
      onAddPending(label);
    }
  }

  // Max score for display
  const maxScore = Math.max(...Object.values(row.scores), 0);

  const spectrogramSrc = hasUtc && detectionJobId
    ? detectionSpectrogramUrl(detectionJobId, row.start_utc!, duration)
    : null;
  const audioSrc = hasUtc && detectionJobId
    ? detectionAudioSliceUrl(detectionJobId, row.start_utc!, duration)
    : null;

  function handleTogglePredicted(type: string) {
    if (readonly) return;
    if (pendingAddSet.has(type)) {
      onRemovePendingAdd(type);
    } else if (!existingTypeNames.has(type)) {
      handleAddWithExclusivity(type);
    }
  }

  function handleToggleSavedRemoval(label: VocalizationLabel) {
    if (readonly) return;
    if (pendingRemovalSet.has(label.id)) {
      onRemovePendingRemoval(label.id);
    } else {
      onAddPendingRemoval(label.id);
    }
  }

  function formatUtcShort(epoch: number): string {
    return (
      new Date(epoch * 1000)
        .toISOString()
        .replace("T", " ")
        .slice(0, 19) + " UTC"
    );
  }

  return (
    <>
      <div className="flex items-start gap-3 px-3 py-3">
        {/* Spectrogram — larger size, clickable */}
        {spectrogramSrc ? (
          <img
            src={spectrogramSrc}
            alt="spectrogram"
            className="w-[120px] h-[80px] object-cover rounded border shrink-0 cursor-pointer hover:ring-2 hover:ring-primary/50 transition-shadow"
            loading="lazy"
            onClick={() => setSpectrogramOpen(true)}
          />
        ) : (
          <div className="w-[120px] h-[80px] bg-muted rounded border flex items-center justify-center text-xs text-muted-foreground shrink-0">
            no preview
          </div>
        )}

        {/* Info + labels */}
        <div className="flex-1 min-w-0 space-y-2">
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

          {/* Vocalization labels — three visual states */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs text-muted-foreground">Voc:</span>

            {/* 1. Inference-suggested tags */}
            {predictedTags.map(([type, score]) => {
              const isSaved = existingTypeNames.has(type);
              const isPendingAdd = pendingAddSet.has(type);
              const savedLabel = existingLabels.find(
                (l) => l.label === type,
              );
              const isPendingRemoval =
                savedLabel && pendingRemovalSet.has(savedLabel.id);

              return (
                <Badge
                  key={`pred-${type}`}
                  variant="outline"
                  className={`text-xs ${typeColorMap.get(type) ?? ""} ${
                    isPendingAdd
                      ? "ring-2 ring-primary/60 opacity-100"
                      : isSaved && !isPendingRemoval
                        ? "opacity-100"
                        : isPendingRemoval
                          ? "opacity-30 line-through"
                          : "opacity-50"
                  } ${readonly ? "" : "cursor-pointer"}`}
                  title={
                    readonly
                      ? `Predicted: ${(score * 100).toFixed(0)}%`
                      : isPendingAdd
                        ? "Pending add — click to undo"
                        : isSaved
                          ? `Saved label — ${(score * 100).toFixed(0)}%`
                          : "Click to add as label"
                  }
                  onClick={() => {
                    if (readonly) return;
                    if (isSaved && savedLabel) {
                      handleToggleSavedRemoval(savedLabel);
                    } else {
                      handleTogglePredicted(type);
                    }
                  }}
                >
                  {isPendingAdd && (
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary mr-1" />
                  )}
                  {type} {(score * 100).toFixed(0)}%
                </Badge>
              );
            })}

            {/* 2. Saved labels not in predictions */}
            {existingLabels
              .filter((lbl) => !predictedTypeNames.has(lbl.label))
              .map((lbl) => {
                const isPendingRemoval = pendingRemovalSet.has(lbl.id);
                return (
                  <Badge
                    key={lbl.id}
                    variant="outline"
                    className={`text-xs ${typeColorMap.get(lbl.label) ?? ""} ${
                      isPendingRemoval ? "opacity-30 line-through" : ""
                    } ${readonly ? "" : "cursor-pointer"}`}
                    onClick={() => {
                      if (!readonly) handleToggleSavedRemoval(lbl);
                    }}
                  >
                    {lbl.label}
                    {!readonly && !isPendingRemoval && (
                      <X className="h-2.5 w-2.5 ml-0.5" />
                    )}
                  </Badge>
                );
              })}

            {/* 3. Pending adds not in predictions */}
            {[...pendingAddSet]
              .filter((t) => !predictedTypeNames.has(t))
              .map((t) => (
                <Badge
                  key={`pending-${t}`}
                  variant="outline"
                  className={`text-xs ${typeColorMap.get(t) ?? ""} ring-2 ring-primary/60 cursor-pointer`}
                  onClick={() => onRemovePendingAdd(t)}
                  title="Pending add — click to undo"
                >
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary mr-1" />
                  {t}
                </Badge>
              ))}

            {/* Add label popover */}
            {!readonly && (availableTypes.length > 0 || showNegativeOption) && (
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
                      onClick={() => handleAddWithExclusivity(t)}
                    >
                      {t}
                    </button>
                  ))}
                  {showNegativeOption && (
                    <>
                      {availableTypes.length > 0 && (
                        <div className="border-t my-1" />
                      )}
                      <button
                        className="w-full text-left text-xs px-2 py-1 hover:bg-red-50 rounded text-red-700"
                        onClick={() => handleAddWithExclusivity(NEGATIVE_LABEL)}
                      >
                        {NEGATIVE_LABEL}
                      </button>
                    </>
                  )}
                </PopoverContent>
              </Popover>
            )}
          </div>

          {/* Time range */}
          {hasUtc && (
            <div className="text-xs text-muted-foreground">
              {formatUtcShort(row.start_utc!)} —{" "}
              {formatUtcShort(row.end_utc!)}
            </div>
          )}
        </div>
      </div>

      {/* Spectrogram popup dialog */}
      <Dialog open={spectrogramOpen} onOpenChange={setSpectrogramOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-sm">Spectrogram Detail</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            {spectrogramSrc && (
              <img
                src={spectrogramSrc}
                alt="spectrogram detail"
                className="w-full rounded border"
              />
            )}
            {hasUtc && (
              <div className="text-sm text-muted-foreground">
                {formatUtcShort(row.start_utc!)} —{" "}
                {formatUtcShort(row.end_utc!)}
              </div>
            )}
            {audioSrc && (
              <audio controls src={audioSrc} className="w-full" preload="none" />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
