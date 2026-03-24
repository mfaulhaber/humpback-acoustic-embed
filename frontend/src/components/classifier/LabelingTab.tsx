import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchDetectionJobs,
  fetchHydrophoneDetectionJobs,
  fetchDetectionContent,
  detectionSpectrogramUrl,
  detectionAudioSliceUrl,
  audioWindowUrl,
} from "@/api/client";
import type { DetectionJob, DetectionRow, NeighborHit, PredictionRow } from "@/api/types";
import {
  useLabelingSummary,
  useVocalizationLabels,
  useAddVocalizationLabel,
  useDeleteVocalizationLabel,
  useLabelVocabulary,
  useDetectionNeighbors,
  useVocalizationModels,
  usePredictVocalizationLabels,
  useAnnotations,
  useCreateAnnotation,
  useDeleteAnnotation,
} from "@/hooks/queries/useLabeling";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import {
  ROOT_SENTINEL,
  buildFolderTree,
  makeToggleChild,
  makeToggleParent,
  makeToggleAll,
  makeToggleCollapse,
  EmbeddingSetPanel,
} from "@/components/shared/EmbeddingSetPanel";
import { ActiveLearningDashboard } from "./ActiveLearningDashboard";
import { VocalizationTrainingPanel } from "./VocalizationTrainingPanel";
import { AnnotationOverlay } from "./AnnotationOverlay";
import { AnnotationList } from "./AnnotationList";
import {
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Play,
  Pause,
  SkipForward,
  X,
  Volume2,
  PenTool,
} from "lucide-react";

type FilterMode = "all" | "unlabeled" | "labeled";
type SortMode = "confidence" | "time" | "uncertainty";

export function LabelingTab() {
  // ---- Job selection ----
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const localJobsQuery = useQuery({
    queryKey: ["detection-jobs"],
    queryFn: fetchDetectionJobs,
  });
  const hydroJobsQuery = useQuery({
    queryKey: ["hydrophone-detection-jobs"],
    queryFn: fetchHydrophoneDetectionJobs,
  });

  const allJobs = useMemo(() => {
    const local = localJobsQuery.data ?? [];
    const hydro = hydroJobsQuery.data ?? [];
    return [...local, ...hydro]
      .filter(
        (j) =>
          (j.status === "complete" || j.status === "canceled") &&
          j.detection_mode !== "merged",
      )
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      );
  }, [localJobsQuery.data, hydroJobsQuery.data]);

  // ---- Embedding set filter for neighbors ----
  const [esFilterSelected, setEsFilterSelected] = useState<Set<string>>(
    new Set(),
  );
  const [esFilterCollapsed, setEsFilterCollapsed] = useState<Set<string> | null>(
    null,
  );
  const [esFilterExpanded, setEsFilterExpanded] = useState(true);

  // ---- Detection content ----
  const contentQuery = useQuery({
    queryKey: ["detection-content", selectedJobId],
    queryFn: () => fetchDetectionContent(selectedJobId!),
    enabled: selectedJobId !== null,
  });

  // ---- Filter / sort / index ----
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [sortMode, setSortMode] = useState<SortMode>("confidence");
  const [currentIndex, setCurrentIndex] = useState(0);
  const summaryQuery = useLabelingSummary(selectedJobId);

  // ---- Vocalization classifier ----
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const vocModelsQuery = useVocalizationModels();
  const predictionsQuery = usePredictVocalizationLabels(
    selectedJobId,
    selectedModelId,
  );

  // ---- Embedding sets for neighbor filter ----
  const { data: embeddingSets = [] } = useEmbeddingSets();
  const { data: audioFiles = [] } = useAudioFiles();

  const audioMap = useMemo(
    () => new Map(audioFiles.map((af) => [af.id, af])),
    [audioFiles],
  );

  const esFilterTree = useMemo(
    () => buildFolderTree(embeddingSets, audioMap),
    [embeddingSets, audioMap],
  );

  const esFilterParentKeys = useMemo(
    () => new Set(esFilterTree.map((n) => n.parent)),
    [esFilterTree],
  );

  // Auto-select all embedding sets when the list loads
  const prevEsRef = useRef<string>("");
  useEffect(() => {
    const key = embeddingSets
      .map((es) => es.id)
      .sort()
      .join(",");
    if (key && key !== prevEsRef.current) {
      prevEsRef.current = key;
      setEsFilterSelected(new Set(embeddingSets.map((es) => es.id)));
    }
  }, [embeddingSets]);

  // Toggle callbacks for embedding set filter
  const toggleEsChild = useCallback(makeToggleChild(setEsFilterSelected), []);
  const toggleEsParent = useCallback(
    makeToggleParent(setEsFilterSelected),
    [],
  );
  const toggleEsAll = useCallback(
    makeToggleAll(embeddingSets, esFilterSelected, setEsFilterSelected),
    [embeddingSets, esFilterSelected],
  );
  const toggleEsCollapse = useCallback(
    makeToggleCollapse(
      esFilterParentKeys,
      esFilterCollapsed,
      setEsFilterCollapsed,
    ),
    [esFilterParentKeys],
  );
  const esDisplayName = useCallback(
    (key: string) => (key === ROOT_SENTINEL ? "(root)" : key),
    [],
  );

  // Index predictions by row_id for fast lookup
  const predictionsByRowId = useMemo(() => {
    const map = new Map<string, PredictionRow>();
    for (const p of predictionsQuery.data ?? []) {
      map.set(p.row_id, p);
    }
    return map;
  }, [predictionsQuery.data]);

  // Track which row_ids have vocalization labels (from summary)
  const labeledRowIds = useMemo(() => {
    if (!summaryQuery.data) return new Set<string>();
    // We'll use labels query per row, but for filter we need bulk info.
    // For now, we re-derive from vocalization labels queries.
    // TODO: optimize with bulk endpoint if needed.
    return new Set<string>();
  }, [summaryQuery.data]);

  const filteredRows = useMemo(() => {
    const rows = contentQuery.data ?? [];
    let filtered = rows;
    // Note: filtering by labeled/unlabeled requires knowing which rows have
    // vocalization labels. For Phase 1, we do filtering based on species labels
    // as a proxy since the summary query gives us counts but not specific row IDs.
    // Full vocalization-based filtering will come with bulk label loading.
    if (filterMode === "labeled") {
      filtered = filtered.filter(
        (r) =>
          r.humpback === 1 || r.orca === 1 || r.ship === 1 || r.background === 1,
      );
    } else if (filterMode === "unlabeled") {
      filtered = filtered.filter(
        (r) =>
          r.humpback !== 1 &&
          r.orca !== 1 &&
          r.ship !== 1 &&
          r.background !== 1,
      );
    }

    const sorted = [...filtered];
    if (sortMode === "confidence") {
      sorted.sort((a, b) => b.peak_confidence - a.peak_confidence);
    } else if (sortMode === "uncertainty") {
      // Sort by prediction uncertainty (least confident first)
      sorted.sort((a, b) => {
        const pa = predictionsByRowId.get(a.row_id ?? "");
        const pb = predictionsByRowId.get(b.row_id ?? "");
        const ca = pa?.confidence ?? 1;
        const cb = pb?.confidence ?? 1;
        return ca - cb; // ascending = least confident first
      });
    } else {
      sorted.sort((a, b) => a.start_sec - b.start_sec);
    }
    return sorted;
  }, [contentQuery.data, filterMode, sortMode, predictionsByRowId]);

  const currentRow = filteredRows[currentIndex] ?? null;

  // Reset index when job changes
  useEffect(() => {
    setCurrentIndex(0);
  }, [selectedJobId, filterMode, sortMode]);

  // ---- Audio playback ----
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioUrl = useMemo(() => {
    if (!selectedJobId || !currentRow) return null;
    const duration = currentRow.end_sec - currentRow.start_sec;
    return detectionAudioSliceUrl(
      selectedJobId,
      currentRow.filename,
      currentRow.start_sec,
      duration,
    );
  }, [selectedJobId, currentRow]);

  useEffect(() => {
    setIsPlaying(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [currentIndex, selectedJobId]);

  const togglePlay = useCallback(() => {
    const el = audioRef.current;
    if (!el || !audioUrl) return;
    if (isPlaying) {
      el.pause();
      setIsPlaying(false);
    } else {
      el.src = audioUrl;
      el.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [audioUrl, isPlaying]);

  // ---- Vocalization labels for current row ----
  const rowId = currentRow?.row_id ?? null;
  const vocLabelsQuery = useVocalizationLabels(selectedJobId, rowId);
  const addLabel = useAddVocalizationLabel();
  const removeLabel = useDeleteVocalizationLabel();
  const vocabularyQuery = useLabelVocabulary();
  const [labelInput, setLabelInput] = useState("");

  const handleAddLabel = useCallback(
    (labelText: string) => {
      if (!selectedJobId || !rowId || !labelText.trim()) return;
      addLabel.mutate({
        detectionJobId: selectedJobId,
        rowId,
        label: labelText.trim(),
      });
      setLabelInput("");
    },
    [selectedJobId, rowId, addLabel],
  );

  const handleRemoveLabel = useCallback(
    (labelId: string) => {
      if (!selectedJobId || !rowId) return;
      removeLabel.mutate({
        labelId,
        detectionJobId: selectedJobId,
        rowId,
      });
    },
    [selectedJobId, rowId, removeLabel],
  );

  // ---- Neighbors (vector search) ----
  // Only send embedding_set_ids when a subset is selected;
  // omit when all are selected so the backend searches everything (avoids URL length limits).
  const selectedEsIds = useMemo(() => {
    if (esFilterSelected.size === 0 || esFilterSelected.size === embeddingSets.length)
      return undefined;
    return [...esFilterSelected];
  }, [esFilterSelected, embeddingSets.length]);

  const neighborsQuery = useDetectionNeighbors(
    selectedJobId,
    currentRow?.filename ?? null,
    currentRow?.start_sec ?? null,
    currentRow?.end_sec ?? null,
    selectedEsIds,
  );

  // Neighbor audio playback
  const neighborAudioRef = useRef<HTMLAudioElement>(null);
  const [playingNeighborIdx, setPlayingNeighborIdx] = useState<number | null>(
    null,
  );

  const playNeighborAudio = useCallback(
    (hit: NeighborHit, idx: number) => {
      const el = neighborAudioRef.current;
      if (!el) return;
      if (playingNeighborIdx === idx) {
        el.pause();
        setPlayingNeighborIdx(null);
        return;
      }
      const url = audioWindowUrl(hit.audio_file_id, hit.window_offset_seconds, 5);
      el.src = url;
      el.load();
      setPlayingNeighborIdx(idx);
      el.play().catch(() => setPlayingNeighborIdx(null));
    },
    [playingNeighborIdx],
  );

  // ---- Annotation mode ----
  const [annotationMode, setAnnotationMode] = useState(false);
  const [highlightedAnnotationId, setHighlightedAnnotationId] = useState<
    string | null
  >(null);
  const [pendingRegion, setPendingRegion] = useState<{
    start: number;
    end: number;
  } | null>(null);
  const [annotationLabelInput, setAnnotationLabelInput] = useState("");

  const annotationsQuery = useAnnotations(selectedJobId, rowId);
  const createAnn = useCreateAnnotation();
  const deleteAnn = useDeleteAnnotation();

  const handleCreateRegion = useCallback(
    (startSec: number, endSec: number) => {
      setPendingRegion({ start: startSec, end: endSec });
      setAnnotationLabelInput("");
    },
    [],
  );

  const handleConfirmAnnotation = useCallback(() => {
    if (
      !selectedJobId ||
      !rowId ||
      !pendingRegion ||
      !annotationLabelInput.trim()
    )
      return;
    createAnn.mutate({
      detectionJobId: selectedJobId,
      rowId,
      start_offset_sec: pendingRegion.start,
      end_offset_sec: pendingRegion.end,
      label: annotationLabelInput.trim(),
    });
    setPendingRegion(null);
    setAnnotationLabelInput("");
  }, [selectedJobId, rowId, pendingRegion, annotationLabelInput, createAnn]);

  const handleDeleteAnnotation = useCallback(
    (annotationId: string) => {
      if (!selectedJobId || !rowId) return;
      deleteAnn.mutate({
        annotationId,
        detectionJobId: selectedJobId,
        rowId,
      });
      if (highlightedAnnotationId === annotationId) {
        setHighlightedAnnotationId(null);
      }
    },
    [selectedJobId, rowId, deleteAnn, highlightedAnnotationId],
  );

  // ---- Navigation helpers ----
  const goNext = useCallback(() => {
    setCurrentIndex((i) => Math.min(i + 1, filteredRows.length - 1));
  }, [filteredRows.length]);

  const goPrev = useCallback(() => {
    setCurrentIndex((i) => Math.max(i - 1, 0));
  }, []);

  const goNextUnlabeled = useCallback(() => {
    const startIdx = currentIndex + 1;
    for (let i = startIdx; i < filteredRows.length; i++) {
      const r = filteredRows[i];
      if (
        r.humpback !== 1 &&
        r.orca !== 1 &&
        r.ship !== 1 &&
        r.background !== 1
      ) {
        setCurrentIndex(i);
        return;
      }
    }
  }, [currentIndex, filteredRows]);

  // ---- Keyboard shortcuts ----
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (!selectedJobId || filteredRows.length === 0) return;

      if (e.key === "ArrowRight" || e.key === "j") {
        e.preventDefault();
        goNext();
        return;
      }
      if (e.key === "ArrowLeft" || e.key === "k") {
        e.preventDefault();
        goPrev();
        return;
      }
      if (e.key === " ") {
        e.preventDefault();
        togglePlay();
        return;
      }
      if (e.key === "u") {
        e.preventDefault();
        goNextUnlabeled();
        return;
      }
      if (e.key === "x") {
        e.preventDefault();
        // Clear all vocalization labels on current row
        const labels = vocLabelsQuery.data ?? [];
        for (const lbl of labels) {
          handleRemoveLabel(lbl.id);
        }
        return;
      }

      // Number keys 1-9: apply label from neighbor hits
      const num = parseInt(e.key);
      if (num >= 1 && num <= 9) {
        const hits = neighborsQuery.data?.hits ?? [];
        const hit = hits[num - 1];
        if (hit?.inferred_label) {
          e.preventDefault();
          handleAddLabel(hit.inferred_label);
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [
    selectedJobId,
    filteredRows,
    goNext,
    goPrev,
    togglePlay,
    goNextUnlabeled,
    vocLabelsQuery.data,
    handleRemoveLabel,
    handleAddLabel,
    neighborsQuery.data,
  ]);

  // ---- Spectrogram URL ----
  const spectrogramUrl = useMemo(() => {
    if (!selectedJobId || !currentRow) return null;
    const duration = currentRow.end_sec - currentRow.start_sec;
    return detectionSpectrogramUrl(
      selectedJobId,
      currentRow.filename,
      currentRow.start_sec,
      duration,
    );
  }, [selectedJobId, currentRow]);

  // ---- Job label for dropdown ----
  const jobLabel = useCallback((job: DetectionJob) => {
    const date = new Date(job.created_at).toLocaleDateString();
    const source = job.hydrophone_name
      ? job.hydrophone_name
      : job.audio_folder
        ? job.audio_folder.split("/").pop()
        : "local";
    return `${date} — ${source} (${job.status})`;
  }, []);

  // ---- Vocabulary suggestions for autocomplete ----
  const suggestions = useMemo(() => {
    const vocab = vocabularyQuery.data ?? [];
    if (!labelInput.trim()) return vocab.slice(0, 10);
    const q = labelInput.toLowerCase();
    return vocab.filter((v) => v.toLowerCase().includes(q)).slice(0, 10);
  }, [vocabularyQuery.data, labelInput]);

  // ---- Render ----
  if (!localJobsQuery.data && !hydroJobsQuery.data) {
    return (
      <div className="p-6 text-sm text-slate-500">Loading detection jobs...</div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-4 px-6 py-3 border-b bg-white">
        <select
          value={selectedJobId ?? ""}
          onChange={(e) => setSelectedJobId(e.target.value || null)}
          className="border rounded px-2 py-1.5 text-sm max-w-md"
        >
          <option value="">Select a detection job...</option>
          {allJobs.map((j) => (
            <option key={j.id} value={j.id}>
              {jobLabel(j)}
            </option>
          ))}
        </select>

        <select
          value={filterMode}
          onChange={(e) => setFilterMode(e.target.value as FilterMode)}
          className="border rounded px-2 py-1.5 text-sm"
        >
          <option value="all">All</option>
          <option value="unlabeled">Unlabeled</option>
          <option value="labeled">Labeled</option>
        </select>

        <select
          value={sortMode}
          onChange={(e) => setSortMode(e.target.value as SortMode)}
          className="border rounded px-2 py-1.5 text-sm"
        >
          <option value="confidence">Sort: Confidence</option>
          <option value="time">Sort: Time</option>
          {selectedModelId && (
            <option value="uncertainty">Sort: Uncertainty</option>
          )}
        </select>

        {/* Classifier selector */}
        {(vocModelsQuery.data ?? []).length > 0 && (
          <select
            value={selectedModelId ?? ""}
            onChange={(e) => setSelectedModelId(e.target.value || null)}
            className="border rounded px-2 py-1.5 text-sm"
          >
            <option value="">No classifier</option>
            {(vocModelsQuery.data ?? []).map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        )}

        {/* Progress bar */}
        {summaryQuery.data && (
          <div className="flex items-center gap-2 ml-auto text-sm text-slate-600">
            <span>
              {summaryQuery.data.labeled_rows}/{summaryQuery.data.total_rows}{" "}
              labeled
            </span>
            <div className="w-32 h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all"
                style={{
                  width: `${summaryQuery.data.total_rows > 0 ? (summaryQuery.data.labeled_rows / summaryQuery.data.total_rows) * 100 : 0}%`,
                }}
              />
            </div>
          </div>
        )}
      </div>

      {!selectedJobId ? (
        <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
          Select a detection job to start labeling
        </div>
      ) : !currentRow ? (
        <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
          {contentQuery.isLoading
            ? "Loading detections..."
            : "No detections match the current filter"}
        </div>
      ) : (
        <div className="flex-1 flex overflow-hidden">
          {/* Main panel (65%) */}
          <div className="flex-[65] flex flex-col p-4 overflow-y-auto border-r">
            {/* Navigation bar */}
            <div className="flex items-center justify-between mb-3">
              <button
                onClick={goPrev}
                disabled={currentIndex === 0}
                className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30"
                title="Previous (Left arrow / k)"
              >
                <ChevronLeft className="h-5 w-5" />
              </button>
              <span className="text-sm text-slate-600 font-mono">
                {currentIndex + 1} / {filteredRows.length}
              </span>
              <button
                onClick={goNext}
                disabled={currentIndex >= filteredRows.length - 1}
                className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30"
                title="Next (Right arrow / j)"
              >
                <ChevronRight className="h-5 w-5" />
              </button>
            </div>

            {/* Spectrogram + annotation overlay */}
            {spectrogramUrl && (
              <div className="mb-3 bg-slate-900 rounded overflow-hidden relative">
                <img
                  src={spectrogramUrl}
                  alt="Detection spectrogram"
                  className="w-full h-auto"
                />
                {annotationMode && currentRow && (
                  <AnnotationOverlay
                    windowDuration={currentRow.end_sec - currentRow.start_sec}
                    annotations={annotationsQuery.data ?? []}
                    highlightedId={highlightedAnnotationId}
                    onCreateRegion={handleCreateRegion}
                    onDeleteAnnotation={handleDeleteAnnotation}
                    onSelectAnnotation={setHighlightedAnnotationId}
                  />
                )}
              </div>
            )}

            {/* Pending annotation label input */}
            {pendingRegion && (
              <div className="mb-3 flex items-center gap-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm">
                <span className="text-xs text-slate-600 font-mono">
                  {pendingRegion.start.toFixed(2)}s &ndash;{" "}
                  {pendingRegion.end.toFixed(2)}s
                </span>
                <input
                  type="text"
                  value={annotationLabelInput}
                  onChange={(e) => setAnnotationLabelInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleConfirmAnnotation();
                    if (e.key === "Escape") setPendingRegion(null);
                  }}
                  placeholder="Label (e.g. whup, moan)..."
                  className="flex-1 border rounded px-2 py-1 text-sm"
                  autoFocus
                />
                <button
                  onClick={handleConfirmAnnotation}
                  disabled={!annotationLabelInput.trim()}
                  className="px-2 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600 disabled:opacity-50"
                >
                  Save
                </button>
                <button
                  onClick={() => setPendingRegion(null)}
                  className="px-2 py-1 bg-slate-200 rounded text-xs hover:bg-slate-300"
                >
                  Cancel
                </button>
              </div>
            )}

            {/* Audio controls */}
            <div className="flex items-center gap-3 mb-3">
              <button
                onClick={togglePlay}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded text-sm"
                title="Play/Pause (Space)"
              >
                {isPlaying ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {isPlaying ? "Pause" : "Play"}
              </button>
              <audio
                ref={audioRef}
                onEnded={() => setIsPlaying(false)}
                onPause={() => setIsPlaying(false)}
              />

              <button
                onClick={goNextUnlabeled}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded text-sm"
                title="Next unlabeled (u)"
              >
                <SkipForward className="h-4 w-4" />
                Next unlabeled
              </button>

              <button
                onClick={() => setAnnotationMode((m) => !m)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm ${
                  annotationMode
                    ? "bg-blue-500 text-white hover:bg-blue-600"
                    : "bg-slate-100 hover:bg-slate-200"
                }`}
                title="Toggle annotation mode"
              >
                <PenTool className="h-4 w-4" />
                {annotationMode ? "Annotating" : "Annotate"}
              </button>
            </div>

            {/* Detection metadata */}
            <div className="flex flex-wrap gap-4 text-xs text-slate-500 mb-3">
              <span>
                <strong>File:</strong> {currentRow.filename}
              </span>
              <span>
                <strong>Range:</strong> {currentRow.start_sec.toFixed(1)}s &ndash;{" "}
                {currentRow.end_sec.toFixed(1)}s
              </span>
              <span>
                <strong>Avg conf:</strong>{" "}
                {currentRow.avg_confidence.toFixed(3)}
              </span>
              <span>
                <strong>Peak conf:</strong>{" "}
                {currentRow.peak_confidence.toFixed(3)}
              </span>
            </div>

            {/* Species labels (read-only) */}
            <div className="flex gap-2 mb-3">
              {currentRow.humpback === 1 && (
                <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                  humpback
                </span>
              )}
              {currentRow.orca === 1 && (
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs font-medium">
                  orca
                </span>
              )}
              {currentRow.ship === 1 && (
                <span className="px-2 py-0.5 bg-orange-100 text-orange-700 rounded text-xs font-medium">
                  ship
                </span>
              )}
              {currentRow.background === 1 && (
                <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs font-medium">
                  background
                </span>
              )}
            </div>

            {/* Classifier prediction */}
            {selectedModelId && currentRow?.row_id && (() => {
              const pred = predictionsByRowId.get(currentRow.row_id ?? "");
              if (!pred) return null;
              const sortedProbs = Object.entries(pred.probabilities).sort(
                ([, a], [, b]) => b - a,
              );
              return (
                <div className="border rounded p-3 bg-amber-50 mb-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs text-slate-500 font-medium">
                      Classifier Prediction
                    </div>
                    <button
                      onClick={() => handleAddLabel(pred.predicted_label)}
                      className="text-xs px-2 py-0.5 bg-amber-100 hover:bg-amber-200 text-amber-800 rounded font-medium"
                    >
                      Accept &ldquo;{pred.predicted_label}&rdquo;
                    </button>
                  </div>
                  <div className="space-y-1">
                    {sortedProbs.map(([name, prob]) => (
                      <div key={name} className="flex items-center gap-2 text-xs">
                        <span className="w-20 text-slate-600 truncate">{name}</span>
                        <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              name === pred.predicted_label
                                ? "bg-amber-500"
                                : "bg-slate-300"
                            }`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="w-10 text-right text-slate-500 font-mono">
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Vocalization type labels */}
            <div className="border rounded p-3 bg-white">
              <div className="text-xs text-slate-500 font-medium mb-2">
                Vocalization Type Labels
              </div>
              <div className="flex flex-wrap gap-1.5 mb-2">
                {(vocLabelsQuery.data ?? []).map((vl) => (
                  <span
                    key={vl.id}
                    className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded text-xs font-medium"
                  >
                    {vl.label}
                    <button
                      onClick={() => handleRemoveLabel(vl.id)}
                      className="hover:text-red-500"
                      title="Remove label"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                ))}
                {(vocLabelsQuery.data ?? []).length === 0 && (
                  <span className="text-xs text-slate-400 italic">
                    No vocalization labels yet
                  </span>
                )}
              </div>
              <div className="relative">
                <input
                  type="text"
                  value={labelInput}
                  onChange={(e) => setLabelInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && labelInput.trim()) {
                      handleAddLabel(labelInput);
                    }
                  }}
                  placeholder="Type to add label (e.g. whup, moan)..."
                  className="w-full border rounded px-2 py-1 text-sm"
                />
                {labelInput.trim() && suggestions.length > 0 && (
                  <div className="absolute left-0 right-0 top-full mt-1 bg-white border rounded shadow-lg z-10 max-h-40 overflow-y-auto">
                    {suggestions.map((s) => (
                      <button
                        key={s}
                        onClick={() => handleAddLabel(s)}
                        className="block w-full text-left px-3 py-1.5 text-sm hover:bg-slate-50"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Annotation list (when annotation mode is active) */}
            {annotationMode && (
              <div className="border rounded p-3 bg-white">
                <div className="text-xs text-slate-500 font-medium mb-2">
                  Annotations
                </div>
                <AnnotationList
                  annotations={annotationsQuery.data ?? []}
                  highlightedId={highlightedAnnotationId}
                  onSelect={setHighlightedAnnotationId}
                  onDelete={handleDeleteAnnotation}
                />
              </div>
            )}
          </div>

          {/* Similar sounds panel (35%) */}
          <div className="flex-[35] flex flex-col p-4 overflow-y-auto bg-slate-50">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-medium text-slate-700">
                Similar Sounds
              </div>
              <button
                onClick={() => setEsFilterExpanded((v) => !v)}
                className="text-xs text-slate-500 hover:text-slate-700 flex items-center gap-1"
              >
                {esFilterExpanded ? (
                  <ChevronUp className="h-3.5 w-3.5" />
                ) : (
                  <ChevronDown className="h-3.5 w-3.5" />
                )}
                {esFilterSelected.size}/{embeddingSets.length} sets
              </button>
            </div>

            {esFilterExpanded && (
              <div className="mb-3">
                <EmbeddingSetPanel
                  label="Search in"
                  selected={esFilterSelected}
                  collapsed={esFilterCollapsed ?? esFilterParentKeys}
                  folderTree={esFilterTree}
                  embeddingSets={embeddingSets}
                  onToggleChild={toggleEsChild}
                  onToggleParent={toggleEsParent}
                  onToggleAll={toggleEsAll}
                  onToggleCollapse={toggleEsCollapse}
                  displayName={esDisplayName}
                />
              </div>
            )}

            <audio ref={neighborAudioRef} onEnded={() => setPlayingNeighborIdx(null)} />

            {neighborsQuery.isLoading && (
              <div className="text-xs text-slate-400">Searching...</div>
            )}
            {neighborsQuery.error && (
              <div className="text-xs text-red-500">
                {(neighborsQuery.error as Error).message}
              </div>
            )}
            {neighborsQuery.data?.hits.length === 0 && (
              <div className="text-xs text-slate-400">
                No similar sounds found
              </div>
            )}

            <div className="space-y-2">
              {(neighborsQuery.data?.hits ?? []).map((hit, idx) => (
                <div
                  key={`${hit.embedding_set_id}-${hit.row_index}`}
                  className="border rounded p-2 bg-white"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono text-slate-500">
                      #{idx + 1}
                    </span>
                    <span
                      className={`text-xs font-medium ${
                        hit.score > 0.8
                          ? "text-emerald-600"
                          : hit.score > 0.6
                            ? "text-amber-600"
                            : "text-slate-500"
                      }`}
                    >
                      {(hit.score * 100).toFixed(1)}%
                    </span>
                  </div>

                  {hit.inferred_label && (
                    <div className="mb-1">
                      <span className="px-1.5 py-0.5 bg-violet-100 text-violet-700 rounded text-xs font-medium">
                        {hit.inferred_label}
                      </span>
                    </div>
                  )}

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => playNeighborAudio(hit, idx)}
                      className="p-1 rounded hover:bg-slate-100"
                      title="Play"
                    >
                      {playingNeighborIdx === idx ? (
                        <Pause className="h-3.5 w-3.5" />
                      ) : (
                        <Volume2 className="h-3.5 w-3.5" />
                      )}
                    </button>
                    {hit.inferred_label && (
                      <button
                        onClick={() => handleAddLabel(hit.inferred_label!)}
                        className="text-xs px-2 py-0.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 rounded"
                        title={`Apply "${hit.inferred_label}" label`}
                      >
                        Apply
                      </button>
                    )}
                  </div>

                  <div
                    className="text-[10px] text-slate-400 mt-1 truncate"
                    title={hit.audio_folder_path ?? ""}
                  >
                    {hit.audio_filename}
                  </div>
                </div>
              ))}
            </div>

            {neighborsQuery.data &&
              neighborsQuery.data.hits.length > 0 && (
                <div className="mt-3 text-[10px] text-slate-400">
                  Searched {neighborsQuery.data.total_candidates.toLocaleString()}{" "}
                  embeddings
                </div>
              )}

            {/* Vocalization classifier training */}
            <div className="mt-4 pt-4 border-t">
              <VocalizationTrainingPanel
                onModelReady={(modelId) => {
                  setSelectedModelId(modelId);
                  vocModelsQuery.refetch();
                }}
              />
            </div>

            {/* Active Learning Dashboard */}
            {selectedModelId && selectedJobId && (
              <div className="mt-4 pt-4 border-t">
                <div className="text-sm font-medium text-slate-700 mb-3">
                  Active Learning
                </div>
                <ActiveLearningDashboard
                  selectedModelId={selectedModelId}
                  selectedJobId={selectedJobId}
                  onCycleStarted={() => {}}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Keyboard shortcuts help */}
      <div className="px-6 py-1.5 border-t bg-slate-50 text-[10px] text-slate-400 flex gap-4">
        <span>
          <kbd className="font-mono bg-white border rounded px-1">j</kbd>/
          <kbd className="font-mono bg-white border rounded px-1">&rarr;</kbd>{" "}
          Next
        </span>
        <span>
          <kbd className="font-mono bg-white border rounded px-1">k</kbd>/
          <kbd className="font-mono bg-white border rounded px-1">&larr;</kbd>{" "}
          Prev
        </span>
        <span>
          <kbd className="font-mono bg-white border rounded px-1">Space</kbd>{" "}
          Play
        </span>
        <span>
          <kbd className="font-mono bg-white border rounded px-1">u</kbd> Next
          unlabeled
        </span>
        <span>
          <kbd className="font-mono bg-white border rounded px-1">1</kbd>-
          <kbd className="font-mono bg-white border rounded px-1">9</kbd> Apply
          neighbor label
        </span>
        <span>
          <kbd className="font-mono bg-white border rounded px-1">x</kbd> Clear
          labels
        </span>
      </div>
    </div>
  );
}
