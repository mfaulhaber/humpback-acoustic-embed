import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchDetectionJobs,
  fetchHydrophoneDetectionJobs,
  fetchDetectionContent,
  detectionSpectrogramUrl,
  detectionAudioSliceUrl,
  audioWindowUrl,
  audioSpectrogramPngUrl,
} from "@/api/client";
import type { DetectionJob, DetectionRow, NeighborHit } from "@/api/types";
import {
  useLabelingSummary,
  useVocalizationLabels,
  useAddVocalizationLabel,
  useDeleteVocalizationLabel,
  useLabelVocabulary,
  useDetectionNeighbors,
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
import { SpectrogramPopup } from "./SpectrogramPopup";
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
} from "lucide-react";

type FilterMode = "all" | "unlabeled" | "labeled";
type SortMode = "confidence" | "time";

function normalizeLabel(raw: string): string {
  return raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatUtcDateTime(timestampSeconds: number): string {
  const date = new Date(timestampSeconds * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${date.getUTCFullYear()}-${p(date.getUTCMonth() + 1)}-${p(date.getUTCDate())} ${p(date.getUTCHours())}:${p(date.getUTCMinutes())} UTC`;
}

function formatUtcDateRange(startTs: number, endTs: number): string {
  return `${formatUtcDateTime(startTs)} - ${formatUtcDateTime(endTs)}`;
}

function formatCreatedAtUtc(createdAtIso: string): string {
  const date = new Date(createdAtIso);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${date.getUTCFullYear()}-${p(date.getUTCMonth() + 1)}-${p(date.getUTCDate())} ${p(date.getUTCHours())}:${p(date.getUTCMinutes())} UTC`;
}


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

  const selectedJob = useMemo(
    () => allJobs.find((job) => job.id === selectedJobId) ?? null,
    [allJobs, selectedJobId],
  );

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

  // Index predictions by UTC key for fast lookup
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
      sorted.sort((a, b) => (b.peak_confidence ?? 0) - (a.peak_confidence ?? 0));
    } else {
      sorted.sort((a, b) => a.start_utc - b.start_utc);
    }
    return sorted;
  }, [contentQuery.data, filterMode, sortMode]);

  const currentRow = filteredRows[currentIndex] ?? null;

  // Reset index when job changes
  useEffect(() => {
    setCurrentIndex(0);
  }, [selectedJobId, filterMode, sortMode]);

  // ---- Audio playback ----
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const currentClip = useMemo(() => {
    if (!currentRow) return null;
    return {
      startUtc: currentRow.start_utc,
      endUtc: currentRow.end_utc,
      durationSec: Math.max(0.1, currentRow.end_utc - currentRow.start_utc),
    };
  }, [currentRow]);

  const audioUrl = useMemo(() => {
    if (!selectedJobId || !currentClip) return null;
    return detectionAudioSliceUrl(
      selectedJobId,
      currentClip.startUtc,
      currentClip.durationSec,
    );
  }, [selectedJobId, currentClip]);

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
  const currentRowId = currentRow?.row_id ?? null;
  const vocLabelsQuery = useVocalizationLabels(selectedJobId, currentRowId);
  const addLabel = useAddVocalizationLabel();
  const removeLabel = useDeleteVocalizationLabel();
  const vocabularyQuery = useLabelVocabulary();
  const [labelInput, setLabelInput] = useState("");

  const handleAddLabel = useCallback(
    (labelText: string) => {
      if (!selectedJobId || !currentRowId || !labelText.trim()) return;
      addLabel.mutate({
        detectionJobId: selectedJobId,
        rowId: currentRowId,
        label: labelText.trim(),
      });
      setLabelInput("");
    },
    [selectedJobId, currentRowId, addLabel],
  );

  const handleRemoveLabel = useCallback(
    (labelId: string) => {
      if (!selectedJobId || !currentRowId) return;
      removeLabel.mutate({
        labelId,
        detectionJobId: selectedJobId,
        rowId: currentRowId,
      });
    },
    [selectedJobId, currentRowId, removeLabel],
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
    currentRowId,
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

  // Neighbor spectrogram popup
  const [spectrogramPopup, setSpectrogramPopup] = useState<{
    imageUrl: string;
    position: { x: number; y: number };
    durationSec: number;
  } | null>(null);

  const handleNeighborSpectrogramClick = useCallback(
    (imageUrl: string, durationSec: number, e: React.MouseEvent) => {
      setSpectrogramPopup({
        imageUrl,
        position: { x: e.clientX, y: e.clientY },
        durationSec,
      });
    },
    [],
  );

  // Unique normalized inferred labels from neighbor hits for annotation dropdown
  const inferredLabelOptions = useMemo(() => {
    const hits = neighborsQuery.data?.hits ?? [];
    const seen = new Set<string>();
    const options: { value: string; raw: string }[] = [];
    for (const hit of hits) {
      if (hit.inferred_label && !seen.has(hit.inferred_label)) {
        seen.add(hit.inferred_label);
        options.push({ value: normalizeLabel(hit.inferred_label), raw: hit.inferred_label });
      }
    }
    return options;
  }, [neighborsQuery.data]);

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
    if (!selectedJobId || !currentClip) return null;
    return detectionSpectrogramUrl(
      selectedJobId,
      currentClip.startUtc,
      currentClip.durationSec,
    );
  }, [selectedJobId, currentClip]);

  const currentClipDuration = useMemo(() => {
    return currentClip?.durationSec ?? 0;
  }, [currentClip]);

  const currentDetectionName = useMemo(() => {
    if (!currentRow) return null;
    // Format UTC range as compact detection name
    const d = (epoch: number) => {
      const dt = new Date(epoch * 1000);
      const p = (n: number) => String(n).padStart(2, "0");
      return `${dt.getUTCFullYear()}${p(dt.getUTCMonth() + 1)}${p(dt.getUTCDate())}T${p(dt.getUTCHours())}${p(dt.getUTCMinutes())}${p(dt.getUTCSeconds())}Z`;
    };
    return `${d(currentRow.start_utc)}_${d(currentRow.end_utc)}`;
  }, [currentRow]);

  // ---- Job label for dropdown ----
  const jobLabel = useCallback((job: DetectionJob) => {
    const source = job.hydrophone_name
      ? job.hydrophone_name
      : job.audio_folder
        ? job.audio_folder.split("/").pop()
        : "local";
    const rangeLabel =
      job.start_timestamp !== null && job.end_timestamp !== null
        ? formatUtcDateRange(job.start_timestamp, job.end_timestamp)
        : formatCreatedAtUtc(job.created_at);
    return `${source} - ${rangeLabel}`;
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
        </select>

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
            </div>

            {/* Detection metadata */}
            <div className="flex flex-wrap gap-4 text-xs text-slate-500 mb-3">
              <span>
                <strong>Detection:</strong> {currentDetectionName}
              </span>
              <span>
                <strong>Confidence:</strong>{" "}
                {currentRow.peak_confidence != null
                  ? currentRow.peak_confidence.toFixed(3)
                  : "—"}
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
              {(neighborsQuery.data?.hits ?? []).map((hit, idx) => {
                const specUrl = audioSpectrogramPngUrl(hit.audio_file_id, hit.window_offset_seconds, 5);
                return (
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

                  <img
                    src={specUrl}
                    alt="spectrogram"
                    className="w-full h-[50px] object-cover rounded cursor-pointer border mb-1"
                    loading="lazy"
                    onClick={(e) => handleNeighborSpectrogramClick(specUrl, 5, e)}
                  />

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
                );
              })}
            </div>

            {neighborsQuery.data &&
              neighborsQuery.data.hits.length > 0 && (
                <div className="mt-3 text-[10px] text-slate-400">
                  Searched {neighborsQuery.data.total_candidates.toLocaleString()}{" "}
                  embeddings
                </div>
              )}

          </div>
        </div>
      )}

      {spectrogramPopup && (
        <SpectrogramPopup
          imageUrl={spectrogramPopup.imageUrl}
          position={spectrogramPopup.position}
          durationSec={spectrogramPopup.durationSec}
          onClose={() => setSpectrogramPopup(null)}
        />
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
