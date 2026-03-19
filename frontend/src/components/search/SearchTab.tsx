import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import { useLocation } from "react-router-dom";
import {
  Search,
  Play,
  Pause,
  Loader2,
  ChevronLeft,
  ChevronRight,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useSearchSimilar } from "@/hooks/queries/useSearch";
import {
  audioWindowUrl,
  audioSpectrogramPngUrl,
  detectionAudioSliceUrl,
  detectionSpectrogramUrl,
  searchSimilarByVector,
} from "@/api/client";
import type { EmbeddingSet, SimilaritySearchResponse } from "@/api/types";
import { SpectrogramPopup } from "@/components/classifier/SpectrogramPopup";
import {
  ROOT_SENTINEL,
  buildFolderTree,
  makeToggleChild,
  makeToggleParent,
  makeToggleAll,
  makeToggleCollapse,
  EmbeddingSetPanel,
  EmbeddingQueryPanel,
} from "@/components/shared/EmbeddingSetPanel";
import type { ParentNode } from "@/components/shared/EmbeddingSetPanel";

interface DetectionSearchState {
  source: "detection";
  detectionJobId: string;
  filename: string;
  startSec: number;
  endSec: number;
  clipDuration: number;
}

function isDetectionState(state: unknown): state is DetectionSearchState {
  return (
    typeof state === "object" &&
    state !== null &&
    (state as Record<string, unknown>).source === "detection"
  );
}

export function SearchTab() {
  const location = useLocation();
  const detectionState = isDetectionState(location.state) ? location.state : null;

  const [mode, setMode] = useState<"standalone" | "detection">(
    detectionState ? "detection" : "standalone",
  );

  // Standalone query state
  const [selectedModelVersion, setSelectedModelVersion] = useState<string>("");
  const [selectedEsId, setSelectedEsId] = useState<string>("");
  const [windowIndex, setWindowIndex] = useState(0);
  const [topK, setTopK] = useState(20);
  const [metric, setMetric] = useState<"cosine" | "euclidean">("cosine");
  const [searchTrigger, setSearchTrigger] = useState<{
    embedding_set_id: string;
    row_index: number;
    top_k: number;
    metric: string;
    exclude_self: boolean;
    embedding_set_ids?: string[];
  } | null>(null);

  // Detection audio search state
  const [audioSearchLoading, setAudioSearchLoading] = useState(false);
  const [audioSearchResults, setAudioSearchResults] = useState<SimilaritySearchResponse | null>(null);
  const [audioSearchError, setAudioSearchError] = useState<string | null>(null);
  const [detectionQueryVector, setDetectionQueryVector] = useState<number[] | null>(null);
  const [detectionModelVersion, setDetectionModelVersion] = useState<string | null>(null);
  const [reSearchLoading, setReSearchLoading] = useState(false);

  // Embedding set filter state (shared by both modes)
  const [filterSelected, setFilterSelected] = useState<Set<string>>(new Set());
  const [filterCollapsed, setFilterCollapsed] = useState<Set<string> | null>(null);

  // Standalone query panel collapse state
  const [queryCollapsed, setQueryCollapsed] = useState<Set<string> | null>(null);

  // Playback
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

  // Spectrogram popup
  const [spectrogramPopup, setSpectrogramPopup] = useState<{
    imageUrl: string;
    position: { x: number; y: number };
    durationSec: number;
  } | null>(null);

  // Data hooks
  const { data: embeddingSets = [] } = useEmbeddingSets();
  const { data: audioFiles = [] } = useAudioFiles();
  const standaloneResult = useSearchSimilar(searchTrigger);

  // Audio map for folder tree
  const audioMap = useMemo(
    () => new Map(audioFiles.map((af) => [af.id, af])),
    [audioFiles],
  );

  // Model versions from embedding sets
  const modelVersions = useMemo(() => {
    const versions = new Set(embeddingSets.map((es) => es.model_version));
    return Array.from(versions).sort();
  }, [embeddingSets]);

  // Determine active model version for filtering
  const activeModelVersion = mode === "detection"
    ? detectionModelVersion
    : selectedModelVersion && selectedModelVersion !== "__all__"
      ? selectedModelVersion
      : null;

  // Filter embedding sets by active model version
  const filteredSets = useMemo(() => {
    if (!activeModelVersion) return embeddingSets;
    return embeddingSets.filter((es) => es.model_version === activeModelVersion);
  }, [embeddingSets, activeModelVersion]);

  // Folder tree for filter panel
  const filterTree = useMemo(
    () => buildFolderTree(filteredSets, audioMap),
    [filteredSets, audioMap],
  );

  const filterParentKeys = useMemo(
    () => new Set(filterTree.map((n) => n.parent)),
    [filterTree],
  );

  // Folder tree for standalone query panel
  const queryTree = useMemo(
    () => buildFolderTree(filteredSets, audioMap),
    [filteredSets, audioMap],
  );

  // All collapsible keys for query panel: parent keys + child composite keys
  const queryAllKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const node of queryTree) {
      keys.add(node.parent);
      for (const child of node.children) {
        keys.add(`${node.parent}/${child.child}`);
      }
    }
    return keys;
  }, [queryTree]);

  // Toggle helpers for filter panel
  const toggleFilterChild = useCallback(makeToggleChild(setFilterSelected), []);
  const toggleFilterParent = useCallback(makeToggleParent(setFilterSelected), []);
  const toggleFilterAll = useCallback(
    makeToggleAll(filteredSets, filterSelected, setFilterSelected),
    [filteredSets, filterSelected],
  );
  const toggleFilterCollapse = useCallback(
    makeToggleCollapse(filterParentKeys, filterCollapsed, setFilterCollapsed),
    [filterParentKeys],
  );

  // Toggle collapse for query panel
  const toggleQueryCollapse = useCallback(
    makeToggleCollapse(queryAllKeys, queryCollapsed, setQueryCollapsed),
    [queryAllKeys],
  );

  // Display name helper
  const displayName = useCallback(
    (key: string) => (key === ROOT_SENTINEL ? "(root)" : key),
    [],
  );

  // Selected query embedding set
  const selectedEs = useMemo(
    () => embeddingSets.find((es) => es.id === selectedEsId) ?? null,
    [embeddingSets, selectedEsId],
  );

  // Max window index for selected embedding set
  const maxWindowIndex = useMemo(() => {
    if (!selectedEs) return 0;
    return 9999;
  }, [selectedEs]);

  // Initialize filter selection: select all when filteredSets changes
  // (for detection mode after model version is known, or when model filter changes)
  const prevFilteredRef = useRef<string>("");
  useEffect(() => {
    const key = filteredSets.map((es) => es.id).sort().join(",");
    if (key && key !== prevFilteredRef.current) {
      prevFilteredRef.current = key;
      setFilterSelected(new Set(filteredSets.map((es) => es.id)));
    }
  }, [filteredSets]);

  // Auto-trigger audio search when entering detection mode.
  useEffect(() => {
    if (mode !== "detection" || !detectionState || audioSearchResults) return;

    const controller = new AbortController();
    setAudioSearchLoading(true);
    setAudioSearchError(null);

    (async () => {
      try {
        const createRes = await fetch("/search/similar-by-audio", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            detection_job_id: detectionState.detectionJobId,
            filename: detectionState.filename,
            start_sec: detectionState.startSec,
            end_sec: detectionState.endSec,
            top_k: topK,
            metric,
          }),
          signal: controller.signal,
        });
        if (!createRes.ok) {
          const text = await createRes.text();
          throw new Error(`Failed to create search job: ${text}`);
        }
        const { id: jobId } = await createRes.json();

        while (!controller.signal.aborted) {
          await new Promise((r) => setTimeout(r, 500));
          if (controller.signal.aborted) break;

          const pollRes = await fetch(`/search/jobs/${jobId}`, {
            signal: controller.signal,
          });
          if (!pollRes.ok) {
            if (pollRes.status === 404) continue;
            const text = await pollRes.text();
            throw new Error(`Poll failed: ${text}`);
          }
          const job = await pollRes.json();

          if (job.status === "complete" && job.results) {
            setAudioSearchResults(job.results);
            setAudioSearchLoading(false);
            // Store vector for re-search
            if (job.query_vector) {
              setDetectionQueryVector(job.query_vector);
            }
            if (job.model_version) {
              setDetectionModelVersion(job.model_version);
            }
            return;
          }
          if (job.status === "failed") {
            throw new Error(job.error || "Search failed");
          }
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setAudioSearchError(err instanceof Error ? err.message : "Search failed");
        setAudioSearchLoading(false);
      }
    })();

    return () => controller.abort();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, detectionState]);

  // Active results
  const activeResults = mode === "detection" ? audioSearchResults : standaloneResult.data;
  const isSearching = mode === "detection" ? (audioSearchLoading || reSearchLoading) : standaloneResult.isFetching;
  const searchError = mode === "detection"
    ? (audioSearchError ? new Error(audioSearchError) : null)
    : standaloneResult.error;
  const results = activeResults?.results ?? [];
  const totalCandidates = activeResults?.total_candidates ?? 0;
  const resultModelVersion = activeResults?.model_version ?? "";

  // Find window_size_seconds for result rows
  const esMap = useMemo(() => {
    const map = new Map<string, EmbeddingSet>();
    embeddingSets.forEach((es) => map.set(es.id, es));
    return map;
  }, [embeddingSets]);

  const handleStandaloneSearch = useCallback(() => {
    if (!selectedEsId) return;
    const esIds = filterSelected.size > 0 && filterSelected.size < filteredSets.length
      ? [...filterSelected]
      : undefined;
    setSearchTrigger({
      embedding_set_id: selectedEsId,
      row_index: windowIndex,
      top_k: topK,
      metric,
      exclude_self: true,
      embedding_set_ids: esIds,
    });
  }, [selectedEsId, windowIndex, topK, metric, filterSelected, filteredSets.length]);

  const handleDetectionReSearch = useCallback(async () => {
    if (!detectionQueryVector || !detectionModelVersion) return;
    setReSearchLoading(true);
    setAudioSearchError(null);
    try {
      const esIds = filterSelected.size > 0 && filterSelected.size < filteredSets.length
        ? [...filterSelected]
        : undefined;
      const results = await searchSimilarByVector({
        vector: detectionQueryVector,
        model_version: detectionModelVersion,
        top_k: topK,
        metric,
        embedding_set_ids: esIds,
      });
      setAudioSearchResults(results);
    } catch (err: unknown) {
      setAudioSearchError(err instanceof Error ? err.message : "Re-search failed");
    } finally {
      setReSearchLoading(false);
    }
  }, [detectionQueryVector, detectionModelVersion, topK, metric, filterSelected, filteredSets.length]);

  const handlePlay = useCallback(
    (key: string, url: string) => {
      const audio = audioRef.current;
      if (!audio) return;
      if (playingKey === key) {
        audio.pause();
        setPlayingKey(null);
        return;
      }
      audio.src = url;
      audio.load();
      setPlayingKey(key);
      audio.play().catch(() => setPlayingKey(null));
    },
    [playingKey],
  );

  const handleSpectrogramClick = useCallback(
    (imageUrl: string, durationSec: number, e: React.MouseEvent) => {
      setSpectrogramPopup({
        imageUrl,
        position: { x: e.clientX, y: e.clientY },
        durationSec,
      });
    },
    [],
  );

  const scoreColor = (score: number) => {
    const pct = score * 100;
    if (pct >= 80) return "text-green-700 font-semibold";
    if (pct >= 60) return "text-yellow-700";
    return "text-muted-foreground";
  };

  // Handle query ES selection in standalone mode
  const handleQueryEsSelect = useCallback((esId: string) => {
    setSelectedEsId(esId);
    setWindowIndex(0);
    // Auto-set model version from selected set
    const es = embeddingSets.find((e) => e.id === esId);
    if (es && selectedModelVersion !== es.model_version) {
      setSelectedModelVersion(es.model_version);
    }
  }, [embeddingSets, selectedModelVersion]);

  return (
    <div className="space-y-4">
      <audio ref={audioRef} onEnded={() => setPlayingKey(null)} />

      {/* Query Card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Search className="h-4 w-4" />
            {mode === "detection" ? "Detection Search" : "Embedding Search"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {mode === "detection" && detectionState ? (
            <>
              <DetectionQueryCard
                state={detectionState}
                embLoading={audioSearchLoading}
                embError={audioSearchError ? new Error(audioSearchError) : null}
                onPlay={handlePlay}
                playingKey={playingKey}
                onSpectrogramClick={handleSpectrogramClick}
                onSwitchToStandalone={() => {
                  setMode("standalone");
                  setAudioSearchResults(null);
                  setAudioSearchError(null);
                  setDetectionQueryVector(null);
                  setDetectionModelVersion(null);
                }}
              />

              {/* Embedding Sets filter for detection re-search */}
              {detectionModelVersion && (
                <div className="space-y-2 pt-2 border-t">
                  <EmbeddingSetPanel
                    label="Embedding Sets"
                    selected={filterSelected}
                    collapsed={filterCollapsed ?? filterParentKeys}
                    folderTree={filterTree}
                    embeddingSets={filteredSets}
                    onToggleChild={toggleFilterChild}
                    onToggleParent={toggleFilterParent}
                    onToggleAll={toggleFilterAll}
                    onToggleCollapse={toggleFilterCollapse}
                    displayName={displayName}
                  />

                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                      <label className="text-xs text-muted-foreground">Top K:</label>
                      <Input
                        type="number"
                        min={1}
                        max={500}
                        value={topK}
                        onChange={(e) => setTopK(Math.max(1, Math.min(500, parseInt(e.target.value) || 20)))}
                        className="h-7 w-16 text-xs"
                      />
                    </div>
                    <div className="flex items-center gap-1.5">
                      <label className="text-xs text-muted-foreground">Metric:</label>
                      <Select value={metric} onValueChange={(v) => setMetric(v as "cosine" | "euclidean")}>
                        <SelectTrigger className="h-7 w-24 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="cosine">Cosine</SelectItem>
                          <SelectItem value="euclidean">Euclidean</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button
                      size="sm"
                      className="h-7 text-xs"
                      disabled={!detectionQueryVector || filterSelected.size === 0 || reSearchLoading}
                      onClick={handleDetectionReSearch}
                    >
                      {reSearchLoading ? (
                        <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
                      ) : (
                        <Search className="h-3.5 w-3.5 mr-1" />
                      )}
                      Re-search
                    </Button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <StandaloneQueryCard
              modelVersions={modelVersions}
              selectedModelVersion={selectedModelVersion}
              onModelVersionChange={setSelectedModelVersion}
              filteredSets={filteredSets}
              selectedEsId={selectedEsId}
              onEsChange={handleQueryEsSelect}
              selectedEs={selectedEs}
              windowIndex={windowIndex}
              onWindowIndexChange={setWindowIndex}
              maxWindowIndex={maxWindowIndex}
              topK={topK}
              onTopKChange={setTopK}
              metric={metric}
              onMetricChange={setMetric}
              onSearch={handleStandaloneSearch}
              isSearching={isSearching}
              onPlay={handlePlay}
              playingKey={playingKey}
              onSpectrogramClick={handleSpectrogramClick}
              queryTree={queryTree}
              queryCollapsed={queryCollapsed ?? queryAllKeys}
              onToggleQueryCollapse={toggleQueryCollapse}
              audioMap={audioMap}
              filterTree={filterTree}
              filterSelected={filterSelected}
              filterCollapsed={filterCollapsed ?? filterParentKeys}
              onToggleFilterChild={toggleFilterChild}
              onToggleFilterParent={toggleFilterParent}
              onToggleFilterAll={toggleFilterAll}
              onToggleFilterCollapse={toggleFilterCollapse}
              displayName={displayName}
            />
          )}
        </CardContent>
      </Card>

      {/* Search metadata */}
      {activeResults && (
        <div className="text-sm text-muted-foreground px-1">
          Found {results.length} results from {totalCandidates.toLocaleString()} candidates ({activeResults.metric}, model: {resultModelVersion})
        </div>
      )}

      {/* Error */}
      {searchError && (
        <div className="flex items-center gap-2 text-sm text-red-600 px-1">
          <AlertCircle className="h-4 w-4" />
          {searchError instanceof Error ? searchError.message : "Search failed"}
        </div>
      )}

      {/* Loading */}
      {isSearching && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground px-1">
          <Loader2 className="h-4 w-4 animate-spin" />
          Searching...
        </div>
      )}

      {/* Results Table */}
      {results.length > 0 && (
        <Card>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/30">
                  <th className="px-3 py-2 text-left w-10">#</th>
                  <th className="px-3 py-2 text-left w-20">Score</th>
                  <th className="px-3 py-2 text-left w-[170px]">Spectrogram</th>
                  <th className="px-3 py-2 text-left w-10">Play</th>
                  <th className="px-3 py-2 text-left">Audio File</th>
                  <th className="px-3 py-2 text-left">Folder</th>
                </tr>
              </thead>
              <tbody>
                {results.map((hit, idx) => {
                  const es = esMap.get(hit.embedding_set_id);
                  const windowSizeSec = es?.window_size_seconds ?? 5;
                  const playUrl = audioWindowUrl(
                    hit.audio_file_id,
                    hit.window_offset_seconds,
                    windowSizeSec,
                  );
                  const specUrl = audioSpectrogramPngUrl(
                    hit.audio_file_id,
                    hit.window_offset_seconds,
                    windowSizeSec,
                  );
                  const playKey = `${hit.audio_file_id}:${hit.window_offset_seconds}`;
                  const isPlaying = playingKey === playKey;

                  return (
                    <tr key={idx} className="border-b hover:bg-muted/20">
                      <td className="px-3 py-1.5 text-muted-foreground">
                        {idx + 1}
                      </td>
                      <td className={`px-3 py-1.5 ${scoreColor(hit.score)}`}>
                        {(hit.score * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-1.5">
                        <img
                          src={specUrl}
                          alt="spectrogram"
                          className="h-[60px] w-[160px] object-cover rounded cursor-pointer border"
                          loading="lazy"
                          onClick={(e) =>
                            handleSpectrogramClick(specUrl, windowSizeSec, e)
                          }
                        />
                      </td>
                      <td className="px-3 py-1.5">
                        <button
                          onClick={() => handlePlay(playKey, playUrl)}
                          className="p-1 rounded hover:bg-muted"
                          title={isPlaying ? "Pause" : "Play"}
                        >
                          {isPlaying ? (
                            <Pause className="h-3.5 w-3.5" />
                          ) : (
                            <Play className="h-3.5 w-3.5" />
                          )}
                        </button>
                      </td>
                      <td className="px-3 py-1.5 truncate max-w-[200px]" title={hit.audio_filename}>
                        {hit.audio_filename}
                      </td>
                      <td className="px-3 py-1.5 truncate max-w-[150px] text-muted-foreground" title={hit.audio_folder_path ?? ""}>
                        {hit.audio_folder_path ?? "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Empty state */}
      {!isSearching && !searchError && results.length === 0 && !activeResults && mode === "standalone" && (
        <div className="text-center py-12 text-muted-foreground">
          <Search className="h-8 w-8 mx-auto mb-3 opacity-50" />
          <p>Select an embedding set and window, then click Search to find similar audio.</p>
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
    </div>
  );
}

// ---- Detection Query Sub-card ----

function DetectionQueryCard({
  state,
  embLoading,
  embError,
  onPlay,
  playingKey,
  onSpectrogramClick,
  onSwitchToStandalone,
}: {
  state: DetectionSearchState;
  embLoading: boolean;
  embError: Error | null;
  onPlay: (key: string, url: string) => void;
  playingKey: string | null;
  onSpectrogramClick: (url: string, dur: number, e: React.MouseEvent) => void;
  onSwitchToStandalone: () => void;
}) {
  const clipDuration = state.endSec - state.startSec;
  const specUrl = detectionSpectrogramUrl(
    state.detectionJobId,
    state.filename,
    state.startSec,
    clipDuration,
  );
  const audioUrl = detectionAudioSliceUrl(
    state.detectionJobId,
    state.filename,
    state.startSec,
    clipDuration,
  );
  const playKey = `det:${state.detectionJobId}:${state.startSec}`;
  const isPlaying = playingKey === playKey;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <img
          src={specUrl}
          alt="query spectrogram"
          className="h-[60px] w-[160px] object-cover rounded border cursor-pointer"
          onClick={(e) => onSpectrogramClick(specUrl, clipDuration, e)}
        />
        <div className="flex-1 text-sm space-y-0.5">
          <div className="font-medium">Detection Query</div>
          <div className="text-muted-foreground">
            Job: {state.detectionJobId.slice(0, 8)}... | {state.startSec.toFixed(1)}s - {state.endSec.toFixed(1)}s
          </div>
          <div className="text-muted-foreground truncate text-xs">
            {state.filename}
          </div>
        </div>
        <button
          onClick={() => onPlay(playKey, audioUrl)}
          className="p-2 rounded hover:bg-muted"
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </button>
      </div>
      {embLoading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Encoding audio &amp; searching...
        </div>
      )}
      {embError && (
        <div className="flex items-center gap-2 text-sm text-red-600">
          <AlertCircle className="h-3.5 w-3.5" />
          {embError instanceof Error ? embError.message : "Search failed"}
        </div>
      )}
      <button
        onClick={onSwitchToStandalone}
        className="text-xs text-blue-600 hover:underline"
      >
        Switch to manual search
      </button>
    </div>
  );
}

// ---- Standalone Query Sub-card ----

function StandaloneQueryCard({
  modelVersions,
  selectedModelVersion,
  onModelVersionChange,
  filteredSets,
  selectedEsId,
  onEsChange,
  selectedEs,
  windowIndex,
  onWindowIndexChange,
  maxWindowIndex,
  topK,
  onTopKChange,
  metric,
  onMetricChange,
  onSearch,
  isSearching,
  onPlay,
  playingKey,
  onSpectrogramClick,
  queryTree,
  queryCollapsed,
  onToggleQueryCollapse,
  audioMap,
  filterTree,
  filterSelected,
  filterCollapsed,
  onToggleFilterChild,
  onToggleFilterParent,
  onToggleFilterAll,
  onToggleFilterCollapse,
  displayName,
}: {
  modelVersions: string[];
  selectedModelVersion: string;
  onModelVersionChange: (v: string) => void;
  filteredSets: EmbeddingSet[];
  selectedEsId: string;
  onEsChange: (id: string) => void;
  selectedEs: EmbeddingSet | null;
  windowIndex: number;
  onWindowIndexChange: (n: number) => void;
  maxWindowIndex: number;
  topK: number;
  onTopKChange: (n: number) => void;
  metric: "cosine" | "euclidean";
  onMetricChange: (m: "cosine" | "euclidean") => void;
  onSearch: () => void;
  isSearching: boolean;
  onPlay: (key: string, url: string) => void;
  playingKey: string | null;
  onSpectrogramClick: (url: string, dur: number, e: React.MouseEvent) => void;
  queryTree: ParentNode[];
  queryCollapsed: Set<string>;
  onToggleQueryCollapse: (parent: string) => void;
  audioMap: Map<string, { folder_path: string; filename: string }>;
  filterTree: ParentNode[];
  filterSelected: Set<string>;
  filterCollapsed: Set<string>;
  onToggleFilterChild: (sets: EmbeddingSet[]) => void;
  onToggleFilterParent: (node: ParentNode) => void;
  onToggleFilterAll: () => void;
  onToggleFilterCollapse: (parent: string) => void;
  displayName: (key: string) => string;
}) {
  const windowSizeSec = selectedEs?.window_size_seconds ?? 5;
  const specUrl = selectedEs
    ? audioSpectrogramPngUrl(
        selectedEs.audio_file_id,
        windowIndex * windowSizeSec,
        windowSizeSec,
      )
    : null;
  const audioUrl = selectedEs
    ? audioWindowUrl(
        selectedEs.audio_file_id,
        windowIndex * windowSizeSec,
        windowSizeSec,
      )
    : null;
  const playKey = selectedEs ? `es:${selectedEs.id}:${windowIndex}` : "";
  const isPlaying = playingKey === playKey;

  return (
    <div className="space-y-3">
      {/* Model filter */}
      <div className="max-w-xs">
        <label className="text-xs text-muted-foreground mb-1 block">Model</label>
        <Select value={selectedModelVersion || "__all__"} onValueChange={onModelVersionChange}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder="All models" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">All models</SelectItem>
            {modelVersions.map((v) => (
              <SelectItem key={v} value={v}>
                {v}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Two-panel layout: Query Source + Search Targets */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <EmbeddingQueryPanel
            label="Search Query Embedding"
            selectedEsId={selectedEsId}
            onSelectEs={onEsChange}
            collapsed={queryCollapsed}
            folderTree={queryTree}
            embeddingSets={filteredSets}
            onToggleCollapse={onToggleQueryCollapse}
            displayName={displayName}
            audioMap={audioMap}
          />
        </div>
        <div>
          <EmbeddingSetPanel
            label="Embedding Sets"
            selected={filterSelected}
            collapsed={filterCollapsed}
            folderTree={filterTree}
            embeddingSets={filteredSets}
            onToggleChild={onToggleFilterChild}
            onToggleParent={onToggleFilterParent}
            onToggleAll={onToggleFilterAll}
            onToggleCollapse={onToggleFilterCollapse}
            displayName={displayName}
          />
        </div>
      </div>

      {/* Window index + preview */}
      {selectedEs && (
        <div className="flex items-center gap-3">
          {specUrl && (
            <img
              src={specUrl}
              alt="query spectrogram"
              className="h-[60px] w-[160px] object-cover rounded border cursor-pointer"
              onClick={(e) => onSpectrogramClick(specUrl, windowSizeSec, e)}
            />
          )}
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              className="h-7 w-7"
              disabled={windowIndex <= 0}
              onClick={() => onWindowIndexChange(Math.max(0, windowIndex - 1))}
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <Input
              type="number"
              min={0}
              max={maxWindowIndex}
              value={windowIndex}
              onChange={(e) => onWindowIndexChange(Math.max(0, parseInt(e.target.value) || 0))}
              className="h-7 w-16 text-xs text-center"
            />
            <Button
              variant="outline"
              size="icon"
              className="h-7 w-7"
              onClick={() => onWindowIndexChange(windowIndex + 1)}
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
          {audioUrl && (
            <button
              onClick={() => onPlay(playKey, audioUrl)}
              className="p-2 rounded hover:bg-muted"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </button>
          )}
        </div>
      )}

      {/* Search controls */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-muted-foreground">Top K:</label>
          <Input
            type="number"
            min={1}
            max={500}
            value={topK}
            onChange={(e) => onTopKChange(Math.max(1, Math.min(500, parseInt(e.target.value) || 20)))}
            className="h-7 w-16 text-xs"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-muted-foreground">Metric:</label>
          <Select value={metric} onValueChange={(v) => onMetricChange(v as "cosine" | "euclidean")}>
            <SelectTrigger className="h-7 w-24 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cosine">Cosine</SelectItem>
              <SelectItem value="euclidean">Euclidean</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button
          size="sm"
          className="h-7 text-xs"
          disabled={!selectedEsId || filterSelected.size === 0 || isSearching}
          onClick={onSearch}
        >
          {isSearching ? (
            <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
          ) : (
            <Search className="h-3.5 w-3.5 mr-1" />
          )}
          Search
        </Button>
      </div>
    </div>
  );
}
