import { useState, useCallback, useEffect } from "react";
import { useLocation, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import type { DetectionRow, TimelineVocalizationLabel, ZoomLevel } from "@/api/types";
import { useTimelineConfidence, useTimelineDetections, usePrepareStatus, useSaveLabels } from "@/hooks/queries/useTimeline";
import { useHydrophoneDetectionJobs, useExtractLabeledSamples } from "@/hooks/queries/useClassifier";
import { useVocalizationOverlay } from "@/hooks/queries/useVocalizationOverlay";
import { useLabelEdits, type Action as LabelAction, type LabelEdit } from "@/hooks/queries/useLabelEdits";
import { useVocLabelEdits, serializeEdits, type VocLabelAction } from "@/hooks/queries/useVocLabelEdits";
import { useEmbeddingStatus, useSyncEmbeddings, useEmbeddingGenerationStatus } from "@/hooks/queries/useVocalization";
import { timelineTileUrl, timelineAudioUrl, patchVocalizationLabels } from "@/api/client";
import { TimelineProvider } from "./provider/TimelineProvider";
import { useTimelineContext } from "./provider/useTimelineContext";
import { FULL_ZOOM } from "./provider/types";
import { Spectrogram } from "./spectrogram/Spectrogram";
import { DetectionOverlay } from "./overlays/DetectionOverlay";
import { VocalizationOverlay } from "./overlays/VocalizationOverlay";
import { useOverlayContext } from "./overlays/OverlayContext";
import { ZoomSelector } from "./controls/ZoomSelector";
import { PlaybackControls } from "./controls/PlaybackControls";
import { EditToggle } from "./controls/EditToggle";
import { OverlayToggles } from "./controls/OverlayToggles";
import { TimelineFooter } from "./controls/TimelineFooter";
import { TimelineHeader } from "./TimelineHeader";
import { LabelEditor } from "./LabelEditor";
import { LabelToolbar } from "./LabelToolbar";
import { VocLabelEditor } from "./VocLabelEditor";
import { VocLabelToolbar } from "./VocLabelToolbar";
import { ExtractDialog } from "../classifier/ExtractDialog";
import { COLORS, AUDIO_FORMAT } from "./constants";
import type { LabelType } from "./constants";

const LABEL_KEYS: Record<string, LabelType | null> = {
  u: null, h: "humpback", o: "orca", s: "ship", b: "background",
};

function tileUrlBuilder(jobId: string, zoomLevel: string, tileIndex: number, freqMin: number, freqMax: number): string {
  return timelineTileUrl(jobId, zoomLevel, tileIndex, freqMin, freqMax);
}

function LabelEditorBridge({
  mergedRows, mode, selectedLabel, selectedId, dispatch, jobStart, jobDuration,
}: {
  mergedRows: DetectionRow[];
  mode: "select" | "add";
  selectedLabel: LabelType | null;
  selectedId: string | null;
  dispatch: React.Dispatch<LabelAction>;
  jobStart: number;
  jobDuration: number;
}) {
  const { canvasWidth, canvasHeight } = useOverlayContext();
  const ctx = useTimelineContext();
  return (
    <LabelEditor
      mergedRows={mergedRows}
      mode={mode}
      selectedLabel={selectedLabel}
      selectedId={selectedId}
      dispatch={dispatch}
      jobStart={jobStart}
      jobDuration={jobDuration}
      centerTimestamp={ctx.centerTimestamp}
      zoomLevel={ctx.activePreset.key as ZoomLevel}
      width={canvasWidth}
      height={canvasHeight}
    />
  );
}

function VocLabelEditorBridge({
  detectionRows, vocLabels, edits, selectedRowId, dispatch,
}: {
  detectionRows: DetectionRow[];
  vocLabels: TimelineVocalizationLabel[];
  edits: Map<string, { adds: Set<string>; removes: Set<string> }>;
  selectedRowId: string | null;
  dispatch: React.Dispatch<VocLabelAction>;
}) {
  const { canvasWidth, canvasHeight } = useOverlayContext();
  const ctx = useTimelineContext();
  return (
    <VocLabelEditor
      detectionRows={detectionRows}
      vocLabels={vocLabels}
      edits={edits}
      selectedRowId={selectedRowId}
      dispatch={dispatch}
      centerTimestamp={ctx.centerTimestamp}
      zoomLevel={ctx.activePreset.key as ZoomLevel}
      width={canvasWidth}
      height={canvasHeight}
    />
  );
}

export function ClassifierTimeline() {
  const { jobId } = useParams<{ jobId: string }>();
  const location = useLocation();
  const queryClient = useQueryClient();
  const { data: jobs } = useHydrophoneDetectionJobs(0);
  const job = jobs?.find((j) => j.id === jobId);
  const prepareRequested = Boolean(
    (location.state as { prepareRequested?: boolean } | null)?.prepareRequested,
  );

  const [cacheComplete, setCacheComplete] = useState(false);
  const { data: prepareStatus } = usePrepareStatus(jobId ?? "", prepareRequested && !cacheComplete && !!jobId);

  useEffect(() => {
    if (!prepareStatus) return;
    if (Object.values(prepareStatus).every((z) => z.rendered >= z.total)) setCacheComplete(true);
  }, [prepareStatus]);

  // Overlay & label mode state
  const [overlayMode, setOverlayMode] = useState<"off" | "detection" | "vocalization">("off");
  const [labelMode, setLabelMode] = useState(false);
  const [labelEditMode, setLabelEditMode] = useState<"detection" | "vocalization" | null>(null);
  const [labelSubMode, setLabelSubMode] = useState<"select" | "add">("select");
  const [selectedLabel, setSelectedLabel] = useState<LabelType | null>("humpback");
  const [extractOpen, setExtractOpen] = useState(false);

  // Embedding sync
  const { data: embeddingStatus } = useEmbeddingStatus(jobId ?? null);
  const { data: embGenStatus } = useEmbeddingGenerationStatus(jobId ?? null);
  const syncMut = useSyncEmbeddings();
  const isSyncing = embGenStatus?.status === "queued" || embGenStatus?.status === "running";
  const lastSyncSummary = embGenStatus?.mode === "sync" && embGenStatus?.status === "complete"
    ? embGenStatus.result_summary : null;

  // Data queries
  const { data: confidence } = useTimelineConfidence(jobId ?? "");
  const { data: detections } = useTimelineDetections(jobId ?? "");
  const { labels: vocalizationLabels, hasVocalizationData } = useVocalizationOverlay(jobId ?? "");

  // Label editing hooks (detection)
  const { state: labelState, dispatch: labelDispatch, mergedRows, isDirty, selectedId } = useLabelEdits(detections ?? []);
  const saveMutation = useSaveLabels(jobId ?? "");
  const extractMutation = useExtractLabeledSamples();

  // Label editing hooks (vocalization)
  const { state: vocLabelState, dispatch: vocLabelDispatch, isDirty: vocIsDirty, editCount: vocEditCount, selectedRowId: vocSelectedRowId } = useVocLabelEdits();
  const [vocSaving, setVocSaving] = useState(false);

  // Sync selectedLabel to detection's label on selection change
  useEffect(() => {
    if (labelSubMode !== "select" || !selectedId) return;
    const row = mergedRows.find((r) => r.row_id === selectedId);
    if (!row) return;
    const label: LabelType | null =
      row.humpback === 1 ? "humpback" : row.orca === 1 ? "orca" :
      row.ship === 1 ? "ship" : row.background === 1 ? "background" : null;
    setSelectedLabel(label);
  }, [selectedId, labelSubMode, mergedRows]);

  // Warn on unsaved changes
  useEffect(() => {
    if (!isDirty && !vocIsDirty) return;
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty, vocIsDirty]);

  const exitLabelMode = useCallback((force = false) => {
    if (!force && labelEditMode === "detection" && isDirty && !confirm("Discard unsaved label changes?")) return false;
    if (!force && labelEditMode === "vocalization" && vocIsDirty && !confirm("Discard unsaved vocalization label changes?")) return false;
    setLabelMode(false);
    setLabelEditMode(null);
    labelDispatch({ type: "clear" });
    vocLabelDispatch({ type: "clear" });
    return true;
  }, [labelEditMode, isDirty, vocIsDirty, labelDispatch, vocLabelDispatch]);

  const toggleOverlay = useCallback((key: string) => {
    if (key === "detection") {
      if (overlayMode === "detection") {
        if (labelMode && labelEditMode === "detection" && !exitLabelMode()) return;
        setOverlayMode("off");
      } else {
        if (labelMode && labelEditMode === "vocalization" && !exitLabelMode()) return;
        setOverlayMode("detection");
      }
    } else if (key === "vocalization") {
      if (overlayMode === "vocalization") {
        if (labelMode && labelEditMode === "vocalization" && !exitLabelMode()) return;
        setOverlayMode("off");
      } else {
        if (labelMode && labelEditMode === "detection" && !exitLabelMode()) return;
        setOverlayMode("vocalization");
      }
    }
  }, [overlayMode, labelMode, labelEditMode, exitLabelMode]);

  const handleDetectionBarClick = useCallback(
    (row: DetectionRow) => {
      if (labelMode) return;
      setLabelMode(true);
      setLabelEditMode("detection");
      setLabelSubMode("select");
      labelDispatch({ type: "select", id: row.row_id });
      const label: LabelType | null =
        row.humpback === 1 ? "humpback" : row.orca === 1 ? "orca" :
        row.ship === 1 ? "ship" : row.background === 1 ? "background" : null;
      setSelectedLabel(label);
    },
    [labelMode, labelDispatch],
  );

  if (!jobId || !job) {
    return (
      <div className="fixed left-60 flex items-center justify-center" style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}>
        Loading...
      </div>
    );
  }

  const jobStart = job.start_timestamp ?? 0;
  const jobEnd = job.end_timestamp ?? 0;

  const audioUrlBuilder = (startEpoch: number, durationSec: number) =>
    timelineAudioUrl(jobId, startEpoch, durationSec, AUDIO_FORMAT);

  return (
    <div className="fixed left-60 flex flex-col font-mono text-xs overflow-hidden" style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}>
      {!cacheComplete && prepareStatus && (
        <div style={{ position: "absolute", top: 4, right: 16, fontSize: 11, color: COLORS.textMuted, zIndex: 10 }}>
          Caching: {Object.entries(prepareStatus).filter(([, z]) => z.rendered < z.total).map(([zoom, z]) => `${zoom} ${z.rendered}/${z.total}`).join(", ")}
        </div>
      )}
      <TimelineHeader
        hydrophone={job.hydrophone_name ?? job.hydrophone_id ?? ""}
        startTimestamp={jobStart}
        endTimestamp={jobEnd}
        syncNeeded={embeddingStatus?.sync_needed ?? false}
        isSyncing={isSyncing}
        syncSummary={lastSyncSummary}
        onSyncEmbeddings={() => { if (jobId) syncMut.mutate(jobId); }}
      />

      <TimelineProvider
        jobStart={jobStart}
        jobEnd={jobEnd}
        zoomLevels={FULL_ZOOM}
        defaultZoom="1h"
        playback="gapless"
        audioUrlBuilder={audioUrlBuilder}
      >
        <ClassifierTimelineBody
          jobId={jobId}
          jobStart={jobStart}
          jobEnd={jobEnd}
          confidence={confidence}
          detections={detections ?? []}
          vocalizationLabels={vocalizationLabels}
          hasVocalizationData={hasVocalizationData}
          overlayMode={overlayMode}
          labelMode={labelMode}
          labelEditMode={labelEditMode}
          labelSubMode={labelSubMode}
          selectedLabel={selectedLabel}
          mergedRows={mergedRows}
          selectedId={selectedId}
          labelDispatch={labelDispatch}
          labelEdits={labelState.edits}
          isDirty={isDirty}
          vocLabelEdits={vocLabelState.edits}
          vocLabelDispatch={vocLabelDispatch}
          vocIsDirty={vocIsDirty}
          vocEditCount={vocEditCount}
          vocSelectedRowId={vocSelectedRowId}
          vocSaving={vocSaving}
          setVocSaving={setVocSaving}
          saveMutation={saveMutation}
          extractMutation={extractMutation}
          setLabelMode={setLabelMode}
          setLabelEditMode={setLabelEditMode}
          setLabelSubMode={setLabelSubMode}
          setSelectedLabel={setSelectedLabel}
          exitLabelMode={exitLabelMode}
          toggleOverlay={toggleOverlay}
          onDetectionBarClick={handleDetectionBarClick}
          extractOpen={extractOpen}
          setExtractOpen={setExtractOpen}
          queryClient={queryClient}
        />
      </TimelineProvider>
    </div>
  );
}

interface ClassifierTimelineBodyProps {
  jobId: string;
  jobStart: number;
  jobEnd: number;
  confidence: { scores: (number | null)[]; window_sec?: number } | undefined;
  detections: DetectionRow[];
  vocalizationLabels: TimelineVocalizationLabel[];
  hasVocalizationData: boolean;
  overlayMode: "off" | "detection" | "vocalization";
  labelMode: boolean;
  labelEditMode: "detection" | "vocalization" | null;
  labelSubMode: "select" | "add";
  selectedLabel: LabelType | null;
  mergedRows: DetectionRow[];
  selectedId: string | null;
  labelDispatch: React.Dispatch<LabelAction>;
  labelEdits: LabelEdit[];
  isDirty: boolean;
  vocLabelEdits: Map<string, { adds: Set<string>; removes: Set<string> }>;
  vocLabelDispatch: React.Dispatch<VocLabelAction>;
  vocIsDirty: boolean;
  vocEditCount: number;
  vocSelectedRowId: string | null;
  vocSaving: boolean;
  setVocSaving: (v: boolean) => void;
  saveMutation: ReturnType<typeof useSaveLabels>;
  extractMutation: ReturnType<typeof useExtractLabeledSamples>;
  setLabelMode: (v: boolean) => void;
  setLabelEditMode: (v: "detection" | "vocalization" | null) => void;
  setLabelSubMode: (v: "select" | "add") => void;
  setSelectedLabel: (v: LabelType | null) => void;
  exitLabelMode: (force?: boolean) => boolean;
  toggleOverlay: (key: string) => void;
  onDetectionBarClick: (row: DetectionRow) => void;
  extractOpen: boolean;
  setExtractOpen: (v: boolean) => void;
  queryClient: ReturnType<typeof useQueryClient>;
}

function ClassifierTimelineBody({
  jobId, jobStart, jobEnd, confidence, detections, vocalizationLabels, hasVocalizationData,
  overlayMode, labelMode, labelEditMode, labelSubMode, selectedLabel,
  mergedRows, selectedId, labelDispatch, labelEdits, isDirty,
  vocLabelEdits, vocLabelDispatch, vocIsDirty, vocEditCount, vocSelectedRowId,
  vocSaving, setVocSaving, saveMutation, extractMutation,
  setLabelMode, setLabelEditMode, setLabelSubMode, setSelectedLabel,
  exitLabelMode, toggleOverlay, onDetectionBarClick,
  extractOpen, setExtractOpen, queryClient,
}: ClassifierTimelineBodyProps) {
  const ctx = useTimelineContext();

  const labelModeEnabled = !ctx.isPlaying && (ctx.activePreset.key === "5m" || ctx.activePreset.key === "1m");

  const toggleLabelMode = useCallback(() => {
    if (labelMode) {
      exitLabelMode();
    } else {
      ctx.pause();
      setLabelMode(true);
      setLabelEditMode(overlayMode === "vocalization" ? "vocalization" : "detection");
    }
  }, [labelMode, overlayMode, exitLabelMode, ctx, setLabelMode, setLabelEditMode]);

  // Handle detection bar click with zoom/play precondition checks
  const handleDetectionClick = useCallback(
    (row: DetectionRow, _x: number, _y: number) => {
      if (ctx.isPlaying) return;
      if (ctx.activePreset.key !== "5m" && ctx.activePreset.key !== "1m") return;
      if (overlayMode !== "detection") return;
      onDetectionBarClick(row);
    },
    [ctx.isPlaying, ctx.activePreset.key, overlayMode, onDetectionBarClick],
  );

  // Keyboard shortcuts for label keys and space override (capture phase for dirty-check)
  useEffect(() => {
    const handleCapture = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === " " && labelMode) {
        if (!exitLabelMode()) {
          e.stopImmediatePropagation();
          e.preventDefault();
          return;
        }
      }
    };
    window.addEventListener("keydown", handleCapture, true);
    return () => window.removeEventListener("keydown", handleCapture, true);
  }, [labelMode, exitLabelMode]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (labelMode && labelEditMode === "detection" && e.key in LABEL_KEYS) {
        e.preventDefault();
        const label = LABEL_KEYS[e.key];
        setSelectedLabel(label);
        if (labelSubMode === "select" && selectedId) {
          if (label === null) {
            labelDispatch({ type: "clear_label", row_id: selectedId });
          } else {
            labelDispatch({ type: "change_type", row_id: selectedId, label });
          }
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [labelMode, labelEditMode, labelSubMode, selectedId, labelDispatch, setSelectedLabel]);

  return (
    <>
      <div className="flex-1 flex flex-col mx-4 my-2 rounded overflow-hidden min-h-0" style={{ background: COLORS.bgDark }}>
        <Spectrogram
          jobId={jobId}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
          scores={confidence?.scores}
          windowSec={confidence?.window_sec}
        >
          {overlayMode === "detection" && !labelMode && (
            <DetectionOverlay
              detections={detections}
              visible={true}
              onDetectionClick={handleDetectionClick}
            />
          )}
          {overlayMode === "vocalization" && !labelMode && (
            <VocalizationOverlay labels={vocalizationLabels} visible={true} />
          )}
          {labelMode && labelEditMode === "detection" && (
            <LabelEditorBridge
              mergedRows={mergedRows}
              mode={labelSubMode}
              selectedLabel={selectedLabel}
              selectedId={selectedId}
              dispatch={labelDispatch}
              jobStart={jobStart}
              jobDuration={jobEnd - jobStart}
            />
          )}
          {labelMode && labelEditMode === "vocalization" && (
            <VocLabelEditorBridge
              detectionRows={detections}
              vocLabels={vocalizationLabels}
              edits={vocLabelEdits}
              selectedRowId={vocSelectedRowId}
              dispatch={vocLabelDispatch}
            />
          )}
        </Spectrogram>
      </div>

      <TimelineFooter>
        {labelMode && labelEditMode === "detection" && (
          <LabelToolbar
            mode={labelSubMode}
            onModeChange={setLabelSubMode}
            selectedLabel={selectedLabel}
            onLabelChange={(label) => {
              setSelectedLabel(label);
              if (labelSubMode === "select" && selectedId) {
                if (label === null) {
                  labelDispatch({ type: "clear_label", row_id: selectedId });
                } else {
                  labelDispatch({ type: "change_type", row_id: selectedId, label });
                }
              }
            }}
            onDelete={() => { if (selectedId) labelDispatch({ type: "delete", row_id: selectedId }); }}
            onSave={() => {
              const items = labelEdits.map((e) => ({
                action: e.action, row_id: e.row_id,
                start_utc: e.start_utc, end_utc: e.end_utc, label: e.label,
              }));
              saveMutation.mutate(items, { onSuccess: () => labelDispatch({ type: "clear" }) });
            }}
            onExtract={() => setExtractOpen(true)}
            onCancel={() => {
              if (isDirty && !confirm("Discard unsaved label changes?")) return;
              setLabelMode(false);
              setLabelEditMode(null);
              labelDispatch({ type: "clear" });
            }}
            isDirty={isDirty}
            isSaving={saveMutation.isPending}
            hasSelection={selectedId !== null}
          />
        )}
        {labelMode && labelEditMode === "vocalization" && (
          <VocLabelToolbar
            onSave={async () => {
              if (!jobId) return;
              setVocSaving(true);
              try {
                const items = serializeEdits(vocLabelEdits);
                await patchVocalizationLabels(jobId, { edits: items });
                await queryClient.invalidateQueries({ queryKey: ["vocalizationLabelsAll", jobId] });
                vocLabelDispatch({ type: "clear" });
              } finally {
                setVocSaving(false);
              }
            }}
            onCancel={() => {
              if (vocIsDirty && !confirm("Discard unsaved vocalization label changes?")) return;
              setLabelMode(false);
              setLabelEditMode(null);
              vocLabelDispatch({ type: "clear" });
            }}
            isDirty={vocIsDirty}
            isSaving={vocSaving}
            editCount={vocEditCount}
          />
        )}
        <div className="flex items-center">
          <div className="flex-1" />
          <ZoomSelector />
          <div className="flex-1 flex justify-end gap-2">
            <OverlayToggles
              options={[
                { key: "detection", label: "Detections", active: overlayMode === "detection" },
                ...(hasVocalizationData ? [{ key: "vocalization", label: "Vocalizations", active: overlayMode === "vocalization" }] : []),
              ]}
              onToggle={toggleOverlay}
            />
          </div>
        </div>
        <PlaybackControls>
          <EditToggle
            active={labelMode}
            enabled={labelModeEnabled}
            label="Label"
            onToggle={toggleLabelMode}
          />
        </PlaybackControls>
      </TimelineFooter>

      {extractOpen && (
        <ExtractDialog
          open={extractOpen}
          onOpenChange={setExtractOpen}
          selectedIds={new Set([jobId])}
          extractMutation={extractMutation}
          onSuccess={() => setExtractOpen(false)}
        />
      )}
    </>
  );
}
