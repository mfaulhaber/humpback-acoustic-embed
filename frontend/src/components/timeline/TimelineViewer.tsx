// frontend/src/components/timeline/TimelineViewer.tsx
import { useState, useCallback, useEffect, useRef } from "react";
import { useLocation, useParams } from "react-router-dom";
import type { ZoomLevel } from "@/api/types";
import { useTimelineConfidence, useTimelineDetections, usePrepareStatus, useSaveLabels } from "@/hooks/queries/useTimeline";
import { useHydrophoneDetectionJobs, useExtractLabeledSamples } from "@/hooks/queries/useClassifier";
import { useLabelEdits } from "@/hooks/queries/useLabelEdits";
import { timelineAudioUrl } from "@/api/client";
import { TimelineHeader } from "./TimelineHeader";
import { ZoomSelector } from "./ZoomSelector";
import { PlaybackControls } from "./PlaybackControls";
import { SpectrogramViewport } from "./SpectrogramViewport";
import { LabelToolbar } from "./LabelToolbar";
import { LabelEditor } from "./LabelEditor";
import { ExtractDialog } from "../classifier/ExtractDialog";
import { Tag } from "lucide-react";
import { ZOOM_LEVELS, VIEWPORT_SPAN, COLORS, AUDIO_PREFETCH_SEC, AUDIO_FORMAT } from "./constants";
import type { LabelType } from "./constants";

export function TimelineViewer() {
  const { jobId } = useParams<{ jobId: string }>();
  const location = useLocation();
  const { data: jobs } = useHydrophoneDetectionJobs(0);
  const job = jobs?.find((j) => j.id === jobId);
  const prepareRequested = Boolean(
    (location.state as { prepareRequested?: boolean } | null)?.prepareRequested,
  );

  // Background tile cache state
  const [cacheComplete, setCacheComplete] = useState(false);

  // Poll prepare status
  const { data: prepareStatus } = usePrepareStatus(
    jobId ?? "",
    prepareRequested && !cacheComplete && !!jobId,
  );

  // Check completion
  useEffect(() => {
    if (!prepareStatus) return;
    const allDone = Object.values(prepareStatus).every(
      (z) => z.rendered >= z.total,
    );
    if (allDone) setCacheComplete(true);
  }, [prepareStatus]);

  // Core state
  const [centerTimestamp, setCenterTimestamp] = useState<number>(0);
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>("1h");
  const [isPlaying, setIsPlaying] = useState(false);
  const [freqRange, setFreqRange] = useState<[number, number]>([0, 3000]);
  const [showLabels, setShowLabels] = useState(false);
  const [speed, setSpeed] = useState(1);

  // Label mode state
  const [labelMode, setLabelMode] = useState(false);
  const [labelSubMode, setLabelSubMode] = useState<"select" | "add">("select");
  const [selectedLabel, setSelectedLabel] = useState<LabelType>("humpback");
  const [extractOpen, setExtractOpen] = useState(false);

  // Double-buffered audio elements for gapless playback
  const [playbackOriginEpoch, setPlaybackOriginEpoch] = useState(0);
  const audioRefA = useRef<HTMLAudioElement>(null);
  const audioRefB = useRef<HTMLAudioElement>(null);
  const activeRef = useRef<"A" | "B">("A");

  const getActiveAudio = useCallback(
    () => (activeRef.current === "A" ? audioRefA.current : audioRefB.current),
    [],
  );
  const getStandbyAudio = useCallback(
    () => (activeRef.current === "A" ? audioRefB.current : audioRefA.current),
    [],
  );

  const loadChunk = useCallback(
    (startEpoch: number, element: HTMLAudioElement) => {
      element.src = timelineAudioUrl(jobId ?? "", startEpoch, AUDIO_PREFETCH_SEC, AUDIO_FORMAT);
      element.playbackRate = speed;
      element.load();
    },
    [jobId, speed],
  );

  // Initialize center to job midpoint
  useEffect(() => {
    if (job?.start_timestamp && job?.end_timestamp && centerTimestamp === 0) {
      setCenterTimestamp(
        job.start_timestamp + (job.end_timestamp - job.start_timestamp) / 2,
      );
    }
  }, [job, centerTimestamp]);

  // Play/pause: load first chunk into active element, prefetch next into standby
  useEffect(() => {
    const active = getActiveAudio();
    if (!active) return;

    if (isPlaying) {
      const origin = centerTimestamp;
      setPlaybackOriginEpoch(origin);
      loadChunk(origin, active);

      const onCanPlay = () => {
        active.play().catch(() => {});
        const standby = getStandbyAudio();
        if (standby) {
          loadChunk(origin + AUDIO_PREFETCH_SEC, standby);
        }
      };
      active.addEventListener("canplay", onCanPlay, { once: true });
      return () => active.removeEventListener("canplay", onCanPlay);
    } else {
      active.pause();
      const standby = getStandbyAudio();
      if (standby) standby.pause();
    }
    // Only re-run when isPlaying toggles, not on every centerTimestamp tick
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying]);

  // Sync playback rate when speed changes
  useEffect(() => {
    const a = audioRefA.current;
    const b = audioRefB.current;
    if (a) a.playbackRate = speed;
    if (b) b.playbackRate = speed;
  }, [speed]);

  // Chunk-ended swap: when active audio ends, swap to standby and prefetch next
  useEffect(() => {
    const handleEnded = (source: "A" | "B") => () => {
      if (activeRef.current !== source) return; // stale event
      const standby = getStandbyAudio();
      const newOrigin = playbackOriginEpoch + AUDIO_PREFETCH_SEC;

      if (!standby || standby.readyState < 3) {
        // Standby not ready — reload from current position (gap fallback)
        const active = getActiveAudio();
        if (active) {
          setPlaybackOriginEpoch(newOrigin);
          loadChunk(newOrigin, active);
          active.addEventListener(
            "canplay",
            () => active.play().catch(() => {}),
            { once: true },
          );
        }
        return;
      }

      setPlaybackOriginEpoch(newOrigin);
      activeRef.current = activeRef.current === "A" ? "B" : "A";
      standby.play().catch(() => {});

      // Prefetch next into now-idle element
      const nowIdle = getStandbyAudio();
      if (nowIdle) {
        loadChunk(newOrigin + AUDIO_PREFETCH_SEC, nowIdle);
      }
    };

    const a = audioRefA.current;
    const b = audioRefB.current;
    const handleA = handleEnded("A");
    const handleB = handleEnded("B");
    a?.addEventListener("ended", handleA);
    b?.addEventListener("ended", handleB);
    return () => {
      a?.removeEventListener("ended", handleA);
      b?.removeEventListener("ended", handleB);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playbackOriginEpoch, speed]);

  // Audio-authoritative RAF loop: sync centerTimestamp from active audio element
  useEffect(() => {
    if (!isPlaying) return;
    let raf: number;
    const tick = () => {
      const active = getActiveAudio();
      if (active && !active.paused) {
        setCenterTimestamp(playbackOriginEpoch + active.currentTime);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, playbackOriginEpoch, getActiveAudio]);

  // Pan handler: pauses playback then updates position
  const handlePan = useCallback((t: number) => {
    setIsPlaying(false);
    setCenterTimestamp(t);
  }, []);

  // Data queries
  const { data: confidence } = useTimelineConfidence(jobId ?? "");
  const { data: detections } = useTimelineDetections(jobId ?? "");

  // Label editing hooks
  const { state: labelState, dispatch: labelDispatch, mergedRows, isDirty, selectedId } =
    useLabelEdits(detections ?? []);
  const saveMutation = useSaveLabels(jobId ?? "");
  const extractMutation = useExtractLabeledSamples();

  // Label mode is enabled only when paused and zoomed to 5m or 1m
  const labelModeEnabled = !isPlaying && (zoomLevel === "1m" || zoomLevel === "5m");

  // confidence.scores is consumed by the Minimap below

  // Zoom in/out
  const zoomIn = useCallback(() => {
    const idx = ZOOM_LEVELS.indexOf(zoomLevel);
    if (idx < ZOOM_LEVELS.length - 1) setZoomLevel(ZOOM_LEVELS[idx + 1]);
  }, [zoomLevel]);

  const zoomOut = useCallback(() => {
    const idx = ZOOM_LEVELS.indexOf(zoomLevel);
    if (idx > 0) setZoomLevel(ZOOM_LEVELS[idx - 1]);
  }, [zoomLevel]);

  // Enter label mode callback
  const enterLabelMode = useCallback(() => {
    setIsPlaying(false);
    setLabelMode(true);
  }, []);

  // Play/pause — exits label mode if active
  const togglePlay = useCallback(() => {
    if (labelMode) {
      if (isDirty && !confirm("Discard unsaved label changes?")) return;
      setLabelMode(false);
      labelDispatch({ type: "clear" });
    }
    setIsPlaying((prev) => !prev);
  }, [labelMode, isDirty, labelDispatch]);

  // Skip to next/prev detection
  // Detection row start_utc/end_utc are absolute UTC epoch seconds.
  const skipForward = useCallback(() => {
    if (!detections) return;
    const next = detections.find((d) => d.start_utc > centerTimestamp);
    if (next) setCenterTimestamp(next.start_utc);
  }, [detections, centerTimestamp]);

  const skipBack = useCallback(() => {
    if (!detections) return;
    const prev = [...detections].reverse().find((d) => d.end_utc < centerTimestamp);
    if (prev) setCenterTimestamp(prev.start_utc);
  }, [detections, centerTimestamp]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === " ") { e.preventDefault(); togglePlay(); }
      if (e.key === "+" || e.key === "=") zoomIn();
      if (e.key === "-") zoomOut();
      if (e.key === "ArrowLeft") {
        setCenterTimestamp((prev) => {
          const span = VIEWPORT_SPAN[zoomLevel];
          return Math.max(job?.start_timestamp ?? prev, prev - span / 10);
        });
      }
      if (e.key === "ArrowRight") {
        setCenterTimestamp((prev) => {
          const span = VIEWPORT_SPAN[zoomLevel];
          return Math.min(job?.end_timestamp ?? prev, prev + span / 10);
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [togglePlay, zoomIn, zoomOut, zoomLevel, job]);

  // Warn before navigating away with unsaved label edits
  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  if (!jobId || !job) {
    return (
      <div className="fixed left-60 flex items-center justify-center" style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}>
        Loading...
      </div>
    );
  }

  return (
    <div className="fixed left-60 flex flex-col font-mono text-xs overflow-hidden" style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}>
      {!cacheComplete && prepareStatus && (
        <div style={{
          position: "absolute", top: 4, right: 16,
          fontSize: 11, color: COLORS.textMuted, zIndex: 10,
        }}>
          Caching: {
            Object.entries(prepareStatus)
              .filter(([, z]) => z.rendered < z.total)
              .map(([zoom, z]) => `${zoom} ${z.rendered}/${z.total}`)
              .join(", ")
          }
        </div>
      )}
      <TimelineHeader
        hydrophone={job.hydrophone_name ?? job.hydrophone_id ?? ""}
        startTimestamp={job.start_timestamp ?? 0}
        endTimestamp={job.end_timestamp ?? 0}
      />

      {/* Main spectrogram viewport */}
      <div className="flex-1 flex flex-col mx-4 my-2 rounded relative overflow-hidden min-h-0" style={{ background: COLORS.bgDark }}>
        <SpectrogramViewport
          jobId={jobId}
          jobStart={job.start_timestamp ?? 0}
          jobEnd={job.end_timestamp ?? 0}
          centerTimestamp={centerTimestamp}
          zoomLevel={zoomLevel}
          freqRange={freqRange}
          isPlaying={isPlaying}
          scores={confidence?.scores ?? []}
          showLabels={showLabels}
          detections={detections ?? []}
          onCenterChange={setCenterTimestamp}
          onPan={handlePan}
          labelMode={labelMode}
          renderLabelEditor={(w, h) => (
            <LabelEditor
              mergedRows={mergedRows}
              mode={labelSubMode}
              selectedLabel={selectedLabel}
              selectedId={selectedId}
              dispatch={labelDispatch}
              jobStart={job.start_timestamp ?? 0}
              jobDuration={(job.end_timestamp ?? 0) - (job.start_timestamp ?? 0)}
              centerTimestamp={centerTimestamp}
              zoomLevel={zoomLevel}
              width={w}
              height={h}
            />
          )}
        />
        {/* Labels/Freq overlay — bottom-right of viewport */}
        <div
          className="absolute flex items-center gap-2 z-[8]"
          style={{ bottom: 24, right: 8 }}
        >
          <button
            onClick={() => setShowLabels((s) => !s)}
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
            style={{
              background: showLabels
                ? "rgba(64, 224, 192, 0.25)"
                : "rgba(30, 40, 50, 0.75)",
              color: COLORS.accent,
              backdropFilter: "blur(4px)",
            }}
          >
            <Tag size={10} /> Labels: {showLabels ? "ON" : "OFF"}
          </button>
          <span
            className="px-2 py-1 rounded text-[10px]"
            style={{
              background: "rgba(30, 40, 50, 0.75)",
              color: COLORS.accent,
              backdropFilter: "blur(4px)",
            }}
          >
            Freq: {freqRange[0] / 1000}–{freqRange[1] / 1000} kHz
          </span>
        </div>
      </div>

      {/* Double-buffered hidden audio elements for gapless playback */}
      <audio ref={audioRefA} preload="auto" style={{ display: "none" }} />
      <audio ref={audioRefB} preload="auto" style={{ display: "none" }} />

      {/* Pinned footer: zoom, playback controls, label toolbar */}
      <div
        className="shrink-0"
        style={{ background: COLORS.headerBg, borderTop: `1px solid ${COLORS.border}` }}
      >
        {/* Label toolbar (shown only in label mode) */}
        {labelMode && (
          <LabelToolbar
            mode={labelSubMode}
            onModeChange={setLabelSubMode}
            selectedLabel={selectedLabel}
            onLabelChange={(label) => {
              setSelectedLabel(label);
              if (labelSubMode === "select" && selectedId) {
                const parts = selectedId.split(":");
                if (parts.length === 2) {
                  labelDispatch({
                    type: "change_type",
                    start_utc: parseFloat(parts[0]),
                    end_utc: parseFloat(parts[1]),
                    label,
                  });
                }
              }
            }}
            onDelete={() => {
              if (!selectedId) return;
              // selectedId is a UTC key "start_utc:end_utc"
              const parts = selectedId.split(":");
              if (parts.length === 2) {
                labelDispatch({ type: "delete", start_utc: parseFloat(parts[0]), end_utc: parseFloat(parts[1]) });
              } else {
                labelDispatch({ type: "delete_by_id", id: selectedId });
              }
            }}
            onSave={() => {
              const items = labelState.edits.map((e) => ({
                action: e.action,
                start_utc: e.start_utc,
                end_utc: e.end_utc,
                new_start_utc: e.new_start_utc,
                new_end_utc: e.new_end_utc,
                label: e.label,
              }));
              saveMutation.mutate(items, {
                onSuccess: () => labelDispatch({ type: "clear" }),
              });
            }}
            onExtract={() => setExtractOpen(true)}
            onCancel={() => {
              if (isDirty && !confirm("Discard unsaved label changes?")) return;
              setLabelMode(false);
              labelDispatch({ type: "clear" });
            }}
            isDirty={isDirty}
            isSaving={saveMutation.isPending}
            hasSelection={selectedId !== null}
          />
        )}

        <ZoomSelector activeLevel={zoomLevel} onChange={setZoomLevel} />
        <PlaybackControls
          centerTimestamp={centerTimestamp}
          isPlaying={isPlaying}
          speed={speed}
          onPlayPause={togglePlay}
          onSkipBack={skipBack}
          onSkipForward={skipForward}
          onSpeedChange={setSpeed}
          onZoomIn={zoomIn}
          onZoomOut={zoomOut}
          onLabelMode={enterLabelMode}
          labelModeEnabled={labelModeEnabled}
        />
      </div>

      {/* Extract dialog */}
      {extractOpen && (
        <ExtractDialog
          open={extractOpen}
          onOpenChange={setExtractOpen}
          selectedIds={new Set([jobId ?? ""])}
          extractMutation={extractMutation}
          onSuccess={() => setExtractOpen(false)}
        />
      )}
    </div>
  );
}
