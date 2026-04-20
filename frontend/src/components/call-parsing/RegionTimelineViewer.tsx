import { useState, useCallback, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, X, Check, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ZoomLevel, RegionCorrection } from "@/api/types";
import { regionTileUrl, regionAudioSliceUrl } from "@/api/client";
import {
  useRegionDetectionJobs,
  useRegionJobRegions,
  useRegionJobConfidence,
  useRegionCorrections,
  useSaveRegionCorrections,
} from "@/hooks/queries/useCallParsing";
import { RegionEditOverlay } from "./RegionEditOverlay";
import { SpectrogramViewport } from "@/components/timeline/SpectrogramViewport";
import { ZoomSelector } from "@/components/timeline/ZoomSelector";
import { PlaybackControls } from "@/components/timeline/PlaybackControls";
import {
  ZOOM_LEVELS,
  VIEWPORT_SPAN,
  COLORS,
  AUDIO_PREFETCH_SEC,
  FREQ_AXIS_WIDTH_PX,
} from "@/components/timeline/constants";

const NOOP = () => {};

function tileUrlBuilder(
  jobId: string,
  zoomLevel: string,
  tileIndex: number,
  _freqMin: number,
  _freqMax: number,
): string {
  return regionTileUrl(jobId, zoomLevel, tileIndex);
}

function audioSliceUrlBuilder(
  jobId: string,
  jobStart: number,
  startEpoch: number,
  durationSec: number,
): string {
  const jobRelative = startEpoch - jobStart;
  return regionAudioSliceUrl(jobId, Math.max(0, jobRelative), durationSec);
}

export function RegionTimelineViewer() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { data: jobs } = useRegionDetectionJobs(0);
  const job = jobs?.find((j) => j.id === jobId);
  const { data: regions } = useRegionJobRegions(jobId ?? null);
  const { data: confidence } = useRegionJobConfidence(jobId ?? null);
  const { data: savedCorrections } = useRegionCorrections(jobId ?? null);
  const saveCorrections = useSaveRegionCorrections();

  const [centerTimestamp, setCenterTimestamp] = useState<number>(0);
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>("1h");
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [editMode, setEditMode] = useState(false);
  const [addMode, setAddMode] = useState(false);
  const [showRegionOverlay, setShowRegionOverlay] = useState(true);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, RegionCorrection>
  >(new Map());

  // Load saved corrections into pending state when entering edit mode
  useEffect(() => {
    if (editMode && savedCorrections) {
      const map = new Map<string, RegionCorrection>();
      for (const c of savedCorrections) {
        map.set(c.region_id, {
          region_id: c.region_id,
          correction_type: c.correction_type,
          start_sec: c.start_sec,
          end_sec: c.end_sec,
        });
      }
      setPendingCorrections(map);
    }
  }, [editMode, savedCorrections]);

  const handleCorrection = useCallback((correction: RegionCorrection) => {
    setPendingCorrections((prev) => {
      const next = new Map(prev);
      next.set(correction.region_id, correction);
      return next;
    });
    if (correction.correction_type === "add") {
      setAddMode(false);
    }
  }, []);

  const handleSaveCorrections = useCallback(() => {
    if (!jobId) return;
    const corrections = Array.from(pendingCorrections.values());
    saveCorrections.mutate(
      { jobId, corrections },
      {
        onSuccess: () => {
          setEditMode(false);
          setPendingCorrections(new Map());
          setSelectedRegionId(null);
        },
      },
    );
  }, [jobId, pendingCorrections, saveCorrections]);

  const handleCancelEdit = useCallback(() => {
    setEditMode(false);
    setAddMode(false);
    setPendingCorrections(new Map());
    setSelectedRegionId(null);
  }, []);

  const handleRegionClick = useCallback(
    (regionId: string) => {
      if (!isPlaying && job?.status === "complete") {
        setEditMode(true);
        setSelectedRegionId(regionId);
      }
    },
    [isPlaying, job?.status],
  );

  const toggleEditMode = useCallback(() => {
    if (editMode) {
      handleCancelEdit();
    } else {
      setEditMode(true);
    }
  }, [editMode, handleCancelEdit]);

  // Track viewport dimensions for the edit overlay
  const viewportContainerRef = useRef<HTMLDivElement>(null);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

  useEffect(() => {
    const el = viewportContainerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setViewportWidth(entry.contentRect.width);
      setViewportHeight(entry.contentRect.height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const jobStart = job?.start_timestamp ?? 0;
  const jobEnd = job?.end_timestamp ?? 0;

  // Initialize center to job midpoint
  useEffect(() => {
    if (jobStart && jobEnd && centerTimestamp === 0) {
      setCenterTimestamp(jobStart + (jobEnd - jobStart) / 2);
    }
  }, [jobStart, jobEnd, centerTimestamp]);

  // Double-buffered audio for gapless playback
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
      element.src = audioSliceUrlBuilder(
        jobId ?? "",
        jobStart,
        startEpoch,
        AUDIO_PREFETCH_SEC,
      );
      element.playbackRate = speed;
      element.load();
    },
    [jobId, jobStart, speed],
  );

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
        if (standby) loadChunk(origin + AUDIO_PREFETCH_SEC, standby);
      };
      active.addEventListener("canplay", onCanPlay, { once: true });
      return () => active.removeEventListener("canplay", onCanPlay);
    } else {
      active.pause();
      const standby = getStandbyAudio();
      if (standby) standby.pause();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying]);

  useEffect(() => {
    const a = audioRefA.current;
    const b = audioRefB.current;
    if (a) a.playbackRate = speed;
    if (b) b.playbackRate = speed;
  }, [speed]);

  useEffect(() => {
    const handleEnded = (source: "A" | "B") => () => {
      if (activeRef.current !== source) return;
      const standby = getStandbyAudio();
      const newOrigin = playbackOriginEpoch + AUDIO_PREFETCH_SEC;
      if (!standby || standby.readyState < 3) {
        const active = getActiveAudio();
        if (active) {
          setPlaybackOriginEpoch(newOrigin);
          loadChunk(newOrigin, active);
          active.addEventListener("canplay", () => active.play().catch(() => {}), { once: true });
        }
        return;
      }
      setPlaybackOriginEpoch(newOrigin);
      activeRef.current = activeRef.current === "A" ? "B" : "A";
      standby.play().catch(() => {});
      const nowIdle = getStandbyAudio();
      if (nowIdle) loadChunk(newOrigin + AUDIO_PREFETCH_SEC, nowIdle);
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

  // RAF loop: sync centerTimestamp from audio playback
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

  const handlePan = useCallback((t: number) => {
    setIsPlaying(false);
    setCenterTimestamp(t);
  }, []);

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const zoomIn = useCallback(() => {
    const idx = ZOOM_LEVELS.indexOf(zoomLevel);
    if (idx < ZOOM_LEVELS.length - 1) setZoomLevel(ZOOM_LEVELS[idx + 1]);
  }, [zoomLevel]);

  const zoomOut = useCallback(() => {
    const idx = ZOOM_LEVELS.indexOf(zoomLevel);
    if (idx > 0) setZoomLevel(ZOOM_LEVELS[idx - 1]);
  }, [zoomLevel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === " ") { e.preventDefault(); togglePlay(); }
      if (e.key === "+" || e.key === "=") zoomIn();
      if (e.key === "-") zoomOut();
      if (e.key === "ArrowLeft") {
        setCenterTimestamp((prev) => Math.max(jobStart, prev - VIEWPORT_SPAN[zoomLevel] / 10));
      }
      if (e.key === "ArrowRight") {
        setCenterTimestamp((prev) => Math.min(jobEnd, prev + VIEWPORT_SPAN[zoomLevel] / 10));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [togglePlay, zoomIn, zoomOut, zoomLevel, jobStart, jobEnd]);

  if (!jobId || !job) {
    return (
      <div
        className="fixed left-60 flex items-center justify-center"
        style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}
      >
        Loading...
      </div>
    );
  }

  const regionCount = regions?.length ?? job.region_count ?? 0;
  const startStr = job.start_timestamp
    ? new Date(job.start_timestamp * 1000).toISOString().slice(0, 16).replace("T", " ") + " UTC"
    : "";
  const endStr = job.end_timestamp
    ? new Date(job.end_timestamp * 1000).toISOString().slice(0, 16).replace("T", " ") + " UTC"
    : "";

  return (
    <div
      className="fixed left-60 flex flex-col font-mono text-xs overflow-hidden"
      style={{ top: "3rem", right: 0, bottom: 0, background: COLORS.bg, color: COLORS.text, zIndex: 40 }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 py-2 shrink-0"
        style={{ background: COLORS.headerBg, borderBottom: `1px solid ${COLORS.border}` }}
      >
        <Button variant="ghost" size="sm" onClick={() => navigate("/app/call-parsing/detection")}>
          <ArrowLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <span style={{ color: COLORS.text, fontWeight: 600 }}>Region Detection Timeline</span>
        <span className="truncate min-w-0" style={{ color: COLORS.textMuted }}>
          {startStr} — {endStr}
        </span>
        <span className="shrink-0" style={{ color: COLORS.accent }}>
          {regionCount} region{regionCount !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Spectrogram viewport */}
      <div
        ref={viewportContainerRef}
        className="flex-1 flex flex-col mx-4 my-2 rounded relative overflow-hidden min-h-0"
        style={{ background: COLORS.bgDark }}
      >
        <SpectrogramViewport
          jobId={jobId}
          jobStart={jobStart}
          jobEnd={jobEnd}
          centerTimestamp={centerTimestamp}
          zoomLevel={zoomLevel}
          freqRange={[0, 3000]}
          isPlaying={isPlaying}
          scores={confidence?.scores ?? []}
          windowSec={confidence?.window_sec}
          showLabels={true}
          detections={[]}
          onCenterChange={setCenterTimestamp}
          onPan={editMode ? NOOP : handlePan}
          labelMode={false}
          labelEditMode={null}
          overlayMode={showRegionOverlay ? "region" : undefined}
          regions={showRegionOverlay ? regions : undefined}
          tileUrlBuilder={tileUrlBuilder}
          onRegionClick={!isPlaying && !editMode ? handleRegionClick : undefined}
          disablePan={editMode}
        />
        {editMode && regions && (
          <RegionEditOverlay
            regions={regions}
            corrections={pendingCorrections}
            jobStart={jobStart}
            centerTimestamp={centerTimestamp}
            zoomLevel={zoomLevel}
            width={Math.max(0, viewportWidth - FREQ_AXIS_WIDTH_PX)}
            height={viewportHeight}
            leftOffset={FREQ_AXIS_WIDTH_PX}
            addMode={addMode}
            selectedRegionId={selectedRegionId}
            onSelectRegion={setSelectedRegionId}
            onCorrection={handleCorrection}
          />
        )}
      </div>

      {/* Hidden audio elements */}
      <audio ref={audioRefA} preload="auto" style={{ display: "none" }} />
      <audio ref={audioRefB} preload="auto" style={{ display: "none" }} />

      {/* Footer controls */}
      <div
        className="shrink-0"
        style={{ background: COLORS.headerBg, borderTop: `1px solid ${COLORS.border}` }}
      >
        <div className="flex items-center px-4 py-1">
          <div className="flex-1">
            <ZoomSelector activeLevel={zoomLevel} onChange={setZoomLevel} />
          </div>
          {editMode && (
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-[10px] font-mono" style={{ color: COLORS.textMuted }}>
                {pendingCorrections.size} edit{pendingCorrections.size !== 1 ? "s" : ""}
              </span>
              <button
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium"
                style={{ color: COLORS.textMuted, border: `1px solid ${COLORS.border}` }}
                onClick={handleCancelEdit}
              >
                <X size={10} /> Cancel
              </button>
              <button
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium"
                style={{
                  background: addMode ? "rgba(100, 180, 255, 0.3)" : "transparent",
                  color: addMode ? "rgba(100, 180, 255, 1)" : COLORS.accent,
                  border: `1px solid ${addMode ? "rgba(100, 180, 255, 0.8)" : COLORS.accent}`,
                }}
                onClick={() => setAddMode((v) => !v)}
              >
                <Plus size={10} /> Add
              </button>
              <button
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium"
                style={{
                  background: pendingCorrections.size > 0 ? "rgba(16, 185, 129, 0.8)" : "transparent",
                  color: pendingCorrections.size > 0 ? "#fff" : COLORS.textMuted,
                  border: `1px solid ${pendingCorrections.size > 0 ? "rgba(16, 185, 129, 0.8)" : COLORS.border}`,
                  opacity: pendingCorrections.size > 0 && !saveCorrections.isPending ? 1 : 0.5,
                }}
                onClick={handleSaveCorrections}
                disabled={pendingCorrections.size === 0 || saveCorrections.isPending}
              >
                <Check size={10} /> {saveCorrections.isPending ? "Saving..." : "Save"}
              </button>
            </div>
          )}
        </div>
        <PlaybackControls
          centerTimestamp={centerTimestamp}
          isPlaying={isPlaying}
          speed={speed}
          onPlayPause={togglePlay}
          onSkipBack={NOOP}
          onSkipForward={NOOP}
          onSpeedChange={setSpeed}
          onZoomIn={zoomIn}
          onZoomOut={zoomOut}
          labelModeEnabled={false}
          labelModeActive={false}
          overlayMode="off"
          onToggleDetection={NOOP}
          onToggleVocalization={NOOP}
          hasVocalizationData={false}
          freqRange={[0, 3000]}
          regionEditMode={editMode}
          regionEditEnabled={job.status === "complete"}
          onRegionEditToggle={toggleEditMode}
          showRegionOverlay={showRegionOverlay}
          onToggleRegionOverlay={() => setShowRegionOverlay((v) => !v)}
          pendingCorrectionCount={pendingCorrections.size}
          onSaveCorrections={handleSaveCorrections}
          onCancelCorrections={handleCancelEdit}
          isSavingCorrections={saveCorrections.isPending}
        />
      </div>
    </div>
  );
}
