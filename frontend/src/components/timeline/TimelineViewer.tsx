// frontend/src/components/timeline/TimelineViewer.tsx
import { useState, useCallback, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import type { ZoomLevel } from "@/api/types";
import { useTimelineConfidence, useTimelineDetections } from "@/hooks/queries/useTimeline";
import { useHydrophoneDetectionJobs } from "@/hooks/queries/useClassifier";
import { timelineAudioUrl } from "@/api/client";
import { TimelineHeader } from "./TimelineHeader";
import { ZoomSelector } from "./ZoomSelector";
import { PlaybackControls } from "./PlaybackControls";
import { Minimap } from "./Minimap";
import { SpectrogramViewport } from "./SpectrogramViewport";
import { ZOOM_LEVELS, VIEWPORT_SPAN, COLORS, AUDIO_PREFETCH_SEC, AUDIO_FORMAT } from "./constants";

export function TimelineViewer() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data: jobs } = useHydrophoneDetectionJobs(0);
  const job = jobs?.find((j) => j.id === jobId);

  // Core state
  const [centerTimestamp, setCenterTimestamp] = useState<number>(0);
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>("1h");
  const [isPlaying, setIsPlaying] = useState(false);
  const [freqRange, setFreqRange] = useState<[number, number]>([0, 3000]);
  const [showLabels, setShowLabels] = useState(false);
  const [speed, setSpeed] = useState(1);

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

  // Play/pause
  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // Skip to next/prev detection
  // Note: detection row start_sec/end_sec are canonical snapped bounds (timeline-
  // absolute for hydrophone jobs since the detection worker writes them that way).
  // If start_sec is file-relative for your data, you'll need to add job.start_timestamp.
  // Check your detection content endpoint output to confirm.
  const skipForward = useCallback(() => {
    if (!detections || !job) return;
    const jobStart = job.start_timestamp ?? 0;
    const next = detections.find((d) => (jobStart + d.start_sec) > centerTimestamp);
    if (next) setCenterTimestamp(jobStart + next.start_sec);
  }, [detections, centerTimestamp, job]);

  const skipBack = useCallback(() => {
    if (!detections || !job) return;
    const jobStart = job.start_timestamp ?? 0;
    const prev = [...detections].reverse().find((d) => (jobStart + d.end_sec) < centerTimestamp);
    if (prev) setCenterTimestamp(jobStart + prev.start_sec);
  }, [detections, centerTimestamp, job]);

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

  if (!jobId || !job) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: COLORS.bg, color: COLORS.text }}>
        Loading...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen font-mono text-xs" style={{ background: COLORS.bg, color: COLORS.text }}>
      <TimelineHeader
        hydrophone={job.hydrophone_name ?? job.hydrophone_id ?? ""}
        startTimestamp={job.start_timestamp ?? 0}
        endTimestamp={job.end_timestamp ?? 0}
        showLabels={showLabels}
        onToggleLabels={() => setShowLabels((s) => !s)}
        freqRange={freqRange}
        onFreqRangeChange={setFreqRange}
      />

      <Minimap
        scores={confidence?.scores ?? []}
        jobStart={job.start_timestamp ?? 0}
        jobEnd={job.end_timestamp ?? 0}
        centerTimestamp={centerTimestamp}
        viewportSpan={VIEWPORT_SPAN[zoomLevel]}
        onCenterChange={setCenterTimestamp}
      />

      {/* Main spectrogram viewport */}
      <div className="flex-1 flex flex-col mx-4 my-2 rounded relative overflow-hidden" style={{ background: COLORS.bgDark }}>
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
        />
      </div>

      {/* Double-buffered hidden audio elements for gapless playback */}
      <audio ref={audioRefA} preload="auto" style={{ display: "none" }} />
      <audio ref={audioRefB} preload="auto" style={{ display: "none" }} />

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
      />
    </div>
  );
}
