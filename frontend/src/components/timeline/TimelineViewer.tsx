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
import { ZOOM_LEVELS, VIEWPORT_SPAN, COLORS, AUDIO_PREFETCH_SEC } from "./constants";

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

  // Audio element ref
  const audioRef = useRef<HTMLAudioElement>(null);

  // Initialize center to job midpoint
  useEffect(() => {
    if (job?.start_timestamp && job?.end_timestamp && centerTimestamp === 0) {
      setCenterTimestamp(
        job.start_timestamp + (job.end_timestamp - job.start_timestamp) / 2,
      );
    }
  }, [job, centerTimestamp]);

  // Start/stop audio when isPlaying or centerTimestamp changes at play start
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.src = timelineAudioUrl(jobId ?? "", centerTimestamp, AUDIO_PREFETCH_SEC);
      audio.playbackRate = speed;
      audio.play().catch(() => {
        // Ignore AbortError when src is changed rapidly
      });
    } else {
      audio.pause();
    }
    // Only re-run when isPlaying toggles, not on every centerTimestamp tick
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying]);

  // Sync playback rate when speed changes
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.playbackRate = speed;
  }, [speed]);

  // RAF loop: advance centerTimestamp while playing
  useEffect(() => {
    if (!isPlaying) return;
    let lastTime = performance.now();
    let rafId: number;

    const tick = (now: number) => {
      const dt = (now - lastTime) / 1000; // seconds
      lastTime = now;
      setCenterTimestamp((prev) => {
        const next = prev + dt * speed;
        return Math.min(next, job?.end_timestamp ?? next);
      });
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, speed, job]);

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
      <div className="flex-1 mx-4 my-2 rounded relative overflow-hidden" style={{ background: COLORS.bgDark }}>
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
          onCenterChange={setCenterTimestamp}
          onPan={handlePan}
        />
      </div>

      {/* Hidden audio element for playback */}
      <audio ref={audioRef} style={{ display: "none" }} />

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
