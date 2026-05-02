import React, { createContext, forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useReducer, useRef } from "react";
import type { TimelineContextValue, TimelinePlaybackHandle, TimelineProviderProps, ZoomPreset } from "./types";
import { usePlayback } from "./usePlayback";

export const TimelineContext = createContext<TimelineContextValue | null>(null);

interface State {
  centerTimestamp: number;
  zoomLevel: number;
  isPlaying: boolean;
  isDraggingTimeline: boolean;
  speed: number;
  viewportWidth: number;
  viewportHeight: number;
  playbackEpoch: number | null;
}

type Action =
  | { type: "PAN"; center: number }
  | { type: "SET_ZOOM"; index: number }
  | { type: "SET_PLAYING"; playing: boolean }
  | { type: "SET_DRAGGING"; dragging: boolean }
  | { type: "SET_SPEED"; speed: number }
  | { type: "SEEK"; epoch: number }
  | { type: "SET_VIEWPORT"; width: number; height: number }
  | { type: "SET_PLAYBACK_EPOCH"; epoch: number | null };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "PAN":
      return { ...state, centerTimestamp: action.center };
    case "SET_ZOOM":
      return { ...state, zoomLevel: action.index };
    case "SET_PLAYING":
      return { ...state, isPlaying: action.playing };
    case "SET_DRAGGING":
      return { ...state, isDraggingTimeline: action.dragging };
    case "SET_SPEED":
      return { ...state, speed: action.speed };
    case "SEEK":
      return { ...state, centerTimestamp: action.epoch };
    case "SET_VIEWPORT":
      return { ...state, viewportWidth: action.width, viewportHeight: action.height };
    case "SET_PLAYBACK_EPOCH":
      return { ...state, playbackEpoch: action.epoch };
  }
}

function findDefaultZoomIndex(zoomLevels: ZoomPreset[], defaultZoom?: string): number {
  if (!defaultZoom) return 0;
  const idx = zoomLevels.findIndex((z) => z.key === defaultZoom);
  return idx >= 0 ? idx : 0;
}

export const TimelineProvider = forwardRef<TimelinePlaybackHandle, TimelineProviderProps>(function TimelineProvider({
  jobStart,
  jobEnd,
  zoomLevels,
  defaultZoom,
  playback: playbackMode,
  audioUrlBuilder,
  disableKeyboardShortcuts,
  scrollOnPlayback = true,
  onZoomChange,
  onPlayStateChange,
  children,
}, ref) {
  const defaultIndex = useMemo(() => findDefaultZoomIndex(zoomLevels, defaultZoom), [zoomLevels, defaultZoom]);

  const [state, dispatch] = useReducer(reducer, {
    centerTimestamp: jobStart + (jobEnd - jobStart) / 2,
    zoomLevel: defaultIndex,
    isPlaying: false,
    isDraggingTimeline: false,
    speed: 1,
    viewportWidth: 0,
    viewportHeight: 0,
    playbackEpoch: null,
  });

  const activePreset = zoomLevels[state.zoomLevel] ?? zoomLevels[0];
  const viewportSpan = activePreset.span;
  const viewStart = state.centerTimestamp - viewportSpan / 2;
  const viewEnd = state.centerTimestamp + viewportSpan / 2;
  const pxPerSec = state.viewportWidth > 0 ? state.viewportWidth / viewportSpan : 1;

  const clampCenter = useCallback(
    (c: number) => Math.max(jobStart, Math.min(jobEnd, c)),
    [jobStart, jobEnd],
  );

  const pan = useCallback(
    (center: number) => dispatch({ type: "PAN", center: clampCenter(center) }),
    [clampCenter],
  );

  const dragPanRef = useRef<{ startX: number; startCenter: number } | null>(null);

  const beginDragPan = useCallback(
    (clientX: number) => {
      if (state.isPlaying) return false;
      dragPanRef.current = {
        startX: clientX,
        startCenter: state.centerTimestamp,
      };
      dispatch({ type: "SET_DRAGGING", dragging: true });
      return true;
    },
    [state.centerTimestamp, state.isPlaying],
  );

  const updateDragPan = useCallback(
    (clientX: number) => {
      if (!dragPanRef.current || state.isPlaying) return;
      const dx = clientX - dragPanRef.current.startX;
      const dt = dx / pxPerSec;
      pan(dragPanRef.current.startCenter - dt);
    },
    [pan, pxPerSec, state.isPlaying],
  );

  const endDragPan = useCallback(() => {
    if (!dragPanRef.current) return;
    dragPanRef.current = null;
    dispatch({ type: "SET_DRAGGING", dragging: false });
  }, []);

  const onZoomChangeRef = useRef(onZoomChange);
  onZoomChangeRef.current = onZoomChange;

  const setZoomLevel = useCallback(
    (index: number) => {
      if (index >= 0 && index < zoomLevels.length) {
        dispatch({ type: "SET_ZOOM", index });
        onZoomChangeRef.current?.(zoomLevels[index].key);
      }
    },
    [zoomLevels],
  );

  const zoomIn = useCallback(() => {
    setZoomLevel(Math.min(state.zoomLevel + 1, zoomLevels.length - 1));
  }, [state.zoomLevel, zoomLevels.length, setZoomLevel]);

  const zoomOut = useCallback(() => {
    setZoomLevel(Math.max(state.zoomLevel - 1, 0));
  }, [state.zoomLevel, setZoomLevel]);

  const setSpeed = useCallback((speed: number) => dispatch({ type: "SET_SPEED", speed }), []);

  const seekTo = useCallback(
    (epoch: number) => dispatch({ type: "SEEK", epoch: clampCenter(epoch) }),
    [clampCenter],
  );

  const setViewportDimensions = useCallback(
    (width: number, height: number) => dispatch({ type: "SET_VIEWPORT", width, height }),
    [],
  );

  const onPlayStateChangeRef = useRef(onPlayStateChange);
  onPlayStateChangeRef.current = onPlayStateChange;

  const playbackHandle = usePlayback({
    mode: playbackMode,
    audioUrlBuilder,
    speed: state.speed,
    onTimeUpdate: (epoch) => {
      dispatch({ type: "SET_PLAYBACK_EPOCH", epoch });
      if (scrollOnPlayback) {
        dispatch({ type: "PAN", center: clampCenter(epoch) });
      }
    },
    onEnded: () => {
      dispatch({ type: "SET_PLAYING", playing: false });
      dispatch({ type: "SET_PLAYBACK_EPOCH", epoch: null });
      onPlayStateChangeRef.current?.(false);
    },
  });

  const play = useCallback(
    (startEpoch?: number, duration?: number) => {
      const start = startEpoch ?? state.centerTimestamp;
      playbackHandle.play(start, duration);
      dispatch({ type: "SET_PLAYING", playing: true });
      dispatch({ type: "SET_PLAYBACK_EPOCH", epoch: start });
      onPlayStateChangeRef.current?.(true);
    },
    [state.centerTimestamp, playbackHandle],
  );

  const pause = useCallback(() => {
    playbackHandle.pause();
    dispatch({ type: "SET_PLAYING", playing: false });
    dispatch({ type: "SET_PLAYBACK_EPOCH", epoch: null });
    onPlayStateChangeRef.current?.(false);
  }, [playbackHandle]);

  const togglePlay = useCallback(() => {
    if (state.isPlaying) {
      pause();
    } else {
      play();
    }
  }, [state.isPlaying, play, pause]);

  useImperativeHandle(ref, () => ({
    play: (startEpoch: number, duration?: number) => play(startEpoch, duration),
    pause,
    seekTo,
    get isPlaying() { return state.isPlaying; },
  }), [play, pause, seekTo, state.isPlaying]);

  // End drag panning if the pointer is released outside the active panel.
  useEffect(() => {
    const handleGlobalMouseUp = () => endDragPan();
    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, [endDragPan]);

  // Keep panning when the pointer leaves the panel during a drag.
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => updateDragPan(e.clientX);
    window.addEventListener("mousemove", handleGlobalMouseMove);
    return () => window.removeEventListener("mousemove", handleGlobalMouseMove);
  }, [updateDragPan]);

  // Keyboard shortcuts (disabled when consumer manages its own)
  useEffect(() => {
    if (disableKeyboardShortcuts) return;

    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlay();
          break;
        case "+":
        case "=":
          e.preventDefault();
          zoomIn();
          break;
        case "-":
          e.preventDefault();
          zoomOut();
          break;
        case "ArrowLeft":
          e.preventDefault();
          pan(clampCenter(state.centerTimestamp - viewportSpan * 0.1));
          break;
        case "ArrowRight":
          e.preventDefault();
          pan(clampCenter(state.centerTimestamp + viewportSpan * 0.1));
          break;
      }
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [disableKeyboardShortcuts, togglePlay, zoomIn, zoomOut, pan, clampCenter, state.centerTimestamp, viewportSpan]);

  const value: TimelineContextValue = useMemo(
    () => ({
      ...state,
      viewStart,
      viewEnd,
      pxPerSec,
      viewportSpan,
      activePreset,
      jobStart,
      jobEnd,
      zoomLevels,
      pan,
      setZoomLevel,
      zoomIn,
      zoomOut,
      play,
      pause,
      togglePlay,
      beginDragPan,
      updateDragPan,
      endDragPan,
      seekTo,
      setSpeed,
      setViewportDimensions,
    }),
    [
      state,
      viewStart,
      viewEnd,
      pxPerSec,
      viewportSpan,
      activePreset,
      jobStart,
      jobEnd,
      zoomLevels,
      pan,
      setZoomLevel,
      zoomIn,
      zoomOut,
      play,
      pause,
      togglePlay,
      beginDragPan,
      updateDragPan,
      endDragPan,
      seekTo,
      setSpeed,
      setViewportDimensions,
    ],
  );

  return (
    <TimelineContext.Provider value={value}>
      {children}
    </TimelineContext.Provider>
  );
});
