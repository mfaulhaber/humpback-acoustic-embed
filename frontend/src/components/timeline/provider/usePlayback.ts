import { useRef, useCallback, useEffect, useState } from "react";

export interface PlaybackHandle {
  play: (startEpoch: number, duration?: number) => void;
  pause: () => void;
  isPlaying: boolean;
  currentTime: number;
}

interface UsePlaybackOptions {
  mode: "gapless" | "slice";
  audioUrlBuilder: (startEpoch: number, durationSec: number) => string;
  speed: number;
  onTimeUpdate?: (epoch: number) => void;
  onEnded?: () => void;
}

export function usePlayback({
  mode,
  audioUrlBuilder,
  speed,
  onTimeUpdate,
  onEnded,
}: UsePlaybackOptions): PlaybackHandle {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const audioARef = useRef<HTMLAudioElement | null>(null);
  const audioBRef = useRef<HTMLAudioElement | null>(null);
  const activeRef = useRef<"A" | "B">("A");
  const startEpochRef = useRef(0);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!audioARef.current) {
      audioARef.current = new Audio();
      audioARef.current.preload = "auto";
    }
    if (mode === "gapless" && !audioBRef.current) {
      audioBRef.current = new Audio();
      audioBRef.current.preload = "auto";
    }
    return () => {
      cancelAnimationFrame(rafRef.current);
      audioARef.current?.pause();
      audioBRef.current?.pause();
    };
  }, [mode]);

  useEffect(() => {
    const a = audioARef.current;
    const b = audioBRef.current;
    if (a) a.playbackRate = speed;
    if (b) b.playbackRate = speed;
  }, [speed]);

  const getActiveAudio = useCallback(() => {
    return activeRef.current === "A" ? audioARef.current : audioBRef.current;
  }, []);

  const tick = useCallback(() => {
    const audio = getActiveAudio();
    if (audio && !audio.paused) {
      const elapsed = audio.currentTime;
      const epoch = startEpochRef.current + elapsed;
      setCurrentTime(epoch);
      onTimeUpdate?.(epoch);
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [getActiveAudio, onTimeUpdate]);

  const play = useCallback(
    (startEpoch: number, duration?: number) => {
      const audio = getActiveAudio();
      if (!audio) return;

      startEpochRef.current = startEpoch;
      const dur = duration ?? 300;
      const url = audioUrlBuilder(startEpoch, dur);

      audio.src = url;
      audio.currentTime = 0;
      audio.playbackRate = speed;

      const handleEnded = () => {
        if (mode === "gapless") {
          const nextStart = startEpoch + dur;
          activeRef.current = activeRef.current === "A" ? "B" : "A";
          play(nextStart, dur);
        } else {
          setIsPlaying(false);
          onEnded?.();
        }
      };

      audio.onended = handleEnded;
      audio.play().then(() => {
        setIsPlaying(true);
        setCurrentTime(startEpoch);
        rafRef.current = requestAnimationFrame(tick);
      }).catch(() => {
        setIsPlaying(false);
      });

      if (mode === "gapless") {
        const other = activeRef.current === "A" ? audioBRef.current : audioARef.current;
        if (other) {
          const nextUrl = audioUrlBuilder(startEpoch + dur, dur);
          other.src = nextUrl;
          other.preload = "auto";
        }
      }
    },
    [audioUrlBuilder, speed, mode, getActiveAudio, tick, onEnded],
  );

  const pause = useCallback(() => {
    const audio = getActiveAudio();
    if (audio) audio.pause();
    cancelAnimationFrame(rafRef.current);
    setIsPlaying(false);
  }, [getActiveAudio]);

  return { play, pause, isPlaying, currentTime };
}
