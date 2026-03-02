import { useRef, useState, useEffect, useCallback } from "react";
import { Play, Pause, Volume2, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { audioDownloadUrl, audioWindowUrl } from "@/api/client";
import { formatTime } from "@/utils/format";


interface WindowPlayerProps {
  audioId: string;
  windows: number[];
  windowSizeSeconds: number;
  activeWindowIndex?: number;
  onWindowChange?: (windowIndex: number) => void;
  maxChips?: number;
}

export function WindowPlayer({
  audioId,
  windows,
  windowSizeSeconds,
  activeWindowIndex,
  onWindowChange,
  maxChips = 40,
}: WindowPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [listPos, setListPos] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(windowSizeSeconds);
  const [volume, setVolume] = useState(0.8);

  // Refs to avoid stale closures in event handlers
  const windowsRef = useRef(windows);
  const listPosRef = useRef(listPos);
  windowsRef.current = windows;
  listPosRef.current = listPos;

  // Sync listPos when activeWindowIndex prop changes from outside
  useEffect(() => {
    if (activeWindowIndex != null) {
      const pos = windows.indexOf(activeWindowIndex);
      if (pos !== -1 && pos !== listPosRef.current) {
        setListPos(pos);
      }
    }
  }, [activeWindowIndex, windows]);

  // Reset when audio changes; preload src so loadedmetadata fires and
  // the true duration displays before the user clicks play.
  useEffect(() => {
    setListPos(0);
    setPlaying(false);
    setCurrentTime(0);
    setAudioDuration(windowSizeSeconds);
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      if (windows.length > 0) {
        const win = windows[0];
        audio.src = audioWindowUrl(audioId, win * windowSizeSeconds, windowSizeSeconds);
      } else {
        audio.removeAttribute("src");
      }
    }
  }, [audioId, windows, windowSizeSeconds]);

  const playAtPos = useCallback(
    (pos: number) => {
      const audio = audioRef.current;
      if (!audio || pos < 0 || pos >= windowsRef.current.length) return;
      const win = windowsRef.current[pos];
      const start = win * windowSizeSeconds;
      audio.src = audioWindowUrl(audioId, start, windowSizeSeconds);
      audio.volume = volume;
      audio.play().catch(() => {});
      setListPos(pos);
      setCurrentTime(0);
      setAudioDuration(windowSizeSeconds);
      onWindowChange?.(win);
    },
    [audioId, windowSizeSeconds, volume, onWindowChange],
  );

  // Audio element event listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onLoadedMetadata = () => {
      if (audio.duration && isFinite(audio.duration)) {
        setAudioDuration(audio.duration);
      }
    };
    const onEnded = () => {
      setPlaying(false);
      setCurrentTime(0);
    };
    const onTimeUpdate = () => setCurrentTime(audio.currentTime);

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("timeupdate", onTimeUpdate);
    return () => {
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("timeupdate", onTimeUpdate);
    };
  }, [audioId, windowSizeSeconds, volume, onWindowChange]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      if (!audio.src || audio.src === window.location.href) {
        playAtPos(listPos);
      } else {
        audio.play().catch(() => {});
      }
    } else {
      audio.pause();
    }
  }, [listPos, playAtPos]);

  const currentWindow = windows[listPos] ?? 0;
  const effectiveDuration = audioDuration > 0 ? audioDuration : windowSizeSeconds;
  const progress = effectiveDuration > 0 ? (currentTime / effectiveDuration) * 100 : 0;
  const displayedCount = Math.min(windows.length, maxChips);

  return (
    <div className="space-y-2">
      <audio ref={audioRef} preload="metadata" />
      <div className="flex items-center gap-2 flex-wrap">
        {/* Window chips */}
        <div className="flex gap-0.5 flex-wrap">
          {windows.slice(0, displayedCount).map((win, i) => (
            <Button
              key={win}
              variant="outline"
              className="h-7 px-2 gap-1.5"
              onClick={() => {
                if (i === listPos && playing) {
                  audioRef.current?.pause();
                } else if (i === listPos) {
                  togglePlay();
                } else {
                  playAtPos(i);
                }
              }}
            >
              <span className="font-mono text-[14px] leading-none">{win}</span>
              {i === listPos && playing ? (
                <Pause className="h-3.5 w-3.5" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
            </Button>
          ))}
          {windows.length > maxChips && (
            <span className="text-xs text-muted-foreground self-center ml-1">
              +{windows.length - maxChips}
            </span>
          )}
        </div>

        {/* Volume & download */}
        <div className="flex items-center gap-1 ml-auto">
          <Volume2 className="h-3.5 w-3.5 text-muted-foreground" />
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={volume}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setVolume(v);
              if (audioRef.current) audioRef.current.volume = v;
            }}
            className="w-20 h-1 accent-primary"
          />
          <a href={audioDownloadUrl(audioId)} download className="ml-2">
            <Download className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
          </a>
        </div>
      </div>

      {/* Progress bar (within current window) */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>{formatTime(currentTime)}</span>
        <div className="flex-1 relative h-4 bg-secondary rounded overflow-hidden">
          <div
            className="absolute top-0 left-0 h-full bg-primary/30 rounded"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
          <div
            className="absolute top-0 h-full w-0.5 bg-primary rounded"
            style={{ left: `${Math.min(progress, 100)}%` }}
          />
        </div>
        <span>{formatTime(effectiveDuration)}</span>
      </div>
    </div>
  );
}
