import { useRef, useState, useEffect, useCallback } from "react";
import { Play, Pause, Volume2, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { audioDownloadUrl, audioWindowUrl } from "@/api/client";
import { formatTime } from "@/utils/format";
import { cn } from "@/lib/utils";

interface AudioPlayerBarProps {
  audioId: string;
  totalWindows: number;
  windowSizeSeconds: number;
  duration: number;
  activeWindow: number;
  onWindowClick: (windowIndex: number) => void;
  maxChips?: number;
}

export function AudioPlayerBar({
  audioId,
  totalWindows,
  windowSizeSeconds,
  duration,
  activeWindow,
  onWindowClick,
  maxChips = 40,
}: AudioPlayerBarProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [playingWindow, setPlayingWindow] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(0.8);

  const selStart = activeWindow * windowSizeSeconds;
  const selEnd = Math.min(selStart + windowSizeSeconds, duration);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => {
      setPlaying(false);
      setPlayingWindow(null);
    };
    const onTimeUpdate = () => setCurrentTime(audio.currentTime);

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("timeupdate", onTimeUpdate);
    return () => {
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("timeupdate", onTimeUpdate);
    };
  }, []);

  const playWindow = useCallback(
    (idx: number) => {
      const audio = audioRef.current;
      if (!audio) return;
      const start = idx * windowSizeSeconds;
      audio.src = audioWindowUrl(audioId, start, windowSizeSeconds);
      audio.volume = volume;
      audio.play().catch(() => {});
      setPlayingWindow(idx);
      setCurrentTime(0);
      onWindowClick(idx);
    },
    [audioId, windowSizeSeconds, volume, onWindowClick],
  );

  const playFull = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.src = audioDownloadUrl(audioId);
    audio.volume = volume;
    audio.play().catch(() => {});
    setPlayingWindow(null);
    setCurrentTime(0);
  }, [audioId, volume]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      if (!audio.src) playWindow(activeWindow);
      else audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  }, [activeWindow, playWindow]);

  // Compute playhead position
  const playheadPos =
    duration > 0
      ? ((selStart + currentTime) / duration) * 100
      : 0;

  const regionLeft = duration > 0 ? (selStart / duration) * 100 : 0;
  const regionWidth = duration > 0 ? ((selEnd - selStart) / duration) * 100 : 0;

  const displayedWindows = Math.min(totalWindows, maxChips);

  return (
    <div className="space-y-2">
      <audio ref={audioRef} preload="metadata" />
      <div className="flex items-center gap-2 flex-wrap">
        <Button variant="outline" size="icon" className="h-7 w-7" onClick={togglePlay}>
          {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
        </Button>

        {/* Window chips */}
        <div className="flex gap-0.5 flex-wrap">
          {Array.from({ length: displayedWindows }, (_, i) => (
            <button
              key={i}
              onClick={() => playWindow(i)}
              className={cn(
                "px-1.5 py-0.5 text-xs rounded border transition-colors",
                i === activeWindow
                  ? "bg-primary text-primary-foreground border-primary"
                  : playingWindow === i
                    ? "bg-blue-100 border-blue-300 text-blue-800"
                    : "bg-secondary border-border hover:bg-accent",
              )}
            >
              {i}
            </button>
          ))}
          {totalWindows > maxChips && (
            <span className="text-xs text-muted-foreground self-center ml-1">
              +{totalWindows - maxChips}
            </span>
          )}
          <button
            onClick={playFull}
            className="px-2 py-0.5 text-xs rounded border bg-secondary border-border hover:bg-accent font-medium"
          >
            Full
          </button>
        </div>

        {/* Volume */}
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

      {/* Time track */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>{formatTime(selStart)}</span>
        <div className="flex-1 relative h-4 bg-secondary rounded">
          {/* Selected region */}
          <div
            className="absolute top-0 h-full bg-primary/20 rounded"
            style={{ left: `${regionLeft}%`, width: `${regionWidth}%` }}
          />
          {/* Playhead */}
          <div
            className="absolute top-0 h-full w-0.5 bg-primary rounded"
            style={{ left: `${Math.min(playheadPos, 100)}%` }}
          />
        </div>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}
