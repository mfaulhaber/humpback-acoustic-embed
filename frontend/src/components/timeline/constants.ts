import type { ZoomLevel } from "@/api/types";

export const ZOOM_LEVELS: ZoomLevel[] = ["24h", "6h", "1h", "15m", "5m", "1m"];

export const TILE_DURATION: Record<ZoomLevel, number> = {
  "24h": 86400,
  "6h": 21600,
  "1h": 600,
  "15m": 150,
  "5m": 50,
  "1m": 10,
};

export const VIEWPORT_SPAN: Record<ZoomLevel, number> = {
  "24h": 86400,
  "6h": 21600,
  "1h": 3600,
  "15m": 900,
  "5m": 300,
  "1m": 60,
};

export const TILE_WIDTH_PX = 512;
export const TILE_HEIGHT_PX = 256;

export const PIXELS_PER_SEC: Record<ZoomLevel, number> = {
  "24h": TILE_WIDTH_PX / 86400,
  "6h": TILE_WIDTH_PX / 21600,
  "1h": TILE_WIDTH_PX / 600,
  "15m": TILE_WIDTH_PX / 150,
  "5m": TILE_WIDTH_PX / 50,
  "1m": TILE_WIDTH_PX / 10,
};

export const COLORS = {
  bg: "#060d14",
  bgDark: "#040810",
  text: "#a0c8c0",
  textMuted: "#3a6a60",
  textBright: "#5a9a80",
  accent: "#70e0c0",
  accentDim: "#40a080",
  border: "#1a3040",
  headerBg: "#0a1520",
  labelHumpback: "rgba(64, 224, 192, 0.15)",
  labelOrca: "rgba(224, 176, 64, 0.15)",
  labelShip: "rgba(224, 64, 64, 0.15)",
  labelBackground: "rgba(160, 160, 160, 0.15)",
} as const;

export const LABEL_COLORS = {
  humpback: { fill: "rgba(234, 179, 8, 0.45)", hover: "rgba(234, 179, 8, 0.65)", border: "rgb(234, 179, 8)" },
  orca: { fill: "rgba(249, 115, 22, 0.45)", hover: "rgba(249, 115, 22, 0.65)", border: "rgb(249, 115, 22)" },
  ship: { fill: "rgba(100, 149, 237, 0.35)", hover: "rgba(100, 149, 237, 0.55)", border: "rgb(100, 149, 237)" },
  background: { fill: "rgba(156, 163, 175, 0.30)", hover: "rgba(156, 163, 175, 0.50)", border: "rgb(156, 163, 175)" },
} as const;

export type LabelType = keyof typeof LABEL_COLORS;

export const CONFIDENCE_GRADIENT = [
  [0.0, "#0a1a0a"],
  [0.3, "#2a5a20"],
  [0.5, "#60a020"],
  [0.7, "#a0c820"],
  [0.85, "#d0e040"],
  [1.0, "#f0f060"],
] as const;

export const VOCALIZATION_BAR = {
  fill: "rgba(168, 130, 220, 0.40)",
  hover: "rgba(168, 130, 220, 0.60)",
} as const;

export const VOCALIZATION_BADGE_PALETTE = [
  "#e879f9", // fuchsia
  "#38bdf8", // sky
  "#fb923c", // orange
  "#4ade80", // green
  "#f87171", // red
  "#facc15", // yellow
  "#a78bfa", // violet
  "#2dd4bf", // teal
] as const;

export const CROSSFADE_DURATION_MS = 300;
export const AUDIO_PREFETCH_SEC = 300;
export const AUDIO_FORMAT = "mp3";
export const FREQ_AXIS_WIDTH_PX = 44;
