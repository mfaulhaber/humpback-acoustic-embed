export interface ZoomPreset {
  key: string;
  span: number;
  tileDuration: number;
}

export const FULL_ZOOM: ZoomPreset[] = [
  { key: "24h", span: 86400, tileDuration: 86400 },
  { key: "6h", span: 21600, tileDuration: 21600 },
  { key: "1h", span: 3600, tileDuration: 600 },
  { key: "15m", span: 900, tileDuration: 150 },
  { key: "5m", span: 300, tileDuration: 50 },
  { key: "1m", span: 60, tileDuration: 10 },
];

export const REVIEW_ZOOM: ZoomPreset[] = [
  { key: "5m", span: 300, tileDuration: 50 },
  { key: "1m", span: 60, tileDuration: 10 },
  { key: "30s", span: 30, tileDuration: 10 },
  { key: "10s", span: 10, tileDuration: 10 },
];

export interface TimelineState {
  centerTimestamp: number;
  zoomLevel: number;
  isPlaying: boolean;
  speed: number;
  viewportWidth: number;
  viewportHeight: number;
}

export interface TimelineDerived {
  viewStart: number;
  viewEnd: number;
  pxPerSec: number;
  viewportSpan: number;
  activePreset: ZoomPreset;
}

export interface TimelineActions {
  pan: (center: number) => void;
  setZoomLevel: (index: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  play: (startEpoch?: number, duration?: number) => void;
  pause: () => void;
  togglePlay: () => void;
  seekTo: (epoch: number) => void;
  setSpeed: (speed: number) => void;
  setViewportDimensions: (width: number, height: number) => void;
}

export interface TimelineContextValue extends TimelineState, TimelineDerived, TimelineActions {
  jobStart: number;
  jobEnd: number;
  zoomLevels: ZoomPreset[];
}

export interface TimelinePlaybackHandle {
  play: (startEpoch: number, duration?: number) => void;
  pause: () => void;
  isPlaying: boolean;
}

export interface TimelineProviderProps {
  jobStart: number;
  jobEnd: number;
  zoomLevels: ZoomPreset[];
  defaultZoom?: string;
  playback: "gapless" | "slice";
  audioUrlBuilder: (startEpoch: number, durationSec: number) => string;
  disableKeyboardShortcuts?: boolean;
  onZoomChange?: (zoomKey: string) => void;
  onPlayStateChange?: (playing: boolean) => void;
  children: React.ReactNode;
}
