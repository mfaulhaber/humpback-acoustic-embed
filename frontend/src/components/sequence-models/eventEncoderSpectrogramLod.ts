export interface SpectrogramLod {
  key: string;
  tileDuration: number;
  tileWidthPx: number;
}

export interface ChooseSpectrogramLodOptions {
  viewportSpan: number;
  viewportWidth: number;
  currentKey?: string | null;
  lods?: SpectrogramLod[];
  hysteresisRatio?: number;
}

const DEFAULT_TILE_WIDTH_PX = 512;
const DEFAULT_HYSTERESIS_RATIO = 1.2;

export const EVENT_ENCODER_SPECTROGRAM_LODS: SpectrogramLod[] = [
  { key: "24h", tileDuration: 86400, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
  { key: "6h", tileDuration: 21600, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
  { key: "1h", tileDuration: 600, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
  { key: "15m", tileDuration: 150, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
  { key: "5m", tileDuration: 50, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
  { key: "1m", tileDuration: 10, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
  { key: "30s", tileDuration: 5, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
  { key: "10s", tileDuration: 2, tileWidthPx: DEFAULT_TILE_WIDTH_PX },
];

export function chooseSpectrogramLod({
  viewportSpan,
  viewportWidth,
  currentKey,
  lods = EVENT_ENCODER_SPECTROGRAM_LODS,
  hysteresisRatio = DEFAULT_HYSTERESIS_RATIO,
}: ChooseSpectrogramLodOptions): SpectrogramLod {
  const validLods = lods.filter(
    (lod) => lod.tileDuration > 0 && lod.tileWidthPx > 0,
  );
  if (!validLods.length) {
    throw new Error("At least one spectrogram LOD is required.");
  }

  const current = currentKey
    ? validLods.find((lod) => lod.key === currentKey)
    : undefined;
  if (viewportSpan <= 0 || viewportWidth <= 0) {
    return current ?? validLods[0];
  }

  const targetSecondsPerPixel = viewportSpan / viewportWidth;
  const ranked = validLods
    .map((lod) => ({
      lod,
      secondsPerPixel: nativeSecondsPerPixel(lod),
    }))
    .sort((a, b) => b.secondsPerPixel - a.secondsPerPixel);
  const best =
    ranked.find((candidate) => candidate.secondsPerPixel <= targetSecondsPerPixel) ??
    ranked[ranked.length - 1];
  const bestError = lodError(targetSecondsPerPixel, best.secondsPerPixel);
  if (!current || current.key === best.lod.key) {
    return best.lod;
  }

  const currentError = lodError(
    targetSecondsPerPixel,
    nativeSecondsPerPixel(current),
  );
  if (currentError <= bestError * hysteresisRatio) {
    return current;
  }
  return best.lod;
}

function nativeSecondsPerPixel(lod: SpectrogramLod): number {
  return lod.tileDuration / lod.tileWidthPx;
}

function lodError(targetSecondsPerPixel: number, lodSecondsPerPixel: number) {
  const target = Math.max(targetSecondsPerPixel, Number.EPSILON);
  const candidate = Math.max(lodSecondsPerPixel, Number.EPSILON);
  return Math.abs(Math.log2(candidate / target));
}
