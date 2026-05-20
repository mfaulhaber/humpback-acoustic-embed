import type { EventEncoderTimelineEvent } from "@/api/sequenceModels";

export type EventEncoderYMode = "ridge" | "f0" | "peak";

export type EventDisplayBandSource =
  | "ridge"
  | "f0"
  | "band_limited_peak"
  | "peak"
  | "centroid"
  | "none";

export interface EventDisplayBand {
  centerFrequency: number;
  lowFrequency: number;
  highFrequency: number;
  source: EventDisplayBandSource;
  voiced: boolean;
  ridgeTrusted: boolean;
}

export interface EventDisplayBandOptions {
  voicedThreshold?: number;
  ridgeCoverageThreshold?: number;
  ridgeEnergyRatioThreshold?: number;
  fallbackBandSpanHz?: number;
}

export const DEFAULT_VOICED_THRESHOLD = 0.3;
const DEFAULT_RIDGE_COVERAGE_THRESHOLD = 0.35;
const DEFAULT_RIDGE_ENERGY_RATIO_THRESHOLD = 0.01;
const DEFAULT_FALLBACK_BAND_SPAN_HZ = 20;
const SPECTRAL_ENVELOPE_HIGH_BAND_THRESHOLD = 0.6;
const SPECTRAL_ENVELOPE_TONAL_HIGH_BAND_THRESHOLD = 0.45;
const SPECTRAL_ENVELOPE_TONAL_ENTROPY_THRESHOLD = 0.92;
const SPECTRAL_ENVELOPE_CENTROID_MARGIN_HZ = 500;
const SPECTRAL_ENVELOPE_MIN_BANDWIDTH_HZ = 1000;

export function hasRidgeFrequencyDescriptors(
  events: EventEncoderTimelineEvent[],
): boolean {
  return events.some((event) => {
    const values = event.descriptor_values;
    return (
      numeric(values.ridge_median_frequency) != null ||
      numeric(values.ridge_low_frequency) != null ||
      numeric(values.ridge_high_frequency) != null
    );
  });
}

export function isEventVoiced(
  event: EventEncoderTimelineEvent,
  voicedThreshold = DEFAULT_VOICED_THRESHOLD,
): boolean {
  return (numeric(event.descriptor_values.voicing_fraction) ?? 0) > voicedThreshold;
}

export function resolveEventDisplayBand(
  event: EventEncoderTimelineEvent,
  mode: EventEncoderYMode,
  options: EventDisplayBandOptions = {},
): EventDisplayBand {
  const values = event.descriptor_values;
  const voicedThreshold = options.voicedThreshold ?? DEFAULT_VOICED_THRESHOLD;
  const voiced = isEventVoiced(event, voicedThreshold);
  if (mode === "ridge") {
    const ridge = trustedRidgeBand(event, options, voiced);
    if (ridge) return ridge;
    return fallbackBand(event, options, voiced);
  }

  if (mode === "f0") {
    const medianF0 = numeric(values.median_f0);
    const peak = numeric(values.peak_frequency) ?? 0;
    return centeredBand(
      voiced && medianF0 != null ? medianF0 : peak,
      numeric(values.f0_range),
      voiced ? "f0" : "peak",
      voiced,
      false,
      options,
    );
  }

  return centeredBand(
    numeric(values.peak_frequency) ?? 0,
    numeric(values.f0_range),
    "peak",
    voiced,
    false,
    options,
  );
}

function trustedRidgeBand(
  event: EventEncoderTimelineEvent,
  options: EventDisplayBandOptions,
  voiced: boolean,
): EventDisplayBand | null {
  const values = event.descriptor_values;
  const median = positive(values.ridge_median_frequency);
  const low = positive(values.ridge_low_frequency);
  const high = positive(values.ridge_high_frequency);
  const coverage = numeric(values.ridge_coverage) ?? 0;
  const energyRatio = numeric(values.ridge_energy_ratio) ?? 0;
  const coverageThreshold =
    options.ridgeCoverageThreshold ?? DEFAULT_RIDGE_COVERAGE_THRESHOLD;
  const energyThreshold =
    options.ridgeEnergyRatioThreshold ?? DEFAULT_RIDGE_ENERGY_RATIO_THRESHOLD;
  if (
    median == null ||
    low == null ||
    high == null ||
    coverage < coverageThreshold ||
    energyRatio < energyThreshold ||
    high < low
  ) {
    return null;
  }
  const lowFrequency = Math.min(low, median);
  const ridgeHighFrequency = Math.max(high, median);

  return {
    centerFrequency: median,
    lowFrequency,
    highFrequency: expandedSpectralEnvelopeHighFrequency(
      values,
      ridgeHighFrequency,
    ),
    source: "ridge",
    voiced,
    ridgeTrusted: true,
  };
}

function expandedSpectralEnvelopeHighFrequency(
  values: Record<string, number>,
  ridgeHighFrequency: number,
): number {
  const spectralCentroid = positive(values.spectral_centroid);
  const bandwidth = positive(values.bandwidth);
  const highBandEnergy = numeric(values.high_band_energy_ratio) ?? 0;
  const spectralEntropy = numeric(values.spectral_entropy);
  const hasStrongHighBandEnergy =
    highBandEnergy >= SPECTRAL_ENVELOPE_HIGH_BAND_THRESHOLD;
  const hasTonalHighBandEnergy =
    highBandEnergy >= SPECTRAL_ENVELOPE_TONAL_HIGH_BAND_THRESHOLD &&
    spectralEntropy != null &&
    spectralEntropy <= SPECTRAL_ENVELOPE_TONAL_ENTROPY_THRESHOLD;
  if (
    spectralCentroid == null ||
    bandwidth == null ||
    (!hasStrongHighBandEnergy && !hasTonalHighBandEnergy) ||
    bandwidth < SPECTRAL_ENVELOPE_MIN_BANDWIDTH_HZ ||
    spectralCentroid < ridgeHighFrequency + SPECTRAL_ENVELOPE_CENTROID_MARGIN_HZ
  ) {
    return ridgeHighFrequency;
  }
  return spectralCentroid;
}

function fallbackBand(
  event: EventEncoderTimelineEvent,
  options: EventDisplayBandOptions,
  voiced: boolean,
): EventDisplayBand {
  const values = event.descriptor_values;
  if (voiced) {
    const medianF0 = positive(values.median_f0);
    if (medianF0 != null) {
      return centeredBand(
        medianF0,
        numeric(values.f0_range),
        "f0",
        voiced,
        false,
        options,
      );
    }
  }

  const bandPeak = positive(values.band_limited_peak_frequency);
  if (bandPeak != null) {
    return centeredBand(bandPeak, null, "band_limited_peak", voiced, false, options);
  }

  const peak = positive(values.peak_frequency);
  if (peak != null) {
    return centeredBand(peak, numeric(values.f0_range), "peak", voiced, false, options);
  }

  const centroid = positive(values.spectral_centroid);
  if (centroid != null) {
    return centeredBand(centroid, null, "centroid", voiced, false, options);
  }

  return centeredBand(0, null, "none", voiced, false, options);
}

function centeredBand(
  center: number,
  span: number | null | undefined,
  source: EventDisplayBandSource,
  voiced: boolean,
  ridgeTrusted: boolean,
  options: EventDisplayBandOptions,
): EventDisplayBand {
  const finiteCenter = numeric(center) ?? 0;
  const fallbackSpan = options.fallbackBandSpanHz ?? DEFAULT_FALLBACK_BAND_SPAN_HZ;
  const spanHz = Math.max(0, numeric(span) ?? fallbackSpan);
  return {
    centerFrequency: finiteCenter,
    lowFrequency: Math.max(0, finiteCenter - spanHz / 2),
    highFrequency: finiteCenter + spanHz / 2,
    source,
    voiced,
    ridgeTrusted,
  };
}

function positive(value: unknown): number | null {
  const result = numeric(value);
  return result != null && result > 0 ? result : null;
}

function numeric(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}
