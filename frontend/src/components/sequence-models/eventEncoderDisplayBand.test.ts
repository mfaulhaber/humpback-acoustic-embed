import { describe, expect, it } from "vitest";

import type { EventEncoderTimelineEvent } from "@/api/sequenceModels";

import {
  hasRidgeFrequencyDescriptors,
  isEventVoiced,
  resolveEventDisplayBand,
} from "./eventEncoderDisplayBand";

function event(
  descriptorValues: Record<string, number>,
): EventEncoderTimelineEvent {
  return {
    event_id: "event-1",
    region_id: "region-1",
    source_sequence_key: "source",
    sequence_index: 1,
    start_timestamp: 10,
    end_timestamp: 11,
    token_id: 1,
    token_label: "T01",
    token_confidence: 0.9,
    distance_to_centroid: 0.1,
    second_centroid_distance: 0.2,
    descriptor_values: descriptorValues,
    descriptor_vector_values: {},
  };
}

describe("resolveEventDisplayBand", () => {
  it("uses trusted ridge bounds for ridge mode", () => {
    const band = resolveEventDisplayBand(
      event({
        ridge_median_frequency: 2400,
        ridge_low_frequency: 2100,
        ridge_high_frequency: 2800,
        ridge_coverage: 0.8,
        ridge_energy_ratio: 0.2,
        voicing_fraction: 0.1,
        median_f0: 0,
        peak_frequency: 80,
      }),
      "ridge",
    );

    expect(band.source).toBe("ridge");
    expect(band.ridgeTrusted).toBe(true);
    expect(band.centerFrequency).toBe(2400);
    expect(band.lowFrequency).toBe(2100);
    expect(band.highFrequency).toBe(2800);
  });

  it("trusts calibrated low-energy ridge bands from real whistle artifacts", () => {
    const band = resolveEventDisplayBand(
      event({
        ridge_median_frequency: 2898.4375,
        ridge_low_frequency: 2392.1875,
        ridge_high_frequency: 3260.9375,
        ridge_coverage: 1,
        ridge_energy_ratio: 0.010797901079058647,
        voicing_fraction: 1,
        median_f0: 71.22357940673828,
        f0_range: 3.7352724075317383,
        peak_frequency: 62.5,
        band_limited_peak_frequency: 3125,
      }),
      "ridge",
    );

    expect(band.source).toBe("ridge");
    expect(band.ridgeTrusted).toBe(true);
    expect(band.centerFrequency).toBe(2898.4375);
    expect(band.lowFrequency).toBe(2392.1875);
    expect(band.highFrequency).toBe(3260.9375);
  });

  it("expands trusted ridge bands for broad harmonic moan envelopes", () => {
    const band = resolveEventDisplayBand(
      event({
        ridge_median_frequency: 1015.625,
        ridge_low_frequency: 843.75,
        ridge_high_frequency: 1359.375,
        ridge_coverage: 1,
        ridge_energy_ratio: 0.0319729745388031,
        spectral_centroid: 3080.42138671875,
        bandwidth: 2224.228271484375,
        high_band_energy_ratio: 0.7930237054824829,
        voicing_fraction: 0.15555556118488312,
        median_f0: 70.81336212158203,
        f0_range: 3.3297667503356934,
        peak_frequency: 62.5,
        band_limited_peak_frequency: 1281.25,
      }),
      "ridge",
    );

    expect(band.source).toBe("ridge");
    expect(band.ridgeTrusted).toBe(true);
    expect(band.centerFrequency).toBe(1015.625);
    expect(band.lowFrequency).toBe(843.75);
    expect(band.highFrequency).toBe(3080.42138671875);
  });

  it("expands tonal low-moan bands with moderate high-band energy", () => {
    const band = resolveEventDisplayBand(
      event({
        ridge_median_frequency: 312.5,
        ridge_low_frequency: 296.875,
        ridge_high_frequency: 329.6875,
        ridge_coverage: 1,
        ridge_energy_ratio: 0.09594432264566422,
        spectral_centroid: 2101.532958984375,
        bandwidth: 2097.9912109375,
        spectral_entropy: 0.8581035733222961,
        high_band_energy_ratio: 0.5040363669395447,
        voicing_fraction: 1,
        median_f0: 315.19970703125,
        f0_range: 43.60465621948242,
        peak_frequency: 312.5,
        band_limited_peak_frequency: 312.5,
      }),
      "ridge",
    );

    expect(band.source).toBe("ridge");
    expect(band.ridgeTrusted).toBe(true);
    expect(band.centerFrequency).toBe(312.5);
    expect(band.lowFrequency).toBe(296.875);
    expect(band.highFrequency).toBe(2101.532958984375);
  });

  it("falls back to voiced F0 when ridge coverage is weak", () => {
    const band = resolveEventDisplayBand(
      event({
        ridge_median_frequency: 2400,
        ridge_low_frequency: 2100,
        ridge_high_frequency: 2800,
        ridge_coverage: 0.1,
        ridge_energy_ratio: 0.2,
        voicing_fraction: 0.9,
        median_f0: 440,
        f0_range: 120,
        peak_frequency: 80,
      }),
      "ridge",
    );

    expect(band.source).toBe("f0");
    expect(band.ridgeTrusted).toBe(false);
    expect(band.centerFrequency).toBe(440);
    expect(band.lowFrequency).toBe(380);
    expect(band.highFrequency).toBe(500);
  });

  it("falls back to band-limited peak for unvoiced failed-F0 events", () => {
    const band = resolveEventDisplayBand(
      event({
        ridge_median_frequency: 0,
        ridge_coverage: 0,
        ridge_energy_ratio: 0,
        voicing_fraction: 0,
        median_f0: 0,
        peak_frequency: 62.5,
        band_limited_peak_frequency: 2600,
      }),
      "ridge",
    );

    expect(band.source).toBe("band_limited_peak");
    expect(band.centerFrequency).toBe(2600);
    expect(band.lowFrequency).toBe(2590);
    expect(band.highFrequency).toBe(2610);
  });

  it("keeps legacy v2 artifacts renderable without ridge fields", () => {
    const legacy = event({
      voicing_fraction: 0.8,
      median_f0: 300,
      f0_range: 80,
      peak_frequency: 900,
    });

    expect(hasRidgeFrequencyDescriptors([legacy])).toBe(false);
    const band = resolveEventDisplayBand(legacy, "ridge");

    expect(band.source).toBe("f0");
    expect(band.centerFrequency).toBe(300);
    expect(band.lowFrequency).toBe(260);
    expect(band.highFrequency).toBe(340);
  });

  it("ignores non-finite descriptors and falls back safely", () => {
    const sample = event({
      ridge_median_frequency: Number.NaN,
      ridge_low_frequency: Number.POSITIVE_INFINITY,
      ridge_high_frequency: 2000,
      ridge_coverage: 1,
      ridge_energy_ratio: 1,
      voicing_fraction: 0,
      band_limited_peak_frequency: Number.NaN,
      peak_frequency: 1200,
    });
    const band = resolveEventDisplayBand(sample, "ridge");

    expect(isEventVoiced(sample)).toBe(false);
    expect(band.source).toBe("peak");
    expect(band.centerFrequency).toBe(1200);
  });
});
