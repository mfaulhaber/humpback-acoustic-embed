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
