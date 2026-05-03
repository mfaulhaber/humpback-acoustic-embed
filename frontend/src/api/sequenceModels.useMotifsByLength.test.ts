import { describe, expect, it } from "vitest";
import {
  filterMotifsByLength,
  mergeOccurrencesByStart,
  type MotifOccurrence,
  type MotifSummary,
} from "./sequenceModels";

function motif(key: string, length: number): MotifSummary {
  return {
    motif_key: key,
    states: key.split("-").map((s) => Number.parseInt(s, 10)),
    length,
    occurrence_count: 0,
    event_source_count: 0,
    audio_source_count: 0,
    group_count: 0,
    event_core_fraction: 0,
    background_fraction: 0,
    mean_call_probability: null,
    mean_duration_seconds: 0,
    median_duration_seconds: 0,
    rank_score: 0,
    example_occurrence_ids: [],
  };
}

function occ(motifKey: string, start: number, end: number): MotifOccurrence {
  return {
    occurrence_id: `${motifKey}@${start}`,
    motif_key: motifKey,
    states: [],
    source_kind: "region_crnn",
    group_key: "",
    event_source_key: "",
    audio_source_key: null,
    token_start_index: 0,
    token_end_index: 0,
    raw_start_index: 0,
    raw_end_index: 0,
    start_timestamp: start,
    end_timestamp: end,
    duration_seconds: end - start,
    event_core_fraction: 0,
    background_fraction: 0,
    mean_call_probability: null,
    anchor_event_id: null,
    anchor_timestamp: 0,
    relative_start_seconds: 0,
    relative_end_seconds: 0,
    anchor_strategy: "",
  };
}

describe("filterMotifsByLength", () => {
  const fixture = [
    motif("a-b", 2),
    motif("c-d-e", 3),
    motif("f-g", 2),
    motif("h-i-j-k", 4),
    motif("l-m-n", 3),
  ];

  it("returns only motifs whose length matches", () => {
    expect(filterMotifsByLength(fixture, 2).map((m) => m.motif_key)).toEqual([
      "a-b",
      "f-g",
    ]);
    expect(filterMotifsByLength(fixture, 3).map((m) => m.motif_key)).toEqual([
      "c-d-e",
      "l-m-n",
    ]);
    expect(filterMotifsByLength(fixture, 4).map((m) => m.motif_key)).toEqual([
      "h-i-j-k",
    ]);
  });

  it("returns empty array for length=null", () => {
    expect(filterMotifsByLength(fixture, null)).toEqual([]);
  });

  it("returns empty array when no motifs match", () => {
    expect(filterMotifsByLength(fixture, 7)).toEqual([]);
  });
});

describe("mergeOccurrencesByStart", () => {
  it("flattens and sorts occurrences ascending by start_timestamp", () => {
    const a = [occ("a", 100, 105), occ("a", 50, 55)];
    const b = [occ("b", 75, 80), occ("b", 200, 205)];
    const merged = mergeOccurrencesByStart([a, b]);
    expect(merged.map((o) => o.start_timestamp)).toEqual([50, 75, 100, 200]);
  });

  it("skips undefined entries (queries still pending)", () => {
    const a = [occ("a", 10, 20)];
    const merged = mergeOccurrencesByStart([a, undefined, undefined]);
    expect(merged.map((o) => o.motif_key)).toEqual(["a"]);
  });

  it("returns empty array when every entry is undefined", () => {
    expect(mergeOccurrencesByStart([undefined, undefined])).toEqual([]);
  });

  it("preserves every occurrence exactly once", () => {
    const a = [occ("a", 10, 20), occ("a", 30, 40)];
    const b = [occ("b", 25, 35)];
    const merged = mergeOccurrencesByStart([a, b]);
    expect(merged).toHaveLength(3);
    expect(merged.map((o) => o.occurrence_id).sort()).toEqual([
      "a@10",
      "a@30",
      "b@25",
    ]);
  });
});
