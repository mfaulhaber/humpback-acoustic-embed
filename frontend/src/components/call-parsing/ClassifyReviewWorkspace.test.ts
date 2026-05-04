import { describe, expect, it } from "vitest";

import {
  buildMergedCorrections,
  resolveEventType,
} from "./ClassifyReviewWorkspace";

describe("buildMergedCorrections", () => {
  it("reattaches saved vocalization labels to boundary-added events after reload", () => {
    const merged = buildMergedCorrections(
      [
        {
          event_id: "existing-event",
          region_id: "region-1",
          start_sec: 10,
          end_sec: 11,
          type_name: "Pop",
          score: 0.8,
          above_threshold: true,
        },
      ],
      [
        {
          start_sec: 42,
          end_sec: 42.5,
          type_name: "Growl",
          correction_type: "add",
        },
      ],
      [
        {
          id: "boundary-add-1",
          region_detection_job_id: "rd-1",
          event_segmentation_job_id: null,
          region_id: "region-2",
          source_event_id: null,
          correction_type: "add",
          original_start_sec: null,
          original_end_sec: null,
          corrected_start_sec: 42,
          corrected_end_sec: 42.5,
          created_at: "2026-04-24T00:00:00Z",
          updated_at: "2026-04-24T00:00:00Z",
        },
      ],
      new Map(),
    );

    expect(merged.get("saved-add-boundary-add-1")).toBe("Growl");
  });

  it("preserves negative corrections for existing typed events", () => {
    const merged = buildMergedCorrections(
      [
        {
          event_id: "existing-event",
          region_id: "region-1",
          start_sec: 10,
          end_sec: 11,
          type_name: "Pop",
          score: 0.8,
          above_threshold: true,
        },
      ],
      [
        {
          start_sec: 10,
          end_sec: 11,
          type_name: "Pop",
          correction_type: "remove",
        },
      ],
      [],
      new Map(),
    );

    expect(merged.get("existing-event")).toBeNull();
  });
});

describe("resolveEventType", () => {
  it("returns inference when no correction exists", () => {
    const result = resolveEventType("Upcall", undefined);
    expect(result).toEqual({
      effectiveType: "Upcall",
      typeSource: "inference",
    });
  });

  it("returns negative when correction is null", () => {
    const result = resolveEventType("Upcall", null);
    expect(result).toEqual({ effectiveType: null, typeSource: "negative" });
  });

  it("returns approved when correction matches prediction", () => {
    const result = resolveEventType("Upcall", "Upcall");
    expect(result).toEqual({
      effectiveType: "Upcall",
      typeSource: "approved",
    });
  });

  it("returns correction when correction differs from prediction", () => {
    const result = resolveEventType("Upcall", "Moan");
    expect(result).toEqual({
      effectiveType: "Moan",
      typeSource: "correction",
    });
  });

  it("returns correction when prediction is null and correction is set", () => {
    const result = resolveEventType(null, "Upcall");
    expect(result).toEqual({
      effectiveType: "Upcall",
      typeSource: "correction",
    });
  });

  it("returns null type source when both are undefined/null", () => {
    const result = resolveEventType(null, undefined);
    expect(result).toEqual({ effectiveType: null, typeSource: null });
  });
});
