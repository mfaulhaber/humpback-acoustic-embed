import { describe, expect, it } from "vitest";

import { buildMergedCorrections } from "./ClassifyReviewWorkspace";

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
          region_id: "region-2",
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
