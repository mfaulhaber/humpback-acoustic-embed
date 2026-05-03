import { describe, expect, it } from "vitest";
import { MOTIF_PALETTE, colorForMotifKey } from "./motifColor";

describe("colorForMotifKey", () => {
  it("returns a palette entry for any input", () => {
    const c = colorForMotifKey("0-1-2");
    expect(MOTIF_PALETTE).toContainEqual(c);
  });

  it("is deterministic — same key yields same color across calls", () => {
    expect(colorForMotifKey("0-1-2")).toEqual(colorForMotifKey("0-1-2"));
    expect(colorForMotifKey("3-4")).toEqual(colorForMotifKey("3-4"));
  });

  it("yields distinct palette indices for a small set of distinct keys", () => {
    const keys = ["0-1-2", "3-4-5", "7-2-9", "1-1-1", "5-9", "2-3-4-1"];
    const fills = new Set(keys.map((k) => colorForMotifKey(k).fill));
    // Not strictly all-distinct (palette mod-12 collisions possible), but
    // for this tiny fixture we expect at least 4 distinct hues.
    expect(fills.size).toBeGreaterThanOrEqual(4);
  });

  it("handles empty string without throwing", () => {
    expect(() => colorForMotifKey("")).not.toThrow();
    const c = colorForMotifKey("");
    expect(MOTIF_PALETTE).toContainEqual(c);
  });

  it("returns fill and border that differ in alpha", () => {
    const c = colorForMotifKey("test-key");
    expect(c.fill).not.toEqual(c.border);
  });
});
