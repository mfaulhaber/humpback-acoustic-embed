import { describe, expect, it } from "vitest";
import { regionTileIndexForSpanTile } from "./timelineTileIndex";

describe("regionTileIndexForSpanTile", () => {
  it("maps span-local tile indices back to the source region job origin", () => {
    expect(regionTileIndexForSpanTile(200, 100, 0, 50)).toBe(2);
    expect(regionTileIndexForSpanTile(200, 100, 1, 50)).toBe(3);
  });

  it("clamps tiles before the source region start", () => {
    expect(regionTileIndexForSpanTile(90, 100, 0, 50)).toBe(0);
  });
});
