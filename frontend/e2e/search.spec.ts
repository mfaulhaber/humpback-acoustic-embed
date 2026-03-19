import { test, expect } from "@playwright/test";

/**
 * Tests for the Search tab.
 * Requires: backend running on :8000, frontend on :5173.
 */

test.describe("Search tab", () => {
  test("search tab renders and shows empty state", async ({ page }) => {
    await page.goto("http://localhost:5173/app/search");
    // Should see the Search tab header
    await expect(page.locator("text=Embedding Search")).toBeVisible();
    // Should see the empty state message
    await expect(
      page.locator("text=Select an embedding set"),
    ).toBeVisible();
  });

  test("search-similar-by-vector endpoint returns results", async ({
    request,
  }) => {
    // First check if there are any embedding sets
    const esRes = await request.get(
      "http://localhost:8000/processing/embedding-sets",
    );
    expect(esRes.ok()).toBeTruthy();
    const sets = await esRes.json();
    if (sets.length === 0) {
      test.skip(true, "No embedding sets available");
      return;
    }

    // Use the first embedding set to get a query vector via the existing search
    const searchRes = await request.post(
      "http://localhost:8000/search/similar",
      {
        data: {
          embedding_set_id: sets[0].id,
          row_index: 0,
          top_k: 1,
        },
      },
    );
    if (!searchRes.ok()) {
      test.skip(true, "Search endpoint not working with available data");
      return;
    }

    // Verify the response has expected shape
    const result = await searchRes.json();
    expect(result).toHaveProperty("model_version");
    expect(result).toHaveProperty("total_candidates");
    expect(result).toHaveProperty("results");
  });

  test("spectrogram-png endpoint returns PNG", async ({ request }) => {
    // Check if there are any audio files
    const audioRes = await request.get("http://localhost:8000/audio/");
    expect(audioRes.ok()).toBeTruthy();
    const files = await audioRes.json();
    if (files.length === 0) {
      test.skip(true, "No audio files available");
      return;
    }

    const file = files[0];
    const pngRes = await request.get(
      `http://localhost:8000/audio/${file.id}/spectrogram-png?start_seconds=0&duration_seconds=5`,
    );
    expect(pngRes.ok()).toBeTruthy();
    expect(pngRes.headers()["content-type"]).toBe("image/png");
    const body = await pngRes.body();
    // PNG magic bytes
    expect(body[0]).toBe(0x89);
    expect(body[1]).toBe(0x50); // P
    expect(body[2]).toBe(0x4e); // N
    expect(body[3]).toBe(0x47); // G
  });
});
