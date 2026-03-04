import { test, expect } from "@playwright/test";

/**
 * Tests that detection audio playback works correctly.
 * Requires: backend running on :8000 with at least one completed detection job.
 */

test.describe("Detection audio playback", () => {
  test("audio-slice endpoint returns correct duration WAV", async ({
    request,
  }) => {
    // Fetch detection jobs from API
    const jobsRes = await request.get(
      "http://localhost:8000/classifier/detection-jobs",
    );
    expect(jobsRes.ok()).toBeTruthy();
    const jobs = await jobsRes.json();

    const completedJob = jobs.find(
      (j: { status: string; output_tsv_path: string | null }) =>
        j.status === "complete" && j.output_tsv_path,
    );
    if (!completedJob) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    // Fetch detection content
    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    const row = rows[0];
    const spanDuration = row.end_sec - row.start_sec;
    const duration = Math.max(spanDuration, 5);

    // Request audio slice
    const sliceRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/audio-slice` +
        `?filename=${encodeURIComponent(row.filename)}` +
        `&start_sec=${row.start_sec}&duration_sec=${duration}`,
    );
    expect(sliceRes.ok()).toBeTruthy();

    const body = await sliceRes.body();
    // WAV header: bytes 0-3 = "RIFF", 4-7 = file size - 8
    expect(body.slice(0, 4).toString()).toBe("RIFF");

    // Parse sample rate from WAV header (bytes 24-27, little-endian uint32)
    const sampleRate = body.readUInt32LE(24);
    // Parse data chunk size (bytes 40-43)
    const dataSize = body.readUInt32LE(40);
    const numSamples = dataSize / 2; // 16-bit PCM
    const audioDuration = numSamples / sampleRate;

    console.log(
      `Row: file=${row.filename}, start=${row.start_sec}, end=${row.end_sec}`,
    );
    console.log(
      `WAV: sr=${sampleRate}, samples=${numSamples}, duration=${audioDuration.toFixed(2)}s`,
    );
    console.log(`Expected duration: ~${duration}s`);
    console.log(`Response size: ${body.length} bytes`);

    // Audio should be at least 1 second (not a tiny fragment)
    expect(audioDuration).toBeGreaterThan(1.0);
    // Audio should be approximately the requested duration (within 1 second tolerance)
    expect(audioDuration).toBeGreaterThan(duration - 1);
    expect(audioDuration).toBeLessThan(duration + 1);
  });

  test("play button triggers audio element with correct src", async ({
    page,
  }) => {
    // Navigate directly to the classifier page
    await page.goto("/app/classifier");

    // Click the "Detect" sub-tab button
    await page.locator("button", { hasText: "Detect" }).click();

    // Wait for detection jobs table to appear
    const table = page.locator("table").first();
    const hasTable = await table
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasTable) {
      test.skip(true, "No detection jobs table visible");
      return;
    }

    // Find a completed job row with an expand button
    const completedRow = page
      .locator("table tbody tr")
      .filter({ hasText: "complete" })
      .first();
    const hasRow = await completedRow
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasRow) {
      test.skip(true, "No completed detection job rows");
      return;
    }

    // Click the expand chevron — it's the button in the second td (first td has checkbox)
    const expandBtn = completedRow.locator("td:nth-child(2) button");
    await expect(expandBtn).toBeVisible();
    await expandBtn.click();

    // Wait for the inner detection content table to load
    const innerTable = page.locator("tr td[colspan] table");
    await innerTable.waitFor({ timeout: 10_000 });

    // Find a play button in the inner table
    const playBtn = innerTable.locator("tbody tr button").first();
    await expect(playBtn).toBeVisible();

    // Click play
    await playBtn.click();
    await page.waitForTimeout(500);

    // Check audio element has a src set with audio-slice URL
    const audioInfo = await page.evaluate(() => {
      const audio = document.querySelector("audio");
      if (!audio || !audio.src) return null;
      const url = new URL(audio.src);
      return {
        src: audio.src,
        durationSec: parseFloat(
          url.searchParams.get("duration_sec") || "0",
        ),
        hasAudioSlice: audio.src.includes("audio-slice"),
      };
    });

    expect(audioInfo).toBeTruthy();
    expect(audioInfo!.hasAudioSlice).toBe(true);
    console.log(`Audio src duration_sec param: ${audioInfo!.durationSec}`);
    // Duration should be at least 5 seconds
    expect(audioInfo!.durationSec).toBeGreaterThanOrEqual(5);
  });
});
