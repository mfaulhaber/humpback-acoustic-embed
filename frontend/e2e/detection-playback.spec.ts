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

    // Verify peak normalization: the loudest sample should be near ±32767
    const dataOffset = 44; // WAV header size
    let maxAbs = 0;
    for (let i = dataOffset; i < body.length - 1; i += 2) {
      const sample = body.readInt16LE(i);
      const abs = Math.abs(sample);
      if (abs > maxAbs) maxAbs = abs;
    }
    console.log(`Peak sample magnitude: ${maxAbs} / 32767`);
    // Normalized audio should have peak above 90% of full scale
    expect(maxAbs).toBeGreaterThan(32767 * 0.9);
  });

});
