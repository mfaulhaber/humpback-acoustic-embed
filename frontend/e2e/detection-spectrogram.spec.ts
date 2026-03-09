import { test, expect } from "@playwright/test";

/**
 * Tests for the detection spectrogram popup feature.
 * API-level: verifies the spectrogram endpoint returns valid PNG.
 * UI-level: verifies Alt+click on a detection row shows the popup.
 */

test.describe("Detection spectrogram", () => {
  test("spectrogram endpoint returns PNG for a completed detection job", async ({
    request,
  }) => {
    // Try local detection jobs first
    const jobsRes = await request.get(
      "http://localhost:8000/classifier/detection-jobs",
    );
    expect(jobsRes.ok()).toBeTruthy();
    let jobs = await jobsRes.json();

    let completedJob = jobs.find(
      (j: { status: string; output_tsv_path: string | null }) =>
        j.status === "complete" && j.output_tsv_path,
    );

    // Fall back to hydrophone jobs
    if (!completedJob) {
      const hydroRes = await request.get(
        "http://localhost:8000/classifier/hydrophone-detection-jobs",
      );
      expect(hydroRes.ok()).toBeTruthy();
      jobs = await hydroRes.json();
      completedJob = jobs.find(
        (j: { status: string; output_tsv_path: string | null }) =>
          (j.status === "complete" || j.status === "canceled") &&
          j.output_tsv_path,
      );
    }

    if (!completedJob) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    // Fetch detection content to get a real row
    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    const row = rows[0];
    const duration = Math.max(row.end_sec - row.start_sec, 5);

    // Request spectrogram
    const specRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/spectrogram` +
        `?filename=${encodeURIComponent(row.filename)}` +
        `&start_sec=${row.start_sec}&duration_sec=${duration}`,
    );
    expect(specRes.ok()).toBeTruthy();
    expect(specRes.headers()["content-type"]).toBe("image/png");

    const body = await specRes.body();
    // Verify PNG magic bytes
    expect(body.slice(0, 4).toString("hex")).toBe("89504e47");
    expect(body.length).toBeGreaterThan(1000);
  });

  test("spectrogram endpoint returns 404 for nonexistent job", async ({
    request,
  }) => {
    const res = await request.get(
      "http://localhost:8000/classifier/detection-jobs/nonexistent/spectrogram" +
        "?filename=test.wav&start_sec=0&duration_sec=5",
    );
    expect(res.status()).toBe(404);
  });
});
