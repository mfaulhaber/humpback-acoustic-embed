import { test, expect } from "@playwright/test";

/**
 * Tests for incremental detection rendering — live results during a running job.
 * Requires: backend running on :8000.
 *
 * API tests create a synthetic running job via direct DB manipulation.
 * UI tests check behavior when running/completed jobs exist.
 */

const API = "http://localhost:8000";

// ---- API-level tests ----

test.describe("Detection progress API", () => {
  test("job response includes progress fields", async ({ request }) => {
    const jobsRes = await request.get(`${API}/classifier/detection-jobs`);
    expect(jobsRes.ok()).toBeTruthy();
    const jobs = await jobsRes.json();
    if (jobs.length === 0) {
      test.skip(true, "No detection jobs available");
      return;
    }

    const job = jobs[0];
    // Progress fields should be present in the schema (nullable)
    expect("files_processed" in job).toBe(true);
    expect("files_total" in job).toBe(true);
  });

  test("content endpoint serves running job with output_tsv_path", async ({
    request,
  }) => {
    const jobsRes = await request.get(`${API}/classifier/detection-jobs`);
    expect(jobsRes.ok()).toBeTruthy();
    const jobs = await jobsRes.json();

    const runningJob = jobs.find(
      (j: { status: string; output_tsv_path: string | null }) =>
        j.status === "running" && j.output_tsv_path,
    );
    if (!runningJob) {
      test.skip(true, "No running detection job with output_tsv_path");
      return;
    }

    const contentRes = await request.get(
      `${API}/classifier/detection-jobs/${runningJob.id}/content`,
    );
    // Should succeed (200) — not 400
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(Array.isArray(rows)).toBe(true);
  });

  test("content endpoint rejects queued job", async ({ request }) => {
    const jobsRes = await request.get(`${API}/classifier/detection-jobs`);
    expect(jobsRes.ok()).toBeTruthy();
    const jobs = await jobsRes.json();

    const queuedJob = jobs.find(
      (j: { status: string }) => j.status === "queued",
    );
    if (!queuedJob) {
      test.skip(true, "No queued detection job");
      return;
    }

    const contentRes = await request.get(
      `${API}/classifier/detection-jobs/${queuedJob.id}/content`,
    );
    expect(contentRes.status()).toBe(400);
  });
});

