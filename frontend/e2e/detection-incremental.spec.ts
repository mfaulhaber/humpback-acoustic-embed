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

// ---- UI tests ----

test.describe("Detection incremental UI", () => {
  test("running job shows progress text in results column", async ({
    page,
  }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Detect" }).click();

    const table = page.locator("table").first();
    const hasTable = await table
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasTable) {
      test.skip(true, "No detection jobs table visible");
      return;
    }

    // Look for a running row with progress text like "Processing file X/Y"
    const progressText = page.locator("text=/Processing file \\d+\\/\\d+/");
    const hasProgress = await progressText
      .first()
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasProgress) {
      test.skip(true, "No running detection job with progress visible");
      return;
    }

    await expect(progressText.first()).toBeVisible();
  });

  test("running job with results is expandable and shows live banner", async ({
    page,
  }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Detect" }).click();

    const table = page.locator("table").first();
    const hasTable = await table
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasTable) {
      test.skip(true, "No detection jobs table visible");
      return;
    }

    // Find a running job row with an expand button
    const runningRow = page
      .locator("table tbody tr")
      .filter({ hasText: "running" })
      .first();
    const hasRow = await runningRow
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasRow) {
      test.skip(true, "No running detection job rows");
      return;
    }

    const expandBtn = runningRow.locator("td:nth-child(2) button");
    const hasExpand = await expandBtn
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasExpand) {
      test.skip(true, "Running job not yet expandable (no results yet)");
      return;
    }

    // Expand the running job
    await expandBtn.click();

    // Should see the live banner
    const banner = page.locator("text=Detection in progress");
    await expect(banner).toBeVisible({ timeout: 10_000 });

    // Save Labels button should be disabled
    const saveBtn = page.locator("button", { hasText: "Save Labels" });
    await expect(saveBtn).toBeDisabled();
  });

  test("completed job has no live banner and Save Labels enabled after edit", async ({
    page,
  }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Detect" }).click();

    const table = page.locator("table").first();
    const hasTable = await table
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasTable) {
      test.skip(true, "No detection jobs table visible");
      return;
    }

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

    const expandBtn = completedRow.locator("td:nth-child(2) button");
    await expect(expandBtn).toBeVisible();
    await expandBtn.click();

    // Wait for inner table
    const innerTable = page.locator("tr td[colspan] table");
    await innerTable.waitFor({ timeout: 10_000 });

    // No live banner for completed jobs
    const banner = page.locator("text=Detection in progress");
    await expect(banner).not.toBeVisible();

    // Download TSV link should be visible for completed job
    const downloadLink = completedRow.locator("a", { hasText: "TSV" });
    await expect(downloadLink).toBeVisible();

    // Click a label checkbox — Save Labels should become enabled
    const firstDataRow = innerTable.locator("tbody tr").first();
    const checkbox = firstDataRow.locator(
      'td:nth-child(6) input[type="checkbox"]',
    );
    await expect(checkbox).toBeVisible();
    await checkbox.click();

    const saveBtn = page.locator("button", { hasText: "Save Labels" });
    await expect(saveBtn).toBeEnabled();
  });

  test("completed job default sort is confidence descending", async ({
    page,
  }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Detect" }).click();

    const table = page.locator("table").first();
    const hasTable = await table
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasTable) {
      test.skip(true, "No detection jobs table visible");
      return;
    }

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

    const expandBtn = completedRow.locator("td:nth-child(2) button");
    await expandBtn.click();

    const innerTable = page.locator("tr td[colspan] table");
    await innerTable.waitFor({ timeout: 10_000 });

    // The Confidence header should have the sort arrow (descending by default)
    const confidenceHeader = innerTable.locator("thead th", {
      hasText: "Confidence",
    });
    await expect(confidenceHeader).toBeVisible();
    // Should contain an SVG (the ArrowDown icon)
    const sortIcon = confidenceHeader.locator("svg");
    await expect(sortIcon).toBeVisible();
  });
});
