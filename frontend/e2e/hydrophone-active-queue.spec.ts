import { expect, test, type Page } from "@playwright/test";

const MODEL = {
  id: "model-aq-1",
  name: "AQ Test Model",
  model_path: "/tmp/model.joblib",
  model_version: "perch_v1",
  vector_dim: 1280,
  window_size_seconds: 5,
  target_sample_rate: 32000,
  feature_config: null,
  training_summary: null,
  training_job_id: null,
  created_at: "2026-03-09T00:00:00Z",
  updated_at: "2026-03-09T00:00:00Z",
};

const HYDROPHONE = {
  id: "rpi_north_sjc",
  name: "North San Juan Channel",
  location: "San Juan Channel",
  provider_kind: "orcasound_hls",
};

function buildJob(overrides: Record<string, unknown> = {}) {
  return {
    id: "job-aq-1",
    status: "running",
    classifier_model_id: MODEL.id,
    audio_folder: null,
    confidence_threshold: 0.5,
    hop_seconds: 1.0,
    high_threshold: 0.7,
    low_threshold: 0.45,
    output_tsv_path: "/tmp/detections.tsv",
    result_summary: null,
    error_message: null,
    files_processed: null,
    files_total: null,
    extract_status: null,
    extract_error: null,
    extract_summary: null,
    hydrophone_id: HYDROPHONE.id,
    hydrophone_name: HYDROPHONE.name,
    start_timestamp: 1751644800,
    end_timestamp: 1751648400,
    segments_processed: 5,
    segments_total: 12,
    time_covered_sec: 300,
    alerts: null,
    local_cache_path: null,
    has_positive_labels: null,
    created_at: "2026-03-09T00:00:00Z",
    updated_at: "2026-03-09T00:00:00Z",
    ...overrides,
  };
}

async function setupMocks(page: Page, jobList: Record<string, unknown>[]) {
  let cancelledIds = new Set<string>();

  await page.route("**/classifier/training-jobs", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/classifier/models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([MODEL]),
    }),
  );
  await page.route("**/classifier/hydrophones", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([HYDROPHONE]),
    }),
  );
  await page.route("**/classifier/hydrophone-detection-jobs", (route) => {
    if (route.request().method() === "GET") {
      const current = jobList.map((j) => {
        const id = j.id as string;
        if (cancelledIds.has(id)) return { ...j, status: "canceled" };
        return j;
      });
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(current),
      });
    }
    return route.fallback();
  });

  await page.route(/\/hydrophone-detection-jobs\/[^/]+\/cancel$/, (route) => {
    const url = route.request().url();
    const match = url.match(/\/hydrophone-detection-jobs\/([^/]+)\/cancel$/);
    if (match) cancelledIds.add(match[1]);
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "canceled" }),
    });
  });

  await page.route(/\/hydrophone-detection-jobs\/[^/]+\/pause$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "paused" }),
    }),
  );

  await page.route(/\/hydrophone-detection-jobs\/[^/]+\/resume$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "running" }),
    }),
  );

  // Mock content for paused jobs
  await page.route(/\/detection-jobs\/[^/]+\/content$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          filename: "20250704T090000Z.wav",
          start_sec: 0,
          end_sec: 5,
          avg_confidence: 0.85,
          peak_confidence: 0.92,
          n_windows: 3,
          humpback: null,
          orca: null,
          ship: null,
          background: null,
          detection_filename: "20250704T090000Z_20250704T090005Z.flac",
          extract_filename: "20250704T090000Z_20250704T090005Z.flac",
          raw_start_sec: 0,
          raw_end_sec: 5,
          merged_event_count: 1,
          hydrophone_name: HYDROPHONE.name,
        },
      ]),
    }),
  );

  return { cancelledIds };
}

test.describe("Hydrophone Active Jobs Queue", () => {
  test("multiple active jobs render in table with correct status badges", async ({
    page,
  }) => {
    const jobs = [
      buildJob({ id: "j1", status: "running", segments_processed: 5, segments_total: 12 }),
      buildJob({ id: "j2", status: "paused", segments_processed: 3, segments_total: 10 }),
      buildJob({ id: "j3", status: "queued", segments_processed: 0, segments_total: null }),
    ];
    await setupMocks(page, jobs);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    const activeSection = page.locator("text=Active Jobs").locator("..");
    await expect(activeSection).toBeVisible();

    // Verify status badges exist
    const badges = page.locator("table").first().locator("td .inline-flex, td span");
    await expect(page.locator("table").first().locator("text=running").first()).toBeVisible();
    await expect(page.locator("table").first().locator("text=paused").first()).toBeVisible();
    await expect(page.locator("table").first().locator("text=queued").first()).toBeVisible();
  });

  test("running job shows Pause + Cancel; paused shows Resume + Cancel; queued shows Cancel only", async ({
    page,
  }) => {
    const jobs = [
      buildJob({ id: "j1", status: "running" }),
      buildJob({ id: "j2", status: "paused" }),
      buildJob({ id: "j3", status: "queued", segments_processed: 0, segments_total: null }),
    ];
    await setupMocks(page, jobs);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    // Active Jobs table should exist
    await expect(page.locator("text=Active Jobs")).toBeVisible();

    const rows = page.locator("table").first().locator("tbody tr");

    // Running row (j1): Pause + Cancel
    const runningRow = rows.nth(0);
    await expect(runningRow.locator("button", { hasText: "Pause" })).toBeVisible();
    await expect(runningRow.locator("button", { hasText: "Cancel" })).toBeVisible();
    expect(await runningRow.locator("button", { hasText: "Resume" }).count()).toBe(0);

    // Paused row (j2): Resume + Cancel
    const pausedRow = rows.nth(1);
    await expect(pausedRow.locator("button", { hasText: "Resume" })).toBeVisible();
    await expect(pausedRow.locator("button", { hasText: "Cancel" })).toBeVisible();
    expect(await pausedRow.locator("button", { hasText: "Pause" }).count()).toBe(0);

    // Queued row (j3): Cancel only
    const queuedRow = rows.nth(2);
    await expect(queuedRow.locator("button", { hasText: "Cancel" })).toBeVisible();
    expect(await queuedRow.locator("button", { hasText: "Pause" }).count()).toBe(0);
    expect(await queuedRow.locator("button", { hasText: "Resume" }).count()).toBe(0);
  });

  test("expanding paused job shows detection content table", async ({ page }) => {
    const jobs = [
      buildJob({
        id: "j-paused",
        status: "paused",
        segments_processed: 5,
        segments_total: 10,
        output_tsv_path: "/tmp/detections.tsv",
      }),
    ];
    await setupMocks(page, jobs);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    await expect(page.locator("text=Active Jobs")).toBeVisible();

    // Click expand chevron
    const expandBtn = page.locator("table").first().locator("tbody button").first();
    await expandBtn.click();

    // Detection content table should appear with at least one data row
    await expect(page.locator("text=Detection Range")).toBeVisible({ timeout: 5000 });
  });

  test("cancel queued job moves it to Previous Jobs", async ({ page }) => {
    const jobs = [
      buildJob({ id: "j-q", status: "queued", segments_processed: 0, segments_total: null }),
    ];
    await setupMocks(page, jobs);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    await expect(page.locator("text=Active Jobs")).toBeVisible();

    // Click Cancel on the queued job
    await page.locator("button", { hasText: "Cancel" }).click();

    // After re-fetch, should show in Previous Jobs (mock returns canceled status)
    await expect(page.locator("text=Previous Jobs")).toBeVisible({ timeout: 10_000 });
  });
});
