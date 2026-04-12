import { expect, test, type Page } from "@playwright/test";

const MODEL = {
  id: "model-cp-1",
  name: "CP Test Model",
  model_path: "/tmp/model.joblib",
  model_version: "perch_v2_test",
  vector_dim: 1536,
  window_size_seconds: 5,
  target_sample_rate: 32000,
  feature_config: null,
  training_summary: null,
  training_job_id: null,
  training_source_mode: "embedding_sets",
  source_candidate_id: null,
  source_model_id: null,
  promotion_provenance: null,
  created_at: "2026-04-12T00:00:00Z",
  updated_at: "2026-04-12T00:00:00Z",
};

const MODEL_CONFIG = {
  id: "mc-perch-v2",
  name: "perch_v2_test",
  display_name: "Perch v2 Test",
  path: "/tmp/perch_v2.tflite",
  vector_dim: 1536,
  description: null,
  is_default: true,
  model_type: "tflite",
  input_format: "pcm_float",
  created_at: "2026-04-12T00:00:00Z",
};

const HYDROPHONE = {
  id: "rpi_north_sjc",
  name: "North San Juan Channel",
  location: "San Juan Channel",
  provider_kind: "orcasound_hls",
};

const COMPLETE_JOB = {
  id: "rj-complete-1",
  status: "complete",
  audio_file_id: null,
  hydrophone_id: HYDROPHONE.id,
  start_timestamp: 1751644800,
  end_timestamp: 1751648400,
  model_config_id: MODEL_CONFIG.id,
  classifier_model_id: MODEL.id,
  config_json: JSON.stringify({
    high_threshold: 0.9,
    low_threshold: 0.8,
    hop_seconds: 1.0,
    padding_sec: 1.0,
    min_region_duration_sec: 0.0,
    stream_chunk_sec: 1800,
  }),
  parent_run_id: null,
  error_message: null,
  trace_row_count: 3600,
  region_count: 5,
  created_at: "2026-04-12T01:00:00Z",
  updated_at: "2026-04-12T01:30:00Z",
  started_at: "2026-04-12T01:00:01Z",
  completed_at: "2026-04-12T01:30:00Z",
};

const RUNNING_JOB = {
  id: "rj-running-1",
  status: "running",
  audio_file_id: null,
  hydrophone_id: HYDROPHONE.id,
  start_timestamp: 1751648400,
  end_timestamp: 1751652000,
  model_config_id: MODEL_CONFIG.id,
  classifier_model_id: MODEL.id,
  config_json: JSON.stringify({
    high_threshold: 0.85,
    low_threshold: 0.7,
  }),
  parent_run_id: null,
  error_message: null,
  trace_row_count: null,
  region_count: null,
  created_at: "2026-04-12T02:00:00Z",
  updated_at: "2026-04-12T02:00:00Z",
  started_at: "2026-04-12T02:00:01Z",
  completed_at: null,
};

const FAILED_JOB = {
  id: "rj-failed-1",
  status: "failed",
  audio_file_id: null,
  hydrophone_id: HYDROPHONE.id,
  start_timestamp: 1751652000,
  end_timestamp: 1751655600,
  model_config_id: MODEL_CONFIG.id,
  classifier_model_id: MODEL.id,
  config_json: JSON.stringify({
    high_threshold: 0.9,
    low_threshold: 0.8,
  }),
  parent_run_id: null,
  error_message: "Audio fetch failed: 404",
  trace_row_count: null,
  region_count: null,
  created_at: "2026-04-12T03:00:00Z",
  updated_at: "2026-04-12T03:05:00Z",
  started_at: "2026-04-12T03:00:01Z",
  completed_at: "2026-04-12T03:05:00Z",
};

async function setupMocks(page: Page) {
  await page.route("**/classifier/hydrophones", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([HYDROPHONE]),
    }),
  );
  await page.route("**/classifier/models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([MODEL]),
    }),
  );
  await page.route("**/admin/models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([MODEL_CONFIG]),
    }),
  );
  await page.route("**/call-parsing/region-jobs", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([RUNNING_JOB, COMPLETE_JOB, FAILED_JOB]),
      });
    }
    return route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify(RUNNING_JOB),
    });
  });
}

test.describe("Call Parsing Detection page", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
    await page.goto("/app/call-parsing/detection");
  });

  test("page loads with form visible", async ({ page }) => {
    await expect(page.locator("text=Region Detection")).toBeVisible();
    await expect(page.getByTestId("hydrophone-select")).toBeVisible();
    await expect(page.getByTestId("model-select")).toBeVisible();
    await expect(page.getByTestId("high-threshold-slider")).toBeVisible();
    await expect(page.getByTestId("low-threshold-slider")).toBeVisible();
    await expect(page.getByTestId("date-range-trigger")).toBeVisible();
  });

  test("nav group appears in sidebar with Detection link", async ({ page }) => {
    const navGroup = page.locator("nav").locator("text=Call Parsing");
    await expect(navGroup).toBeVisible();

    const detectionLink = page.locator("nav").locator("a", { hasText: "Detection" });
    await expect(detectionLink).toBeVisible();
  });

  test("Start Detection button disabled when form incomplete", async ({ page }) => {
    const btn = page.getByTestId("start-detection-btn");
    await expect(btn).toBeVisible();
    await expect(btn).toBeDisabled();
  });

  test("advanced settings section is collapsible", async ({ page }) => {
    const trigger = page.locator("button", { hasText: "Advanced Settings" });
    await expect(trigger).toBeVisible();

    await expect(page.getByTestId("hop-size-input")).not.toBeVisible();

    await trigger.click();

    await expect(page.getByTestId("hop-size-input")).toBeVisible();
    await expect(page.getByTestId("padding-input")).toBeVisible();
    await expect(page.getByTestId("min-duration-input")).toBeVisible();
    await expect(page.getByTestId("stream-chunk-input")).toBeVisible();
  });

  test("active jobs panel shows running job with cancel button", async ({ page }) => {
    const activePanel = page.getByTestId("active-jobs-panel");
    await expect(activePanel).toBeVisible();
    await expect(activePanel.locator("text=Active Jobs")).toBeVisible();

    const runningRow = activePanel.locator("tr").filter({ hasText: "running" });
    await expect(runningRow).toBeVisible();

    const cancelBtn = runningRow.locator("button", { hasText: "Cancel" });
    await expect(cancelBtn).toBeVisible();
  });

  test("previous jobs panel shows completed and failed jobs", async ({ page }) => {
    const prevPanel = page.getByTestId("previous-jobs-panel");
    await expect(prevPanel).toBeVisible();
    await expect(prevPanel.locator("text=Previous Jobs")).toBeVisible();

    const completeRow = prevPanel.locator("tr").filter({ hasText: "complete" });
    await expect(completeRow).toBeVisible();
    await expect(completeRow).toContainText("5 regions");

    const failedRow = prevPanel.locator("tr").filter({ hasText: "failed" });
    await expect(failedRow).toBeVisible();
    await expect(failedRow).toContainText("Audio fetch failed");
  });

  test("previous jobs table has correct columns", async ({ page }) => {
    const prevPanel = page.getByTestId("previous-jobs-panel");
    const headers = prevPanel.locator("thead th");

    await expect(headers.filter({ hasText: "Status" })).toBeVisible();
    await expect(headers.filter({ hasText: "Created" })).toBeVisible();
    await expect(headers.filter({ hasText: "Hydrophone" })).toBeVisible();
    await expect(headers.filter({ hasText: "Date Range" })).toBeVisible();
    await expect(headers.filter({ hasText: "Thresholds" })).toBeVisible();
    await expect(headers.filter({ hasText: "Regions" })).toBeVisible();
    await expect(headers.filter({ hasText: "Timeline" })).toBeVisible();
  });

  test("timeline button is disabled", async ({ page }) => {
    const prevPanel = page.getByTestId("previous-jobs-panel");
    const timelineBtn = prevPanel.locator("button", { hasText: "Timeline" }).first();
    await expect(timelineBtn).toBeVisible();
    await expect(timelineBtn).toBeDisabled();
  });

  test("thresholds display from config_json", async ({ page }) => {
    const prevPanel = page.getByTestId("previous-jobs-panel");
    const completeRow = prevPanel.locator("tr").filter({ hasText: "complete" });
    await expect(completeRow).toContainText("0.90 / 0.80");
  });

  test("hydrophone name resolved from ID", async ({ page }) => {
    const prevPanel = page.getByTestId("previous-jobs-panel");
    const completeRow = prevPanel.locator("tr").filter({ hasText: "complete" });
    await expect(completeRow).toContainText("North San Juan Channel");
  });
});
