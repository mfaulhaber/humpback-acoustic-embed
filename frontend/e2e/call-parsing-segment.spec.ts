import { expect, test, type Page } from "@playwright/test";

const HYDROPHONE = {
  id: "rpi_north_sjc",
  name: "North San Juan Channel",
  location: "San Juan Channel",
  provider_kind: "orcasound_hls",
};

const COMPLETE_REGION_JOB = {
  id: "rj-complete-1",
  status: "complete",
  audio_file_id: null,
  hydrophone_id: HYDROPHONE.id,
  start_timestamp: 1751644800,
  end_timestamp: 1751648400,
  model_config_id: "mc-1",
  classifier_model_id: "cm-1",
  config_json: JSON.stringify({ high_threshold: 0.9, low_threshold: 0.8 }),
  parent_run_id: null,
  error_message: null,
  chunks_total: null,
  chunks_completed: null,
  windows_detected: null,
  trace_row_count: 3600,
  region_count: 5,
  created_at: "2026-04-12T01:00:00Z",
  updated_at: "2026-04-12T01:30:00Z",
  started_at: "2026-04-12T01:00:01Z",
  completed_at: "2026-04-12T01:30:00Z",
};

const FAILED_REGION_JOB = {
  id: "rj-failed-1",
  status: "failed",
  audio_file_id: null,
  hydrophone_id: HYDROPHONE.id,
  start_timestamp: 1751652000,
  end_timestamp: 1751655600,
  model_config_id: "mc-1",
  classifier_model_id: "cm-1",
  config_json: JSON.stringify({ high_threshold: 0.9, low_threshold: 0.8 }),
  parent_run_id: null,
  error_message: "fetch failed",
  chunks_total: null,
  chunks_completed: null,
  windows_detected: null,
  trace_row_count: null,
  region_count: null,
  created_at: "2026-04-12T03:00:00Z",
  updated_at: "2026-04-12T03:05:00Z",
  started_at: "2026-04-12T03:00:01Z",
  completed_at: "2026-04-12T03:05:00Z",
};

const SEG_MODEL = {
  id: "sm-1",
  name: "crnn-bootstrap-v1",
  model_family: "pytorch_crnn",
  model_path: "/tmp/crnn.pt",
  config_json: JSON.stringify({
    framewise_f1: 0.81,
    event_f1_iou_0_3: 0.73,
  }),
  training_job_id: "stj-1",
  created_at: "2026-04-11T00:00:00Z",
};

const COMPLETE_SEG_JOB = {
  id: "sj-complete-1",
  status: "complete",
  region_detection_job_id: COMPLETE_REGION_JOB.id,
  segmentation_model_id: SEG_MODEL.id,
  config_json: JSON.stringify({ high_threshold: 0.5, low_threshold: 0.3 }),
  parent_run_id: null,
  event_count: 12,
  error_message: null,
  created_at: "2026-04-12T04:00:00Z",
  updated_at: "2026-04-12T04:05:00Z",
  started_at: "2026-04-12T04:00:01Z",
  completed_at: "2026-04-12T04:05:00Z",
};

const SEG_EVENTS = [
  {
    event_id: "ev-1",
    region_id: "reg-aaaa",
    start_sec: 100.0,
    end_sec: 102.5,
    center_sec: 101.25,
    segmentation_confidence: 0.95,
  },
  {
    event_id: "ev-2",
    region_id: "reg-aaaa",
    start_sec: 110.0,
    end_sec: 111.2,
    center_sec: 110.6,
    segmentation_confidence: 0.82,
  },
];

const TRAINING_DATASET = {
  id: "ds-1",
  name: "bootstrap-orcasound-v1",
  sample_count: 243,
  created_at: "2026-04-10T00:00:00Z",
};

const TRAINING_JOB = {
  id: "stj-1",
  status: "complete",
  training_dataset_id: TRAINING_DATASET.id,
  config_json: JSON.stringify({ epochs: 30, learning_rate: 0.001 }),
  segmentation_model_id: SEG_MODEL.id,
  result_summary: JSON.stringify({
    framewise_f1: 0.81,
    event_f1_iou_0_3: 0.73,
  }),
  error_message: null,
  created_at: "2026-04-11T00:00:00Z",
  updated_at: "2026-04-11T01:00:00Z",
  started_at: "2026-04-11T00:00:01Z",
  completed_at: "2026-04-11T01:00:00Z",
};

async function setupMocks(page: Page) {
  await page.route("**/classifier/hydrophones", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([HYDROPHONE]),
    }),
  );
  await page.route("**/call-parsing/region-jobs", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([COMPLETE_REGION_JOB, FAILED_REGION_JOB]),
    }),
  );
  await page.route("**/call-parsing/segmentation-models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([SEG_MODEL]),
    }),
  );
  await page.route("**/call-parsing/segmentation-jobs", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([COMPLETE_SEG_JOB]),
      });
    }
    return route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify(COMPLETE_SEG_JOB),
    });
  });
  await page.route("**/call-parsing/segmentation-jobs/*/events", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(SEG_EVENTS),
    }),
  );
  await page.route("**/call-parsing/segmentation-training-datasets", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([TRAINING_DATASET]),
    }),
  );
  await page.route("**/call-parsing/segmentation-training-jobs", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([TRAINING_JOB]),
    }),
  );
  // Catch other classifier/admin routes the hooks may call
  await page.route("**/classifier/models", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/admin/models", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
}

test.describe("Call Parsing Segment page", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("page loads with form and both dropdowns", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    await expect(page.locator("text=New Segmentation Job")).toBeVisible();
    await expect(
      page.locator("select").filter({ hasText: "Select a completed region job" }),
    ).toBeVisible();
    await expect(
      page.locator("select").filter({ hasText: "Select a model" }),
    ).toBeVisible();
  });

  test("region job dropdown shows only complete jobs", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    const regionSelect = page.locator("select").first();
    const options = regionSelect.locator("option");
    // placeholder + 1 completed job (failed job should not appear)
    await expect(options).toHaveCount(2);
    await expect(options.nth(1)).toContainText("North San Juan Channel");
    await expect(options.nth(1)).toContainText("5 regions");
  });

  test("model dropdown shows model with F1", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    const modelSelect = page.locator("select").nth(1);
    const options = modelSelect.locator("option");
    await expect(options).toHaveCount(2);
    await expect(options.nth(1)).toContainText("crnn-bootstrap-v1");
    await expect(options.nth(1)).toContainText("F1: 0.73");
  });

  test("previous jobs table shows completed job with linked columns", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment");
    await expect(page.locator("text=Previous Jobs")).toBeVisible();
    const row = page.locator("tr").filter({ hasText: "complete" });
    await expect(row).toBeVisible();
    await expect(row).toContainText("12");
    // Source and model are links
    const sourceLink = row.locator("a").filter({ hasText: "North San Juan" });
    await expect(sourceLink).toBeVisible();
    const modelLink = row.locator("a").filter({ hasText: "crnn-bootstrap-v1" });
    await expect(modelLink).toBeVisible();
  });

  test("expand detail shows stats and events table", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    await expect(page.locator("text=Previous Jobs")).toBeVisible();
    // Click the chevron icon to expand the detail row
    const chevron = page.locator(".lucide-chevron-down").first();
    await expect(chevron).toBeVisible();
    await chevron.click();
    // Summary stats
    await expect(page.locator("text=Mean Duration")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=Median Duration")).toBeVisible();
    await expect(page.locator("text=Min Confidence")).toBeVisible();
    // Events table rows — check region ID prefix and confidence
    await expect(page.locator("td").filter({ hasText: "reg-" }).first()).toBeVisible();
    await expect(page.locator("td").filter({ hasText: "100.00s" }).first()).toBeVisible();
  });

  test("pre-selects region job from query param", async ({ page }) => {
    await page.goto(
      `/app/call-parsing/segment?regionJobId=${COMPLETE_REGION_JOB.id}`,
    );
    const regionSelect = page.locator("select").first();
    await expect(regionSelect).toHaveValue(COMPLETE_REGION_JOB.id);
  });
});

test.describe("Call Parsing Segment Training page", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("page loads with models and training sections", async ({ page }) => {
    await page.goto("/app/call-parsing/segment-training");
    await expect(page.locator("text=Segmentation Models")).toBeVisible();
    await expect(page.locator("text=Training Jobs")).toBeVisible();
  });

  test("models table shows model with metrics", async ({ page }) => {
    await page.goto("/app/call-parsing/segment-training");
    // pytorch_crnn only appears in the models table, not in training jobs
    const row = page.locator("tr").filter({ hasText: "pytorch_crnn" });
    await expect(row).toBeVisible();
    await expect(row).toContainText("crnn-bootstrap-v1");
    await expect(row).toContainText("0.81");
    await expect(row).toContainText("0.73");
  });

  test("training form shows dataset picker with sample count", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment-training");
    const datasetSelect = page.locator("select").first();
    const options = datasetSelect.locator("option");
    await expect(options.nth(1)).toContainText("bootstrap-orcasound-v1");
    await expect(options.nth(1)).toContainText("243 samples");
  });

  test("training jobs table shows completed job with metrics", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment-training");
    const row = page
      .locator("tr")
      .filter({ hasText: "30 ep" });
    await expect(row).toBeVisible();
    await expect(row).toContainText("lr=0.001");
  });
});

test.describe("Sidebar navigation", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("Call Parsing group shows Detection, Segment, and Segment Training links", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/detection");
    const nav = page.locator("nav");
    await expect(nav.locator("a", { hasText: "Detection" })).toBeVisible();
    await expect(nav.locator("a", { hasText: /^Segment$/ })).toBeVisible();
    await expect(
      nav.locator("a", { hasText: "Segment Training" }),
    ).toBeVisible();
  });

  test("navigation between pages works", async ({ page }) => {
    await page.goto("/app/call-parsing/detection");
    const nav = page.locator("nav");

    await nav.locator("a", { hasText: /^Segment$/ }).click();
    await expect(page).toHaveURL(/\/call-parsing\/segment$/);
    await expect(page.locator("text=New Segmentation Job")).toBeVisible();

    await nav.locator("a", { hasText: "Segment Training" }).click();
    await expect(page).toHaveURL(/\/call-parsing\/segment-training$/);
    await expect(page.locator("text=Segmentation Models")).toBeVisible();
  });
});

test.describe("Detection page Segment button", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("Segment button appears on completed jobs only", async ({ page }) => {
    await page.goto("/app/call-parsing/detection");
    const completeRow = page
      .locator("tr")
      .filter({ hasText: "complete" })
      .filter({ hasText: "North San Juan" });
    await expect(
      completeRow.locator("button", { hasText: "Segment →" }),
    ).toBeVisible();

    const failedRow = page.locator("tr").filter({ hasText: "failed" });
    await expect(
      failedRow.locator("button", { hasText: "Segment →" }),
    ).toHaveCount(0);
  });

  test("Segment button navigates to segment page with regionJobId", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/detection");
    const completeRow = page
      .locator("tr")
      .filter({ hasText: "complete" })
      .filter({ hasText: "North San Juan" });
    await completeRow.locator("button", { hasText: "Segment →" }).click();
    await expect(page).toHaveURL(
      new RegExp(`/call-parsing/segment\\?regionJobId=${COMPLETE_REGION_JOB.id}`),
    );
  });
});
