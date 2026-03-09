import { expect, test, type Page } from "@playwright/test";

const MODEL = {
  id: "model-pr-1",
  name: "PR Test Model",
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
};

function buildJob(overrides: Record<string, unknown> = {}) {
  return {
    id: "job-pr-1",
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
    created_at: "2026-03-09T00:00:00Z",
    updated_at: "2026-03-09T00:00:00Z",
    ...overrides,
  };
}

async function setupMocks(page: Page, initialStatus: string) {
  let currentStatus = initialStatus;
  let pauseCalled = false;
  let resumeCalled = false;
  let cancelCalled = false;

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
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([buildJob({ status: currentStatus })]),
      });
    }
    return route.fallback();
  });

  await page.route(/\/hydrophone-detection-jobs\/[^/]+\/pause$/, (route) => {
    pauseCalled = true;
    currentStatus = "paused";
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "paused" }),
    });
  });

  await page.route(/\/hydrophone-detection-jobs\/[^/]+\/resume$/, (route) => {
    resumeCalled = true;
    currentStatus = "running";
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "running" }),
    });
  });

  await page.route(/\/hydrophone-detection-jobs\/[^/]+\/cancel$/, (route) => {
    cancelCalled = true;
    currentStatus = "canceled";
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "canceled" }),
    });
  });

  return {
    getStatus: () => currentStatus,
    wasPauseCalled: () => pauseCalled,
    wasResumeCalled: () => resumeCalled,
    wasCancelCalled: () => cancelCalled,
  };
}

test.describe("Hydrophone Pause/Resume/Cancel", () => {
  test("running job shows Pause + Cancel buttons (no Stop)", async ({ page }) => {
    await setupMocks(page, "running");
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    const activeCard = page.locator("text=Active Job");
    await expect(activeCard).toBeVisible();

    await expect(page.locator("button", { hasText: "Pause" })).toBeVisible();
    await expect(page.locator("button", { hasText: "Cancel" })).toBeVisible();
    expect(await page.locator("button", { hasText: "Stop" }).count()).toBe(0);
  });

  test("pause transitions to paused state with Resume button", async ({ page }) => {
    const mocks = await setupMocks(page, "running");
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    await expect(page.locator("button", { hasText: "Pause" })).toBeVisible();
    await page.locator("button", { hasText: "Pause" }).click();

    await expect.poll(() => mocks.wasPauseCalled()).toBe(true);
    // After poll refetch, should show Resume
    await expect(page.locator("button", { hasText: "Resume" })).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("button", { hasText: "Cancel" })).toBeVisible();
    // Pause button should be gone
    expect(await page.locator("button", { hasText: "Pause" }).count()).toBe(0);
  });

  test("resume transitions back to running", async ({ page }) => {
    const mocks = await setupMocks(page, "paused");
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    await expect(page.locator("button", { hasText: "Resume" })).toBeVisible();
    await page.locator("button", { hasText: "Resume" }).click();

    await expect.poll(() => mocks.wasResumeCalled()).toBe(true);
    await expect(page.locator("button", { hasText: "Pause" })).toBeVisible({ timeout: 10_000 });
  });

  test("cancel from running calls cancel endpoint", async ({ page }) => {
    const mocks = await setupMocks(page, "running");
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    await page.locator("button", { hasText: "Cancel" }).click();
    await expect.poll(() => mocks.wasCancelCalled()).toBe(true);
  });

  test("cancel from paused calls cancel endpoint", async ({ page }) => {
    const mocks = await setupMocks(page, "paused");
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    await expect(page.locator("button", { hasText: "Cancel" })).toBeVisible();
    await page.locator("button", { hasText: "Cancel" }).click();
    await expect.poll(() => mocks.wasCancelCalled()).toBe(true);
  });

  test("paused job stays in Active panel (not Previous Jobs)", async ({ page }) => {
    await setupMocks(page, "paused");
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    await expect(page.locator("text=Active Job")).toBeVisible();
    // The paused badge should show in the active panel
    await expect(
      page.locator(".space-y-3").locator("text=paused").first(),
    ).toBeVisible();
    // Previous Jobs should not exist (only one job, and it's active/paused)
    expect(await page.locator("text=Previous Jobs").count()).toBe(0);
  });
});
