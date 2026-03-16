import { expect, test } from "@playwright/test";

const MODEL = {
  id: "model-pf-1",
  name: "Progress Format Test Model",
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
  id: "rpi_orcasound_lab",
  name: "Orcasound Lab",
  location: "San Juan Islands",
  provider_kind: "orcasound_hls",
};

test.describe("Hydrophone progress display and tab structure", () => {
  test("only Train and Hydrophone tabs exist (no Detect)", async ({ page }) => {
    // Mock APIs to avoid errors
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
    await page.route("**/classifier/hydrophone-detection-jobs", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
    );

    await page.goto("/app/classifier");

    // Train and Hydrophone sub-tabs should exist (within the tab bar)
    const tabBar = page.locator(".flex.gap-2.border-b");
    await expect(tabBar.locator("button", { hasText: "Train" })).toBeVisible();
    await expect(tabBar.locator("button", { hasText: "Hydrophone" })).toBeVisible();

    // Detect tab should NOT exist in the tab bar
    expect(await tabBar.locator("button", { hasText: "Detect" }).count()).toBe(0);
  });

  test("active job displays duration in hours:minutes format", async ({ page }) => {
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
    await page.route("**/classifier/hydrophone-detection-jobs", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([
          {
            id: "job-pf-1",
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
            end_timestamp: 1751662800,
            segments_processed: 50,
            segments_total: 100,
            time_covered_sec: 5432,
            alerts: null,
            local_cache_path: null,
            created_at: "2026-03-09T00:00:00Z",
            updated_at: "2026-03-09T00:00:00Z",
          },
        ]),
      }),
    );

    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    // Active jobs table should show 1h 30m (5432s = 1h 30m 32s → 1h 30m)
    await expect(page.getByRole("heading", { name: "Active Jobs" })).toBeVisible();
    const activeRow = page.locator("table tbody tr").filter({ hasText: "running" }).first();
    await expect(activeRow).toBeVisible();
    await expect(activeRow).toContainText("1h 30m");
    // Should NOT show the raw seconds format
    expect(await activeRow.locator("text=5432s").count()).toBe(0);
  });

  test("date range picker shows UTC label and time inputs", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 1200 });
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
    await page.route("**/classifier/hydrophone-detection-jobs", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
    );

    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    // Date range picker trigger is present with placeholder
    const trigger = page.getByTestId("date-range-trigger");
    await expect(trigger).toBeVisible();
    await expect(trigger).toContainText("Select date range (UTC)");

    // Open the picker
    await trigger.click();

    await expect(page.getByRole("button", { name: "Go to the Previous Year" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Go to the Previous Month" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Go to the Next Month" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Go to the Next Year" })).toBeVisible();

    // Time inputs default to 00:00
    await expect(page.getByTestId("start-time-input")).toHaveValue("00:00");
    await expect(page.getByTestId("end-time-input")).toHaveValue("00:00");

    // UTC label is visible inside the popover
    await expect(page.getByText("All times are UTC.")).toBeVisible();

    // Apply is disabled when no dates selected
    await expect(page.getByTestId("date-range-apply")).toBeDisabled();
  });
});
