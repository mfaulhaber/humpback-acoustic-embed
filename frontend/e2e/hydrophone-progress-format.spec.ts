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

    // Active job panel should show 1h 30m (5432s = 1h 30m 32s → 1h 30m)
    // Target the Card that contains "Active Job" text
    const activeCard = page.locator("[class*=rounded-lg][class*=border]").filter({ hasText: "Active Job" });
    await expect(activeCard).toBeVisible();
    await expect(activeCard).toContainText("1h 30m");
    // Should NOT show the raw seconds format
    expect(await activeCard.locator("text=5432s").count()).toBe(0);
  });

  test("date picker accepts 24hr UTC format with space separator", async ({ page }) => {
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

    // Fill with valid 24hr format using space separator
    const startInput = page
      .locator("label", { hasText: "Start Date/Time (UTC)" })
      .locator("..")
      .locator("input");
    await startInput.fill("2025-07-04 23:30");

    // No validation error should appear
    expect(
      await page.locator("text=Format: YYYY-MM-DD HH:MM").count(),
    ).toBe(0);

    // Fill with invalid format — should show validation error
    await startInput.fill("07/04/2025 23:30");
    await expect(
      page.locator("text=Format: YYYY-MM-DD HH:MM (24hr UTC)").first(),
    ).toBeVisible();

    // Placeholder should show expected format
    const endInput = page
      .locator("label", { hasText: "End Date/Time (UTC)" })
      .locator("..")
      .locator("input");
    await expect(endInput).toHaveAttribute("placeholder", "YYYY-MM-DD HH:MM");
  });
});
