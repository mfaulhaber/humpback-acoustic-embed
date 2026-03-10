import { expect, test, type Page } from "@playwright/test";

const MODEL = {
  id: "model-cj-1",
  name: "Canceled Job Test Model",
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

const CANCELED_JOB = {
  id: "job-canceled-1",
  status: "canceled",
  classifier_model_id: MODEL.id,
  audio_folder: null,
  confidence_threshold: 0.5,
  hop_seconds: 1.0,
  high_threshold: 0.7,
  low_threshold: 0.45,
  output_tsv_path: "/tmp/detections.tsv",
  result_summary: { n_spans: 2, time_covered_sec: 1800 },
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
  segments_processed: 6,
  segments_total: 12,
  time_covered_sec: 1800,
  alerts: null,
  local_cache_path: null,
  created_at: "2026-03-09T00:00:00Z",
  updated_at: "2026-03-09T00:00:00Z",
};

const DETECTION_ROWS = [
  {
    filename: "20250704T165000Z.wav",
    start_sec: 10,
    end_sec: 16,
    avg_confidence: 0.82,
    peak_confidence: 0.86,
    n_windows: 2,
    detection_filename: "20250704T165010Z_20250704T165016Z.wav",
    extract_filename: "20250704T165010Z_20250704T165020Z.wav",
    hydrophone_name: "rpi_north_sjc",
    humpback: null,
    ship: null,
    background: null,
  },
  {
    filename: "20250704T165000Z.wav",
    start_sec: 30,
    end_sec: 40,
    avg_confidence: 0.75,
    peak_confidence: 0.79,
    n_windows: 3,
    detection_filename: "20250704T165030Z_20250704T165040Z.wav",
    extract_filename: "20250704T165030Z_20250704T165040Z.wav",
    hydrophone_name: "rpi_north_sjc",
    humpback: null,
    ship: null,
    background: null,
  },
];

async function setupMocks(page: Page) {
  let labelsSaved = false;

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
      body: JSON.stringify([CANCELED_JOB]),
    }),
  );
  await page.route(/\/classifier\/detection-jobs\/[^/]+\/content$/, (route) => {
    const rows = labelsSaved
      ? DETECTION_ROWS.map((r) => ({ ...r, humpback: 1 }))
      : DETECTION_ROWS;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(rows),
    });
  });
  await page.route(/\/classifier\/detection-jobs\/[^/]+\/labels$/, (route) => {
    labelsSaved = true;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok", updated: 2 }),
    });
  });
  await page.route("**/classifier/extraction-settings", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        positive_output_path: "/tmp/positive",
        negative_output_path: "/tmp/negative",
      }),
    }),
  );

  return { wasLabelsSaved: () => labelsSaved };
}

test.describe("Hydrophone canceled job functionality", () => {
  test("canceled job appears in Previous Jobs with expand chevron", async ({ page }) => {
    await setupMocks(page);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    // Canceled job should be in Previous Jobs
    const prevSection = page.locator("text=Previous Jobs");
    await expect(prevSection).toBeVisible();

    const canceledRow = page
      .locator("table tbody tr")
      .filter({ hasText: "canceled" })
      .first();
    await expect(canceledRow).toBeVisible();

    // Should have expand button
    const expandBtn = canceledRow.locator("td:nth-child(2) button");
    await expect(expandBtn).toBeVisible();
  });

  test("can expand canceled job to view detection content", async ({ page }) => {
    await setupMocks(page);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    const canceledRow = page
      .locator("table tbody tr")
      .filter({ hasText: "canceled" })
      .first();
    await canceledRow.locator("td:nth-child(2) button").click();

    const innerTable = page.locator("tr td[colspan] table");
    await expect(innerTable).toBeVisible({ timeout: 10_000 });

    // Should show detection rows
    const innerRows = innerTable.locator("tbody tr");
    await expect(innerRows).toHaveCount(2);
  });

  test("download TSV link visible for canceled job", async ({ page }) => {
    await setupMocks(page);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    const canceledRow = page
      .locator("table tbody tr")
      .filter({ hasText: "canceled" })
      .first();
    const downloadLink = canceledRow.locator("a", { hasText: "TSV" });
    await expect(downloadLink).toBeVisible();
  });

  test("results column shows hours:minutes format for canceled job", async ({ page }) => {
    await setupMocks(page);
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    const canceledRow = page
      .locator("table tbody tr")
      .filter({ hasText: "canceled" })
      .first();
    // time_covered_sec is 1800 = 0h 30m
    await expect(canceledRow).toContainText("0h 30m");
  });
});
