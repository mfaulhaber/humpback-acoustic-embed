import { expect, test, type Page } from "@playwright/test";

const MODEL = {
  id: "model-utc-1",
  name: "UTC Test Model",
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

function buildHydrophoneJob(overrides: Record<string, unknown> = {}) {
  return {
    id: "job-utc-1",
    status: "complete",
    classifier_model_id: MODEL.id,
    audio_folder: null,
    confidence_threshold: 0.5,
    hop_seconds: 1.0,
    high_threshold: 0.7,
    low_threshold: 0.45,
    output_tsv_path: "/tmp/detections.tsv",
    result_summary: { n_spans: 1, time_covered_sec: 60 },
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
    segments_processed: 12,
    segments_total: 12,
    time_covered_sec: 3600,
    alerts: null,
    local_cache_path: null,
    created_at: "2026-03-09T00:00:00Z",
    updated_at: "2026-03-09T00:00:00Z",
    ...overrides,
  };
}

async function mockHydrophonePageApis({
  page,
  jobs,
  onCreate,
  detectionRows,
}: {
  page: Page;
  jobs: Array<Record<string, unknown>>;
  onCreate?: (body: Record<string, unknown>) => void;
  detectionRows?: Array<Record<string, unknown>>;
}) {
  let jobsState = [...jobs];

  await page.route("**/classifier/training-jobs", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    });
  });

  await page.route("**/classifier/models", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([MODEL]),
    });
  });

  await page.route("**/classifier/hydrophones", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([HYDROPHONE]),
    });
  });

  await page.route("**/classifier/hydrophone-detection-jobs", async (route) => {
    const method = route.request().method();
    if (method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(jobsState),
      });
      return;
    }

    if (method === "POST") {
      const raw = route.request().postData() ?? "{}";
      const body = JSON.parse(raw) as Record<string, unknown>;
      onCreate?.(body);
      const createdJob = buildHydrophoneJob({
        id: "job-utc-created",
        status: "queued",
        output_tsv_path: null,
        result_summary: null,
        start_timestamp: body.start_timestamp,
        end_timestamp: body.end_timestamp,
      });
      jobsState = [createdJob];
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(createdJob),
      });
      return;
    }

    await route.fallback();
  });

  await page.route(/\/classifier\/detection-jobs\/[^/]+\/content$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(detectionRows ?? []),
    });
  });
}

test.describe("Hydrophone UTC timezone semantics", () => {
  test("submits date range picker inputs as UTC timestamps", async ({ page }) => {
    // Tall viewport so the dual-month popover + time inputs fit
    await page.setViewportSize({ width: 1280, height: 1200 });

    const monthFormatter = new Intl.DateTimeFormat("en-US", {
      month: "long",
      year: "numeric",
    });
    const currentMonth = new Date();
    const targetMonth = new Date(2025, 6, 1);
    const currentMonthLabel = monthFormatter.format(new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1));
    const previousYearLabel = monthFormatter.format(
      new Date(currentMonth.getFullYear() - 1, currentMonth.getMonth(), 1),
    );
    const targetMonthLabel = monthFormatter.format(targetMonth);

    let capturedBody: Record<string, unknown> | null = null;
    await mockHydrophonePageApis({
      page,
      jobs: [],
      onCreate: (body) => {
        capturedBody = body;
      },
    });

    await page.goto("/app/classifier/hydrophone");

    await page
      .locator("label", { hasText: "Hydrophone" })
      .locator("..")
      .locator("select")
      .selectOption(HYDROPHONE.id);
    await page
      .locator("label", { hasText: "Classifier Model" })
      .locator("..")
      .locator("select")
      .selectOption(MODEL.id);

    // Open the date range picker
    await page.getByTestId("date-range-trigger").click();

    const prevYearBtn = page.getByRole("button", { name: "Go to the Previous Year" });
    const prevBtn = page.getByRole("button", { name: "Go to the Previous Month" });
    const nextBtn = page.getByRole("button", { name: "Go to the Next Month" });
    const nextYearBtn = page.getByRole("button", { name: "Go to the Next Year" });

    await expect(page.getByText(currentMonthLabel)).toBeVisible();

    await prevYearBtn.click();
    await expect(page.getByText(previousYearLabel)).toBeVisible();

    await nextYearBtn.click();
    await expect(page.getByText(currentMonthLabel)).toBeVisible();

    // Navigate to July 2025 using the new year jump plus the existing month step controls.
    const deltaMonths =
      (targetMonth.getFullYear() - currentMonth.getFullYear()) * 12 +
      (targetMonth.getMonth() - currentMonth.getMonth());
    const navigateBackward = deltaMonths < 0;
    const yearStepCount = Math.floor(Math.abs(deltaMonths) / 12);
    const monthStepCount = Math.abs(deltaMonths) % 12;

    for (let i = 0; i < yearStepCount; i++) {
      await (navigateBackward ? prevYearBtn : nextYearBtn).click({ force: true });
    }
    for (let i = 0; i < monthStepCount; i++) {
      await (navigateBackward ? prevBtn : nextBtn).click({ force: true });
    }
    await expect(page.getByText(targetMonthLabel)).toBeVisible();

    // In range mode: click day 3 for start, day 5 for end (two different days required)
    const julyGrid = page.getByRole("grid", { name: targetMonthLabel });
    await julyGrid.getByRole("button", { name: /July 3rd/ }).click({ force: true });
    await julyGrid.getByRole("button", { name: /July 5th/ }).click({ force: true });

    // Fill start time and end time
    await page.getByTestId("start-time-input").fill("09:00");
    await page.getByTestId("end-time-input").fill("10:00");

    // Verify UTC label
    await expect(page.getByText("All times are UTC.")).toBeVisible();

    // Apply
    await page.getByTestId("date-range-apply").click();

    // Submit the form
    await page.locator("button", { hasText: "Start Detection" }).click();

    // July 3, 2025 09:00 UTC = 1751533200, July 5, 2025 10:00 UTC = 1751709600
    await expect.poll(() => capturedBody).not.toBeNull();
    expect(capturedBody?.start_timestamp).toBe(1751533200);
    expect(capturedBody?.end_timestamp).toBe(1751709600);
  });

  test("renders hydrophone job range and detection ranges in UTC format", async ({ page }) => {
    await mockHydrophonePageApis({
      page,
      jobs: [
        buildHydrophoneJob({
          id: "job-utc-display",
          start_timestamp: 1751644800, // 2025-07-04 16:00:00Z
          end_timestamp: 1751648400, // 2025-07-04 17:00:00Z
        }),
      ],
      detectionRows: [
        {
          start_utc: 1751647810,
          end_utc: 1751647816,
          avg_confidence: 0.82,
          peak_confidence: 0.86,
          n_windows: 2,
          raw_start_utc: 1751647810,
          raw_end_utc: 1751647816,
          humpback: null,
          ship: null,
          background: null,
        },
      ],
    });

    await page.goto("/app/classifier/hydrophone");

    await expect(page.locator("th", { hasText: "Date Range (UTC)" })).toBeVisible();

    const completedRow = page
      .locator("table tbody tr")
      .filter({ hasText: "complete" })
      .first();
    await expect(completedRow).toBeVisible();

    await expect(completedRow).toContainText("2025-07-04 16:00 UTC");
    await expect(completedRow).toContainText("2025-07-04 17:00 UTC");

    await completedRow.locator("td").nth(1).getByRole("button").click({ force: true });
    const expandedCell = page.locator("td[colspan]").first();
    await expect(expandedCell).toBeVisible();
    const innerTable = expandedCell.locator("table");
    await expect(innerTable).toBeVisible();

    await expect(innerTable.locator(".clip-range").first()).toContainText("Z_");
    await expect(innerTable.locator(".raw-range")).toHaveCount(0);
  });
});
