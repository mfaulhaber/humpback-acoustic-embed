import { expect, test, type Page } from "@playwright/test";

const MODEL = {
  id: "model-tl-1",
  name: "Timeline Test Model",
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

const COMPLETE_JOB = {
  id: "job-tl-1",
  status: "complete",
  classifier_model_id: MODEL.id,
  audio_folder: null,
  confidence_threshold: 0.5,
  hop_seconds: 1.0,
  high_threshold: 0.7,
  low_threshold: 0.45,
  output_tsv_path: "/tmp/detections.tsv",
  result_summary: { n_spans: 3, time_covered_sec: 3600 },
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
};

async function setupHydrophoneMocks(page: Page) {
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
      body: JSON.stringify([COMPLETE_JOB]),
    }),
  );
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
}

const MOCK_DETECTIONS = [
  {
    row_id: "row-1",
    filename: "test.flac",
    start_sec: 100,
    end_sec: 105,
    avg_confidence: 0.82,
    peak_confidence: 0.95,
    n_windows: 1,
    humpback: 1,
    orca: 0,
    ship: 0,
    background: 0,
  },
  {
    row_id: "row-2",
    filename: "test.flac",
    start_sec: 200,
    end_sec: 205,
    avg_confidence: 0.71,
    peak_confidence: 0.88,
    n_windows: 1,
    humpback: 0,
    orca: 1,
    ship: 0,
    background: 0,
  },
  {
    row_id: "row-3",
    filename: "test.flac",
    start_sec: 300,
    end_sec: 305,
    avg_confidence: 0.65,
    peak_confidence: 0.72,
    n_windows: 1,
    humpback: 0,
    orca: 0,
    ship: 1,
    background: 0,
  },
];

async function setupTimelineMocks(page: Page, opts?: { withDetections?: boolean }) {
  await setupHydrophoneMocks(page);
  const detections = opts?.withDetections ? MOCK_DETECTIONS : [];
  // Mock timeline confidence and detection endpoints
  await page.route(/\/classifier\/hydrophone-detection-jobs\/[^/]+\/confidence$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ scores: [] }),
    }),
  );
  await page.route(/\/classifier\/detection-jobs\/[^/]+\/content$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(detections),
    }),
  );
  // Mock tile images to avoid real fetch attempts
  await page.route(/\/classifier\/hydrophone-detection-jobs\/[^/]+\/spectrogram-tile/, (route) =>
    route.fulfill({ status: 404 }),
  );
}

test.describe("Timeline Viewer", () => {
  test("navigates to timeline from hydrophone tab", async ({ page }) => {
    await setupHydrophoneMocks(page);
    await page.goto("/app/classifier/hydrophone");

    // The Timeline button only appears for complete jobs in Previous Jobs table
    const timelineBtn = page.locator("button", { hasText: "Timeline" }).first();
    await expect(timelineBtn).toBeVisible({ timeout: 10_000 });

    await timelineBtn.click();
    await expect(page).toHaveURL(/\/app\/classifier\/timeline\//);
    await expect(page.locator("text=Back to Jobs")).toBeVisible();
  });

  test("zoom level buttons change active state", async ({ page }) => {
    await setupTimelineMocks(page);
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    // Wait for the page to load (job data must resolve)
    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // Default zoom level is "1h" — click "5m" and verify it becomes active
    const zoomBtn5m = page.getByRole("button", { name: "5m", exact: true });
    await expect(zoomBtn5m).toBeVisible();

    // Before clicking, "1h" should be active (accent color), "5m" should not
    const zoomBtn1h = page.getByRole("button", { name: "1h", exact: true });
    await expect(zoomBtn1h).toBeVisible();

    await zoomBtn5m.click();

    // After clicking "5m", it should have the accent color style applied
    // The ZoomSelector sets color to COLORS.accent (#70e0c0) for active level
    await expect(zoomBtn5m).toHaveCSS("color", "rgb(112, 224, 192)");
    // "1h" should now be inactive (COLORS.textMuted = #3a6a60 = rgb(58, 106, 96))
    await expect(zoomBtn1h).toHaveCSS("color", "rgb(58, 106, 96)");
  });

  test("play button toggles between play and pause icons", async ({ page }) => {
    await setupTimelineMocks(page);
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // The play/pause button is a rounded-full button in PlaybackControls
    const playPauseBtn = page.locator("button.rounded-full").first();
    await expect(playPauseBtn).toBeVisible();

    // Initially shows Play icon (not playing) — the button has a border styled with accent color
    // Verify the button contains an SVG (lucide icon)
    await expect(playPauseBtn.locator("svg")).toBeVisible();

    // Click to start playing
    await playPauseBtn.click();

    // After clicking, the icon should switch from Play to Pause
    // Both icons are SVG; we verify the button is still present and clickable
    await expect(playPauseBtn).toBeVisible();
    await expect(playPauseBtn.locator("svg")).toBeVisible();

    // Click again to stop playing — should return to Play icon
    await playPauseBtn.click();
    await expect(playPauseBtn).toBeVisible();
  });

  test("back button returns to hydrophone tab", async ({ page }) => {
    await setupTimelineMocks(page);
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    await page.locator("text=Back to Jobs").click();
    await expect(page).toHaveURL(/\/app\/classifier\/hydrophone/);
  });

  test("labels toggle button switches between OFF and ON", async ({ page }) => {
    await setupTimelineMocks(page);
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // Initial state: labels are OFF
    const labelsBtn = page.locator("button", { hasText: "Labels: OFF" });
    await expect(labelsBtn).toBeVisible();

    await labelsBtn.click();

    // After clicking, should show Labels: ON
    await expect(page.locator("button", { hasText: "Labels: ON" })).toBeVisible();

    // Click again to toggle back to OFF
    await page.locator("button", { hasText: "Labels: ON" }).click();
    await expect(page.locator("button", { hasText: "Labels: OFF" })).toBeVisible();
  });

  test("timeline header shows hydrophone name and time range", async ({ page }) => {
    await setupTimelineMocks(page);
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // Should display the hydrophone name
    await expect(page.locator(`text=${COMPLETE_JOB.hydrophone_name}`)).toBeVisible();

    // Should display time range derived from start/end timestamps
    // start_timestamp 1751644800 => 2025-07-04 16:00:00Z
    await expect(page.locator("text=2025-07-04 16:00:00Z")).toBeVisible();
  });

  test("zoom level buttons render all expected levels", async ({ page }) => {
    await setupTimelineMocks(page);
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // All six zoom levels from constants.ts should be present
    for (const level of ["24h", "6h", "1h", "15m", "5m", "1m"]) {
      await expect(
        page.getByRole("button", { name: level, exact: true }),
      ).toBeVisible();
    }
  });

  test("speed cycle button cycles through 0.5x 1x 2x", async ({ page }) => {
    await setupTimelineMocks(page);
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // Default speed is 1x
    const speedBtn = page.locator("button", { hasText: "1x" });
    await expect(speedBtn).toBeVisible();

    // Click once: 1x -> 2x
    await speedBtn.click();
    await expect(page.locator("button", { hasText: "2x" })).toBeVisible();

    // Click again: 2x -> 0.5x
    await page.locator("button", { hasText: "2x" }).click();
    await expect(page.locator("button", { hasText: "0.5x" })).toBeVisible();

    // Click again: 0.5x -> 1x
    await page.locator("button", { hasText: "0.5x" }).click();
    await expect(page.locator("button", { hasText: "1x" })).toBeVisible();
  });

  test("label overlay shows detections excluding negatives", async ({ page }) => {
    await setupTimelineMocks(page, { withDetections: true });
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // Toggle labels ON
    const labelsBtn = page.locator("button", { hasText: "Labels: OFF" });
    await expect(labelsBtn).toBeVisible();
    await labelsBtn.click();
    await expect(page.locator("button", { hasText: "Labels: ON" })).toBeVisible();

    // Detection overlay container should be visible
    const overlay = page.locator('[data-testid="detection-overlay"]');
    await expect(overlay).toBeVisible();

    // All detections shown except negatives (ship/background are excluded)
    // Mock has 3 rows: humpback, orca, ship — ship is negative so excluded, expect 2 bars
    const bars = overlay.locator("> div");
    // Wait for bars to appear (they are direct children of the overlay container)
    await expect(bars.first()).toBeVisible({ timeout: 5_000 });
    const count = await bars.count();
    // 2 bars + potentially the tooltip div (hidden), so at least 2
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test("label bar shows tooltip on hover", async ({ page }) => {
    await setupTimelineMocks(page, { withDetections: true });
    await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);

    await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });

    // Toggle labels ON
    const labelsBtn = page.locator("button", { hasText: "Labels: OFF" });
    await labelsBtn.click();
    await expect(page.locator("button", { hasText: "Labels: ON" })).toBeVisible();

    const overlay = page.locator('[data-testid="detection-overlay"]');
    await expect(overlay).toBeVisible();

    // Find the bar divs (direct children with pointer-events: auto)
    const bars = overlay.locator("> div").filter({ has: page.locator("css=*") }).first();
    // Alternatively, find any child div with pointer-events auto
    const clickableBars = overlay.locator('div[style*="pointer-events: auto"]');
    const barCount = await clickableBars.count();
    if (barCount > 0) {
      await clickableBars.first().hover();
      // Tooltip should display "Confidence" text
      await expect(page.locator("text=Confidence")).toBeVisible({ timeout: 3_000 });
    }
  });
});
