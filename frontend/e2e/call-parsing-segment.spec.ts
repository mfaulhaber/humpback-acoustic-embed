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

const REGIONS = [
  {
    region_id: "reg-aaaa",
    start_sec: 95.0,
    end_sec: 120.0,
    padded_start_sec: 94.0,
    padded_end_sec: 121.0,
    max_score: 0.95,
    mean_score: 0.88,
    n_windows: 5,
  },
  {
    region_id: "reg-bbbb",
    start_sec: 200.0,
    end_sec: 210.0,
    padded_start_sec: 199.0,
    padded_end_sec: 211.0,
    max_score: 0.78,
    mean_score: 0.72,
    n_windows: 2,
  },
];

const BOUNDARY_CORRECTIONS = [
  {
    id: "bc-1",
    event_segmentation_job_id: COMPLETE_SEG_JOB.id,
    event_id: "ev-1",
    region_id: "reg-aaaa",
    correction_type: "adjust",
    start_sec: 100.2,
    end_sec: 102.3,
    created_at: "2026-04-12T05:00:00Z",
    updated_at: "2026-04-12T05:00:00Z",
  },
];

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
  await page.route("**/call-parsing/region-jobs/*/regions", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(REGIONS),
    }),
  );
  await page.route("**/call-parsing/segmentation-jobs/*/corrections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(BOUNDARY_CORRECTIONS),
    }),
  );
  // 1x1 transparent PNG for tile requests
  const TINY_PNG = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB" +
      "Nl7BcQAAAABJRU5ErkJggg==",
    "base64",
  );
  await page.route("**/call-parsing/region-jobs/*/tile*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "image/png",
      body: TINY_PNG,
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

test.describe("Segment page tabs", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("page loads with Jobs and Review tabs", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    await expect(page.getByRole("tab", { name: "Jobs" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Review" })).toBeVisible();
  });

  test("Jobs tab is active by default and shows job content", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment");
    await expect(page.getByRole("tab", { name: "Jobs" })).toHaveAttribute(
      "data-state",
      "active",
    );
    await expect(page.locator("text=New Segmentation Job")).toBeVisible();
  });

  test("clicking Review tab shows review workspace", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    await page.getByRole("tab", { name: "Review" }).click();
    await expect(page.getByRole("tab", { name: "Review" })).toHaveAttribute(
      "data-state",
      "active",
    );
    await expect(page.locator("#review-job-select")).toBeVisible();
  });

  test("tab selection persists in URL query parameter", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    await page.getByRole("tab", { name: "Review" }).click();
    await expect(page).toHaveURL(/tab=review/);
    await page.getByRole("tab", { name: "Jobs" }).click();
    await expect(page).toHaveURL(/tab=jobs/);
  });

  test("navigating with tab=review opens Review tab directly", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await expect(page.getByRole("tab", { name: "Review" })).toHaveAttribute(
      "data-state",
      "active",
    );
    await expect(page.locator("#review-job-select")).toBeVisible();
  });
});

test.describe("Segment Review workspace", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("selecting a job populates the region sidebar", async ({ page }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    const select = page.locator("#review-job-select");
    await select.selectOption(COMPLETE_SEG_JOB.id);
    // Region sidebar should show regions
    await expect(page.locator("text=Regions (2)")).toBeVisible();
    // Use sidebar buttons to avoid matching toolbar "Region 1:35 – 2:00"
    await expect(page.locator("button").filter({ hasText: "1:35" })).toBeVisible();
    await expect(page.locator("button").filter({ hasText: "3:20" })).toBeVisible();
  });

  test("region sidebar shows event counts and correction status", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.locator("text=Regions (2)")).toBeVisible();
    // First region: 2 events, 1 edited (ev-1 has a correction)
    const firstRegion = page.locator("button").filter({ hasText: "1:35" });
    await expect(firstRegion).toContainText("2 events");
    await expect(firstRegion).toContainText("1 edited");
    // Second region: 0 events
    const secondRegion = page.locator("button").filter({ hasText: "3:20" });
    await expect(secondRegion).toContainText("0 events");
  });

  test("clicking a region highlights it and shows spectrogram", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.locator("text=Regions (2)")).toBeVisible();
    // First region is auto-selected — spectrogram viewport should render
    await expect(page.getByTestId("spectrogram-viewport")).toBeVisible();
    // Click second region
    const secondRegion = page.locator("button").filter({ hasText: "3:20" });
    await secondRegion.click();
    // Viewport should still be visible (now for the second region)
    await expect(page.getByTestId("spectrogram-viewport")).toBeVisible();
  });

  test("first region auto-selected on job load shows spectrogram", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("spectrogram-viewport")).toBeVisible();

    // Two-layer overlay container is in place on Segment Review.
    const band = page.getByTestId("overlay-band-layer").first();
    await expect(band).toBeAttached();
    const overflow = await band.evaluate((el) => getComputedStyle(el).overflow);
    expect(overflow).toBe("hidden");
    await expect(page.getByTestId("overlay-tooltip-layer").first()).toBeAttached();
  });

  test("event bars render on spectrogram for selected region", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("spectrogram-viewport")).toBeVisible();
    // Events ev-1 and ev-2 belong to reg-aaaa (auto-selected first region)
    await expect(page.getByTestId("event-bar-ev-1")).toBeVisible();
    await expect(page.getByTestId("event-bar-ev-2")).toBeVisible();
  });

  test("clicking an event bar selects it", async ({ page }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("event-bar-ev-1")).toBeVisible();
    await page.getByTestId("event-bar-ev-1").click();
    // Selected bar should have ring class (white ring)
    await expect(page.getByTestId("event-bar-ev-1")).toHaveClass(/ring-2/);
  });

  test("event bar shows correction state from saved corrections", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    // ev-1 has a saved "adjust" correction
    await expect(page.getByTestId("event-bar-ev-1")).toHaveAttribute(
      "data-correction",
      "adjust",
    );
    // ev-2 has no correction
    await expect(page.getByTestId("event-bar-ev-2")).toHaveAttribute(
      "data-correction",
      "none",
    );
  });

  test("sidebar shows correction progress legend", async ({ page }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.locator("text=reviewed")).toBeVisible();
    await expect(page.locator("text=partial")).toBeVisible();
    await expect(page.locator("text=pending")).toBeVisible();
  });
});

test.describe("EventDetailPanel and ReviewToolbar", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("first event is auto-selected and detail panel is populated", async ({ page }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("event-bar-ev-2")).toBeVisible();
    // First event is auto-selected — detail panel should already show event info
    await expect(page.getByText("Event", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Play Slice" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Delete Event" })).toBeVisible();
    await expect(page.getByText("Confidence")).toBeVisible();
    // Clicking a different event updates the panel
    await page.getByTestId("event-bar-ev-2").click();
    await expect(page.getByText("Event", { exact: true })).toBeVisible();
  });

  test("selecting adjusted event shows original vs adjusted values", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("event-bar-ev-1")).toBeVisible();
    await page.getByTestId("event-bar-ev-1").click();
    // ev-1 has a saved "adjust" correction — should show "adjusted" badge
    await expect(page.locator("text=adjusted")).toBeVisible();
  });

  test("toolbar shows region summary and action buttons", async ({ page }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    // Toolbar should show region time range
    await expect(page.getByText(/Region 1:35/)).toBeVisible();
    // Action buttons
    await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();
    await expect(page.locator("button", { hasText: "+ Add" })).toBeVisible();
    await expect(page.locator("button", { hasText: "Save" })).toBeVisible();
    await expect(page.locator("button", { hasText: "Cancel" })).toBeVisible();
    await expect(page.locator("button", { hasText: "Retrain" })).toBeVisible();
  });

  test("Save button is disabled when not dirty", async ({ page }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.locator("text=Regions (2)")).toBeVisible();
    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeDisabled();
  });

  test("Retrain button is enabled when corrections exist and not dirty", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.locator("text=Regions (2)")).toBeVisible();
    const retrainBtn = page.locator("button", { hasText: "Retrain" });
    // Corrections are mocked — button should become enabled once data loads
    await expect(retrainBtn).toBeEnabled();
  });
});

test.describe("State management — save/cancel/dirty flow", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("deleting an event enables Save and shows unsaved count", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("event-bar-ev-2")).toBeVisible();

    // Save starts disabled
    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeDisabled();

    // Select ev-2 and delete it
    await page.getByTestId("event-bar-ev-2").click();
    await page.getByRole("button", { name: "Delete Event" }).click();

    // Save becomes enabled with badge
    await expect(saveBtn).toBeEnabled();
    await expect(page.locator("text=1 unsaved change")).toBeVisible();
  });

  test("Save POSTs corrections and clears pending state", async ({ page }) => {
    // Intercept the POST to corrections
    let postBody: unknown = null;
    await page.route(
      "**/call-parsing/segmentation-jobs/*/corrections",
      (route) => {
        if (route.request().method() === "POST") {
          postBody = route.request().postDataJSON();
          return route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify({ count: 1 }),
          });
        }
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(BOUNDARY_CORRECTIONS),
        });
      },
    );

    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("event-bar-ev-2")).toBeVisible();

    // Delete ev-2 to create a pending correction
    await page.getByTestId("event-bar-ev-2").click();
    await page.getByRole("button", { name: "Delete Event" }).click();

    // Click Save
    await page.locator("button", { hasText: "Save" }).click();

    // Wait for POST to fire
    await page.waitForTimeout(500);
    expect(postBody).toBeTruthy();
    const body = postBody as { corrections: Array<{ correction_type: string }> };
    expect(body.corrections).toHaveLength(1);
    expect(body.corrections[0].correction_type).toBe("delete");

    // Save goes back to disabled
    await expect(page.locator("button", { hasText: "Save" })).toBeDisabled();
  });

  test("Cancel discards pending changes", async ({ page }) => {
    // Override confirm dialog to auto-accept
    page.on("dialog", (dialog) => dialog.accept());

    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("event-bar-ev-2")).toBeVisible();

    // Delete ev-2
    await page.getByTestId("event-bar-ev-2").click();
    await page.getByRole("button", { name: "Delete Event" }).click();
    await expect(page.locator("text=1 unsaved change")).toBeVisible();

    // Cancel
    await page.locator("button", { hasText: "Cancel" }).click();

    // Pending state cleared — Save disabled, no unsaved indicator
    await expect(page.locator("button", { hasText: "Save" })).toBeDisabled();
    await expect(page.locator("text=unsaved change")).not.toBeVisible();
  });

  test("switching regions preserves pending edits", async ({ page }) => {
    await page.goto("/app/call-parsing/segment?tab=review");
    await page.locator("#review-job-select").selectOption(COMPLETE_SEG_JOB.id);
    await expect(page.getByTestId("event-bar-ev-2")).toBeVisible();

    // Create a pending edit in the first region
    await page.getByTestId("event-bar-ev-2").click();
    await page.getByRole("button", { name: "Delete Event" }).click();
    await expect(page.locator("text=1 unsaved change")).toBeVisible();

    // Switch to second region — no dialog, edits preserved
    await page.locator("button").filter({ hasText: "3:20" }).click();
    await expect(page.locator("text=1 unsaved change")).toBeVisible();

    // Switch back — edits still there
    await page.locator("button").filter({ hasText: "1:35" }).click();
    await expect(page.locator("text=1 unsaved change")).toBeVisible();
  });
});

test.describe("Call Parsing Segment Training page", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("page loads with models section", async ({ page }) => {
    await page.goto("/app/call-parsing/segment-training");
    await expect(page.locator("text=Segmentation Models")).toBeVisible();
  });

  test("models table shows model with metrics", async ({ page }) => {
    await page.goto("/app/call-parsing/segment-training");
    const row = page.locator("tr").filter({ hasText: "pytorch_crnn" });
    await expect(row).toBeVisible();
    await expect(row).toContainText("crnn-bootstrap-v1");
    await expect(row).toContainText("0.81");
    await expect(row).toContainText("0.73");
  });
});

test.describe("Jobs tab Review link", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("completed job row shows Review link", async ({ page }) => {
    await page.goto("/app/call-parsing/segment");
    const row = page.locator("tr").filter({ hasText: "complete" });
    await expect(row).toBeVisible();
    await expect(row.locator("a", { hasText: "Review" })).toBeVisible();
  });

  test("Review link only shown for complete jobs", async ({ page }) => {
    // Add a failed segmentation job to the mocks
    await page.route("**/call-parsing/segmentation-jobs", (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            COMPLETE_SEG_JOB,
            {
              ...COMPLETE_SEG_JOB,
              id: "sj-failed-1",
              status: "failed",
              event_count: null,
              error_message: "model error",
            },
          ]),
        });
      }
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(COMPLETE_SEG_JOB),
      });
    });
    await page.goto("/app/call-parsing/segment");
    // Complete row has Review link
    const completeRow = page.locator("tr").filter({ hasText: "complete" }).first();
    await expect(completeRow.locator("a", { hasText: "Review" })).toBeVisible();
    // Failed row does not
    const failedRow = page.locator("tr").filter({ hasText: "failed" });
    await expect(failedRow.locator("a", { hasText: "Review" })).toHaveCount(0);
  });

  test("clicking Review navigates to Review tab with job selected", async ({
    page,
  }) => {
    await page.goto("/app/call-parsing/segment");
    const row = page.locator("tr").filter({ hasText: "complete" });
    await row.locator("a", { hasText: "Review" }).click();
    await expect(page).toHaveURL(/tab=review/);
    await expect(page).toHaveURL(new RegExp(`reviewJobId=${COMPLETE_SEG_JOB.id}`));
    // Review workspace should be visible with the job select
    await expect(page.locator("#review-job-select")).toBeVisible();
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
