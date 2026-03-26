import { expect, test, type Page } from "@playwright/test";

/**
 * Tests for the timeline labeling workflow.
 *
 * These tests use the mocked timeline setup pattern from timeline.spec.ts,
 * so they run without a real backend — the data is provided via page.route().
 *
 * All tests share COMPLETE_JOB fixture data and mock API responses.
 */

const MODEL = {
  id: "model-label-1",
  name: "Label Test Model",
  model_path: "/tmp/model.joblib",
  model_version: "perch_v1",
  vector_dim: 1280,
  window_size_seconds: 5,
  target_sample_rate: 32000,
  feature_config: null,
  training_summary: null,
  training_job_id: null,
  created_at: "2026-03-01T00:00:00Z",
  updated_at: "2026-03-01T00:00:00Z",
};

const HYDROPHONE = {
  id: "rpi_north_sjc",
  name: "North San Juan Channel",
  location: "San Juan Channel",
  provider_kind: "orcasound_hls",
};

const COMPLETE_JOB = {
  id: "job-label-1",
  status: "complete",
  classifier_model_id: MODEL.id,
  audio_folder: null,
  confidence_threshold: 0.5,
  hop_seconds: 1.0,
  high_threshold: 0.7,
  low_threshold: 0.45,
  output_tsv_path: "/tmp/detections.tsv",
  result_summary: { n_spans: 2, time_covered_sec: 300 },
  error_message: null,
  files_processed: null,
  files_total: null,
  extract_status: null,
  extract_error: null,
  extract_summary: null,
  hydrophone_id: HYDROPHONE.id,
  hydrophone_name: HYDROPHONE.name,
  start_timestamp: 1751644800,
  end_timestamp: 1751645100,
  segments_processed: 5,
  segments_total: 5,
  time_covered_sec: 300,
  alerts: null,
  local_cache_path: null,
  created_at: "2026-03-01T00:00:00Z",
  updated_at: "2026-03-01T00:00:00Z",
};

// Two detection rows: one humpback, one unlabeled
const MOCK_DETECTIONS = [
  {
    row_id: "row-label-1",
    filename: "test.flac",
    start_sec: 30,
    end_sec: 35,
    avg_confidence: 0.85,
    peak_confidence: 0.95,
    n_windows: 1,
    humpback: 1,
    orca: 0,
    ship: 0,
    background: 0,
  },
  {
    row_id: "row-label-2",
    filename: "test.flac",
    start_sec: 120,
    end_sec: 125,
    avg_confidence: 0.61,
    peak_confidence: 0.78,
    n_windows: 1,
    humpback: 0,
    orca: 0,
    ship: 0,
    background: 0,
  },
];

async function setupMocks(page: Page, opts?: { withDetections?: boolean }) {
  const detections = opts?.withDetections ? MOCK_DETECTIONS : [];

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
  // Tile images 404 — canvas will just show nothing
  await page.route(/\/classifier\/hydrophone-detection-jobs\/[^/]+\/spectrogram-tile/, (route) =>
    route.fulfill({ status: 404 }),
  );
  // Mock prepare endpoint
  await page.route(/\/classifier\/hydrophone-detection-jobs\/[^/]+\/prepare-timeline/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    }),
  );
  // Mock prepare status
  await page.route(/\/classifier\/hydrophone-detection-jobs\/[^/]+\/prepare-status/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({}),
    }),
  );
  // Mock label save (PATCH) — return ok
  await page.route(/\/classifier\/detection-jobs\/[^/]+\/labels/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok", updated: 1 }),
    }),
  );
}

/** Navigate to the test timeline and wait for it to finish loading. */
async function navigateToTimeline(page: Page) {
  await page.goto(`/app/classifier/timeline/${COMPLETE_JOB.id}`);
  await expect(page.locator("text=Back to Jobs")).toBeVisible({ timeout: 10_000 });
}

/** Zoom into 5m level (enables label mode). */
async function zoomTo5m(page: Page) {
  const btn5m = page.locator("button", { hasText: "5m" });
  await expect(btn5m).toBeVisible();
  await btn5m.click();
  // Verify the zoom changed (5m now has accent color)
  await expect(btn5m).toHaveCSS("color", "rgb(112, 224, 192)");
}

/** Zoom into 1m level (also enables label mode). */
async function zoomTo1m(page: Page) {
  const btn1m = page.locator("button", { hasText: "1m" });
  await expect(btn1m).toBeVisible();
  await btn1m.click();
  await expect(btn1m).toHaveCSS("color", "rgb(112, 224, 192)");
}

test.describe("Timeline Labeling", () => {
  test("Label button is disabled at wide zoom levels", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);

    // Default zoom is "1h" — Label button should be disabled
    const labelBtn = page.locator("button", { hasText: "Label" });
    await expect(labelBtn).toBeVisible();
    await expect(labelBtn).toBeDisabled();

    // Also verify at 15m — still too wide for label mode
    const btn15m = page.locator("button", { hasText: "15m" });
    await btn15m.click();
    await expect(labelBtn).toBeDisabled();
  });

  test("Label button becomes enabled at 5m zoom", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);

    const labelBtn = page.locator("button", { hasText: "Label" });
    await expect(labelBtn).toBeDisabled();

    await zoomTo5m(page);

    await expect(labelBtn).toBeEnabled();
  });

  test("Label button becomes enabled at 1m zoom", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);

    const labelBtn = page.locator("button", { hasText: "Label" });
    await expect(labelBtn).toBeDisabled();

    await zoomTo1m(page);

    await expect(labelBtn).toBeEnabled();
  });

  test("enter label mode shows LabelToolbar", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo5m(page);

    // LabelToolbar should not be visible yet
    await expect(page.locator("button", { hasText: "Select" })).not.toBeVisible();

    const labelBtn = page.locator("button", { hasText: "Label" });
    await labelBtn.click();

    // Toolbar should now appear: Select/Add mode buttons
    await expect(page.locator("button", { hasText: "Select" })).toBeVisible();
    await expect(page.locator("button", { hasText: "Add" })).toBeVisible();

    // Label radio options visible
    await expect(page.locator("text=humpback")).toBeVisible();
    await expect(page.locator("text=orca")).toBeVisible();
    await expect(page.locator("text=ship")).toBeVisible();
    await expect(page.locator("text=background")).toBeVisible();

    // Action buttons visible
    await expect(page.locator("button", { hasText: "Save" })).toBeVisible();
    await expect(page.locator("button", { hasText: "Cancel" })).toBeVisible();
  });

  test("exit label mode hides LabelToolbar", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo5m(page);

    const labelBtn = page.locator("button", { hasText: "Label" });
    await labelBtn.click();

    // Confirm toolbar is visible
    await expect(page.locator("button", { hasText: "Select" })).toBeVisible();

    // Click Cancel — no dirty state, so no confirm dialog
    await page.locator("button", { hasText: "Cancel" }).click();

    // Toolbar should be gone
    await expect(page.locator("button", { hasText: "Select" })).not.toBeVisible();
    await expect(page.locator("button", { hasText: "Add" })).not.toBeVisible();
  });

  test("Save button starts disabled (not dirty)", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();

    // Save button should be disabled — no edits yet
    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeVisible();
    await expect(saveBtn).toBeDisabled();
  });

  test("clicking Add mode button activates it", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();

    // Initially Select mode is active — it has the accent background color
    const selectBtn = page.locator("button", { hasText: "Select" });
    const addBtn = page.locator("button", { hasText: "Add" });
    await expect(selectBtn).toBeVisible();
    await expect(addBtn).toBeVisible();

    // Select is active: has accent background (rgb(112, 224, 192))
    await expect(selectBtn).toHaveCSS("background-color", "rgb(112, 224, 192)");

    // Click Add
    await addBtn.click();

    // Now Add should be active and Select not
    await expect(addBtn).toHaveCSS("background-color", "rgb(112, 224, 192)");
  });

  test("label editor canvas appears in label mode", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    // Before entering label mode — editor not in DOM
    await expect(page.locator('[data-testid="label-editor"]')).not.toBeVisible();

    await page.locator("button", { hasText: "Label" }).click();

    // After entering label mode — editor overlay is in DOM and visible
    await expect(page.locator('[data-testid="label-editor"]')).toBeVisible();
  });

  test("clicking in add mode on label-editor marks save as dirty", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();
    await page.locator("button", { hasText: "Add" }).click();

    // Confirm we're in add mode (cursor changes but hard to assert — check active bg)
    const addBtn = page.locator("button", { hasText: "Add" });
    await expect(addBtn).toHaveCSS("background-color", "rgb(112, 224, 192)");

    // Save is still disabled before any click
    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeDisabled();

    // Click in the center of the label editor to add a label
    const editor = page.locator('[data-testid="label-editor"]');
    await expect(editor).toBeVisible();
    const editorBox = await editor.boundingBox();
    if (editorBox) {
      await page.mouse.click(editorBox.x + editorBox.width / 2, editorBox.y + editorBox.height / 2);
    }

    // Save should now be enabled (dirty state)
    await expect(saveBtn).toBeEnabled({ timeout: 3_000 });
  });

  test("Save button shows dirty indicator dot when edits exist", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();
    await page.locator("button", { hasText: "Add" }).click();

    const editor = page.locator('[data-testid="label-editor"]');
    await expect(editor).toBeVisible();
    const editorBox = await editor.boundingBox();
    if (editorBox) {
      await page.mouse.click(editorBox.x + editorBox.width / 2, editorBox.y + editorBox.height / 2);
    }

    // When dirty, Save button has accent color border style
    // The LabelToolbar sets borderColor to COLORS.accent when isDirty
    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeEnabled({ timeout: 3_000 });
    // Accent color (#70e0c0) = rgb(112, 224, 192)
    await expect(saveBtn).toHaveCSS("border-color", "rgb(112, 224, 192)");
  });

  test("Save button triggers save and returns to non-dirty state", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();
    await page.locator("button", { hasText: "Add" }).click();

    const editor = page.locator('[data-testid="label-editor"]');
    await expect(editor).toBeVisible();
    const editorBox = await editor.boundingBox();
    if (editorBox) {
      await page.mouse.click(editorBox.x + editorBox.width / 2, editorBox.y + editorBox.height / 2);
    }

    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeEnabled({ timeout: 3_000 });

    // Click Save — the mock endpoint returns ok
    await saveBtn.click();

    // After save, Save button returns to disabled (not dirty)
    await expect(saveBtn).toBeDisabled({ timeout: 5_000 });
  });

  test("dirty state confirm dialog appears on Cancel", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();
    await page.locator("button", { hasText: "Add" }).click();

    // Add a label to create dirty state
    const editor = page.locator('[data-testid="label-editor"]');
    await expect(editor).toBeVisible();
    const editorBox = await editor.boundingBox();
    if (editorBox) {
      await page.mouse.click(editorBox.x + editorBox.width / 2, editorBox.y + editorBox.height / 2);
    }

    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeEnabled({ timeout: 3_000 });

    // Set up dialog handler to dismiss (click Cancel in the confirm dialog)
    let dialogSeen = false;
    page.once("dialog", (dialog) => {
      dialogSeen = true;
      dialog.dismiss(); // dismiss = don't discard changes
    });

    // Click Cancel in the toolbar — should trigger confirm dialog
    await page.locator("button", { hasText: "Cancel" }).click();

    // Wait a tick for dialog handling
    await page.waitForTimeout(500);

    // Dialog should have appeared
    expect(dialogSeen).toBe(true);

    // We dismissed (kept changes), so we should still be in label mode
    await expect(page.locator("button", { hasText: "Select" })).toBeVisible();
    await expect(page.locator("button", { hasText: "Add" })).toBeVisible();
  });

  test("dirty state confirm dialog: accepting discards changes and exits label mode", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();
    await page.locator("button", { hasText: "Add" }).click();

    // Add a label to create dirty state
    const editor = page.locator('[data-testid="label-editor"]');
    await expect(editor).toBeVisible();
    const editorBox = await editor.boundingBox();
    if (editorBox) {
      await page.mouse.click(editorBox.x + editorBox.width / 2, editorBox.y + editorBox.height / 2);
    }

    const saveBtn = page.locator("button", { hasText: "Save" });
    await expect(saveBtn).toBeEnabled({ timeout: 3_000 });

    // Accept the confirm dialog (discard changes)
    page.once("dialog", (dialog) => dialog.accept());

    await page.locator("button", { hasText: "Cancel" }).click();

    await page.waitForTimeout(500);

    // Toolbar should be gone after accepting discard
    await expect(page.locator("button", { hasText: "Select" })).not.toBeVisible();
  });

  test("humpback is the default selected label type", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo5m(page);

    await page.locator("button", { hasText: "Label" }).click();

    // The humpback radio input should be checked by default
    const humpbackRadio = page.locator('input[type="radio"][value="humpback"]');
    await expect(humpbackRadio).toBeChecked();

    // Other label types should not be checked
    await expect(page.locator('input[type="radio"][value="orca"]')).not.toBeChecked();
    await expect(page.locator('input[type="radio"][value="ship"]')).not.toBeChecked();
    await expect(page.locator('input[type="radio"][value="background"]')).not.toBeChecked();
  });

  test("label type can be changed via radio buttons", async ({ page }) => {
    await setupMocks(page);
    await navigateToTimeline(page);
    await zoomTo5m(page);

    await page.locator("button", { hasText: "Label" }).click();

    // humpback is default
    const humpbackRadio = page.locator('input[type="radio"][value="humpback"]');
    const orcaRadio = page.locator('input[type="radio"][value="orca"]');
    await expect(humpbackRadio).toBeChecked();

    // Click the orca label text (radio input is hidden; the label wraps it)
    // The label has the text "orca" — click it to select
    await page.locator('label', { hasText: /^orca$/ }).click();

    await expect(orcaRadio).toBeChecked();
    await expect(humpbackRadio).not.toBeChecked();
  });

  test("Delete button is disabled when no bar is selected", async ({ page }) => {
    await setupMocks(page, { withDetections: true });
    await navigateToTimeline(page);
    await zoomTo1m(page);

    await page.locator("button", { hasText: "Label" }).click();

    // No bar selected yet — Delete should be disabled
    const deleteBtn = page.locator("button", { hasText: "Delete" });
    await expect(deleteBtn).toBeVisible();
    await expect(deleteBtn).toBeDisabled();
  });
});
