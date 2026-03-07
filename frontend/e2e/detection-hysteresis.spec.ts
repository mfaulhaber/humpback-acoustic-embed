import { test, expect } from "@playwright/test";

/**
 * Tests for the detection hysteresis form controls (hop size, start/continue thresholds)
 * and that n_windows is preserved through label save round-trips.
 * Requires: backend running on :8000.
 */

test.describe("Detection hysteresis form controls", () => {
  test("form shows hop size and dual threshold inputs", async ({ page }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Detect" }).click();

    // Hop Size input
    const hopInput = page.locator('input[type="number"]').first();
    await expect(hopInput).toBeVisible();
    await expect(hopInput).toHaveValue("1");

    // Start Threshold slider (high_threshold)
    const startSlider = page
      .locator("label", { hasText: "Start Threshold" })
      .locator("..")
      .locator('input[type="range"]');
    await expect(startSlider).toBeVisible();
    expect(parseFloat(await startSlider.inputValue())).toBeCloseTo(0.7, 1);

    // Continue Threshold slider (low_threshold)
    const continueSlider = page
      .locator("label", { hasText: "Continue Threshold" })
      .locator("..")
      .locator('input[type="range"]');
    await expect(continueSlider).toBeVisible();
    expect(parseFloat(await continueSlider.inputValue())).toBeCloseTo(0.45, 1);
  });

  test("threshold labels update when sliders change", async ({ page }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Detect" }).click();

    // Change Start Threshold slider
    const startSlider = page
      .locator("label", { hasText: "Start Threshold" })
      .locator("..")
      .locator('input[type="range"]');
    await startSlider.fill("0.85");

    // Label should reflect new value
    const startLabel = page.locator("label", { hasText: "Start Threshold" });
    await expect(startLabel).toContainText("0.85");

    // Change Continue Threshold slider
    const continueSlider = page
      .locator("label", { hasText: "Continue Threshold" })
      .locator("..")
      .locator('input[type="range"]');
    await continueSlider.fill("0.30");

    const continueLabel = page.locator("label", {
      hasText: "Continue Threshold",
    });
    await expect(continueLabel).toContainText("0.30");
  });

  test("completed jobs show threshold values in table", async ({ page }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Detect" }).click();

    // Wait for detection jobs table
    const table = page.locator("table").first();
    const hasTable = await table
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasTable) {
      test.skip(true, "No detection jobs table visible");
      return;
    }

    // Find a completed job row
    const completedRow = page
      .locator("table tbody tr")
      .filter({ hasText: "complete" })
      .first();
    const hasRow = await completedRow
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasRow) {
      test.skip(true, "No completed detection job rows");
      return;
    }

    // Threshold column should show high/low values (e.g. "0.7/0.45")
    const thresholdCell = completedRow.locator("td").nth(4);
    const text = await thresholdCell.textContent();
    expect(text).toMatch(/\d+\.?\d*\/\d+\.?\d*/);
  });
});

test.describe("Detection job creation API with hysteresis params", () => {
  test("create endpoint validates high >= low threshold", async ({
    request,
  }) => {
    // Get a classifier model (if any)
    const modelsRes = await request.get(
      "http://localhost:8000/classifier/models",
    );
    expect(modelsRes.ok()).toBeTruthy();
    const models = await modelsRes.json();
    if (models.length === 0) {
      test.skip(true, "No classifier models available");
      return;
    }

    // Try to create a detection job with high < low (should fail validation)
    const res = await request.post(
      "http://localhost:8000/classifier/detection-jobs",
      {
        data: {
          classifier_model_id: models[0].id,
          audio_folder: "/nonexistent",
          confidence_threshold: 0.5,
          hop_seconds: 1.0,
          high_threshold: 0.3,
          low_threshold: 0.7,
        },
      },
    );
    expect(res.status()).toBe(422);
  });

  test("create endpoint validates hop_seconds > 0", async ({ request }) => {
    const modelsRes = await request.get(
      "http://localhost:8000/classifier/models",
    );
    expect(modelsRes.ok()).toBeTruthy();
    const models = await modelsRes.json();
    if (models.length === 0) {
      test.skip(true, "No classifier models available");
      return;
    }

    const res = await request.post(
      "http://localhost:8000/classifier/detection-jobs",
      {
        data: {
          classifier_model_id: models[0].id,
          audio_folder: "/nonexistent",
          hop_seconds: 0,
        },
      },
    );
    expect(res.status()).toBe(422);
  });

  test("detection content includes n_windows field", async ({ request }) => {
    const jobsRes = await request.get(
      "http://localhost:8000/classifier/detection-jobs",
    );
    expect(jobsRes.ok()).toBeTruthy();
    const jobs = await jobsRes.json();

    const completedJob = jobs.find(
      (j: { status: string; output_tsv_path: string | null }) =>
        j.status === "complete" && j.output_tsv_path,
    );
    if (!completedJob) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    // n_windows should be present in the response
    expect("n_windows" in rows[0]).toBe(true);

    // Completed job response should include hysteresis params
    expect(completedJob).toHaveProperty("hop_seconds");
    expect(completedJob).toHaveProperty("high_threshold");
    expect(completedJob).toHaveProperty("low_threshold");
  });

  test("n_windows survives label save round-trip", async ({ request }) => {
    const jobsRes = await request.get(
      "http://localhost:8000/classifier/detection-jobs",
    );
    const jobs = await jobsRes.json();
    const completedJob = jobs.find(
      (j: { status: string; output_tsv_path: string | null }) =>
        j.status === "complete" && j.output_tsv_path,
    );
    if (!completedJob) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    // Read content before label save
    const beforeRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/content`,
    );
    const beforeRows = await beforeRes.json();
    expect(beforeRows.length).toBeGreaterThan(0);

    const firstRow = beforeRows[0];
    const originalNWindows = firstRow.n_windows;

    // Save a label on the first row
    const putRes = await request.put(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/labels`,
      {
        data: [
          {
            filename: firstRow.filename,
            start_sec: firstRow.start_sec,
            end_sec: firstRow.end_sec,
            humpback: 1,
            ship: null,
            background: null,
          },
        ],
      },
    );
    expect(putRes.ok()).toBeTruthy();

    // Read content after label save — n_windows should be preserved
    const afterRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/content`,
    );
    const afterRows = await afterRes.json();
    const updatedRow = afterRows.find(
      (r: any) =>
        r.filename === firstRow.filename &&
        r.start_sec === firstRow.start_sec &&
        r.end_sec === firstRow.end_sec,
    );
    expect(updatedRow).toBeTruthy();
    expect(updatedRow.n_windows).toBe(originalNWindows);

    // Clean up: reset labels
    await request.put(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/labels`,
      {
        data: [
          {
            filename: firstRow.filename,
            start_sec: firstRow.start_sec,
            end_sec: firstRow.end_sec,
            humpback: null,
            ship: null,
            background: null,
          },
        ],
      },
    );
  });
});
