import { test, expect } from "@playwright/test";

/**
 * Tests for the classifier Training tab UI changes:
 * - Positive and Negative embedding set panels with select-all inside panel
 * - Always-visible (disabled) delete buttons on Training Jobs and Trained Models
 * - Training job API accepts negative_embedding_set_ids
 */

test.describe("Classifier Training tab", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/app/classifier/training");
  });

  test("renders positive and negative embedding set panels", async ({
    page,
  }) => {
    // Both panels should be inside the Train Binary Classifier card
    const card = page.locator("text=Train Binary Classifier").locator("..");

    // Look for the panel labels inside bordered panels
    const posLabel = page.locator("text=Positive Embedding Sets");
    const negLabel = page.locator("text=Negative Embedding Sets");

    await expect(posLabel).toBeVisible();
    await expect(negLabel).toBeVisible();
  });

  test("select-all checkbox is inside each panel", async ({ page }) => {
    // Each panel is a bordered div containing a checkbox + label on the first row
    // The positive panel has a checkbox next to "Positive Embedding Sets"
    const panels = page.locator(".border.rounded.p-2");

    // Should have at least 2 panels (positive and negative)
    const panelCount = await panels.count();
    expect(panelCount).toBeGreaterThanOrEqual(2);

    // Each panel's first child row should contain a checkbox
    for (let i = 0; i < 2; i++) {
      const panel = panels.nth(i);
      const firstRow = panel.locator(".border-b").first();
      const checkbox = firstRow.locator('button[role="checkbox"]');
      await expect(checkbox).toBeVisible();
    }
  });

  test("Train Classifier button disabled when no selections", async ({
    page,
  }) => {
    const trainBtn = page.locator("button", { hasText: "Train Classifier" });
    await expect(trainBtn).toBeVisible();
    await expect(trainBtn).toBeDisabled();
  });

  test("Model Name input is present", async ({ page }) => {
    const nameInput = page.locator('input[placeholder="e.g. humpback-detector-v1"]');
    await expect(nameInput).toBeVisible();
  });

  test("no folder browser or folder path input present", async ({ page }) => {
    // The old negative audio folder input should be gone
    const folderInput = page.locator(
      'input[placeholder="/path/to/negative/audio"]',
    );
    await expect(folderInput).toHaveCount(0);

    // No FolderOpen icon button for negative folder
    const folderBrowseBtn = page.locator('button[title="Browse folders"]');
    await expect(folderBrowseBtn).toHaveCount(0);
  });
});

test.describe("Advanced Options", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/app/classifier/training");
  });

  test("advanced options collapsible is present and expands", async ({
    page,
  }) => {
    const trigger = page.locator("button", { hasText: "Advanced Options" });
    await expect(trigger).toBeVisible();

    // Click to expand
    await trigger.click();

    // Should show classifier type select
    await expect(
      page.locator("text=Classifier Type"),
    ).toBeVisible();

    // Should show L2 Normalize checkbox
    await expect(
      page.locator("text=L2 Normalize Embeddings"),
    ).toBeVisible();

    // Should show Class Weight select
    await expect(page.locator("text=Class Weight")).toBeVisible();
  });

  test("regularization C shown only for logistic regression", async ({
    page,
  }) => {
    const trigger = page.locator("button", { hasText: "Advanced Options" });
    await trigger.click();

    // Default is logistic regression — regularization should be visible
    await expect(
      page.locator("text=Regularization (C)"),
    ).toBeVisible();
  });
});

test.describe("Trained Models table columns", () => {
  test("table has Precision and F1 columns when models exist", async ({
    page,
  }) => {
    await page.goto("/app/classifier/training");

    const modelsHeading = page.locator("text=Trained Models");
    const hasModels = await modelsHeading
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasModels) {
      test.skip(true, "No trained models to verify table columns");
      return;
    }

    // Check for new column headers
    const table = modelsHeading.locator("..").locator("..").locator("table");
    await expect(table.locator("th", { hasText: "Precision" })).toBeVisible();
    await expect(table.locator("th", { hasText: "F1" })).toBeVisible();
  });
});

test.describe("Training API overlap validation", () => {
  test("rejects same embedding set in both positive and negative", async ({
    request,
  }) => {
    const resp = await request.post(
      "http://localhost:8000/classifier/training-jobs",
      {
        data: {
          name: "overlap-test",
          positive_embedding_set_ids: ["shared-id", "pos-only"],
          negative_embedding_set_ids: ["shared-id", "neg-only"],
        },
      },
    );
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.detail).toContain("both positive and negative");
  });
});

test.describe("Training Jobs delete button visibility", () => {
  test("delete button always visible but disabled when nothing selected", async ({
    page,
  }) => {
    await page.goto("/app/classifier/training");

    // Check for Training Jobs section
    const jobsHeading = page.locator("text=Training Jobs");
    const hasJobs = await jobsHeading
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasJobs) {
      test.skip(true, "No training jobs to test delete button visibility");
      return;
    }

    // Delete button should be visible even with nothing selected
    const deleteBtn = jobsHeading
      .locator("..")
      .locator("..")
      .locator("button", { hasText: /^Delete/ });
    await expect(deleteBtn).toBeVisible();
    await expect(deleteBtn).toBeDisabled();
  });
});

test.describe("Trained Models delete button visibility", () => {
  test("delete button always visible but disabled when nothing selected", async ({
    page,
  }) => {
    await page.goto("/app/classifier/training");

    // Check for Trained Models section
    const modelsHeading = page.locator("text=Trained Models");
    const hasModels = await modelsHeading
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasModels) {
      test.skip(true, "No trained models to test delete button visibility");
      return;
    }

    // Delete button should be visible even with nothing selected
    const deleteBtn = modelsHeading
      .locator("..")
      .locator("..")
      .locator("button", { hasText: /^Delete/ });
    await expect(deleteBtn).toBeVisible();
    await expect(deleteBtn).toBeDisabled();
  });
});

test.describe("Training job API", () => {
  test("create training job validates negative_embedding_set_ids", async ({
    request,
  }) => {
    // Should reject when negative embedding sets don't exist
    const resp = await request.post(
      "http://localhost:8000/classifier/training-jobs",
      {
        data: {
          name: "playwright-test",
          positive_embedding_set_ids: ["nonexistent-pos"],
          negative_embedding_set_ids: ["nonexistent-neg"],
        },
      },
    );
    expect(resp.status()).toBe(400);
  });

  test("create training job rejects empty negative_embedding_set_ids", async ({
    request,
  }) => {
    const resp = await request.post(
      "http://localhost:8000/classifier/training-jobs",
      {
        data: {
          name: "playwright-test",
          positive_embedding_set_ids: ["some-id"],
          negative_embedding_set_ids: [],
        },
      },
    );
    expect(resp.status()).toBe(400);
  });

  test("training job response includes negative_embedding_set_ids field", async ({
    request,
  }) => {
    // List existing training jobs and check response shape
    const resp = await request.get(
      "http://localhost:8000/classifier/training-jobs",
    );
    expect(resp.ok()).toBeTruthy();
    const jobs = await resp.json();

    if (jobs.length === 0) {
      test.skip(true, "No training jobs to verify response shape");
      return;
    }

    const job = jobs[0];
    expect(job).toHaveProperty("positive_embedding_set_ids");
    expect(job).toHaveProperty("negative_embedding_set_ids");
    expect(Array.isArray(job.negative_embedding_set_ids)).toBe(true);
    // Should NOT have the old field
    expect(job).not.toHaveProperty("negative_audio_folder");
  });
});
