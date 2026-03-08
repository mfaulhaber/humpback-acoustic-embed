import { test, expect } from "@playwright/test";

/**
 * Tests for multi-model support UI:
 * - ModelFilter dropdown with "Model:" label on Processing, Clustering, and Classifier pages
 * - Model version badges on embedding set and job rows
 */

test.describe("Model filter on Processing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/app/processing");
  });

  test("Model filter dropdown is visible with label", async ({ page }) => {
    const label = page.locator("label", { hasText: "Model:" });
    await expect(label).toBeVisible();

    // The select trigger should be next to the label
    const filter = label.locator("..").locator('button[role="combobox"]');
    await expect(filter).toBeVisible();
  });

  test("model badges appear on processing job rows", async ({ page }) => {
    // Wait for jobs to load — look for the Processing Jobs card
    const jobsCard = page.locator("text=Processing Jobs");
    const hasJobs = await jobsCard
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasJobs) {
      test.skip(true, "No processing jobs to verify badges");
      return;
    }

    // Model version badges use Badge variant="outline"
    const badges = page.locator(
      '.rounded-full.border:has-text("perch"), .rounded-full.border:has-text("surfperch")',
    );
    const count = await badges.count();
    expect(count).toBeGreaterThan(0);
  });
});

test.describe("Model filter on Clustering page", () => {
  test("Model filter dropdown is visible with label", async ({ page }) => {
    await page.goto("/app/clustering");

    // The clustering page has the selector card
    const heading = page.getByRole("heading", { name: "Queue Clustering Job" });
    await expect(heading).toBeVisible();

    const label = page.locator("label", { hasText: "Model:" });
    await expect(label).toBeVisible();
  });
});

test.describe("Model filter on Classifier Training page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/app/classifier");
    await page.getByRole("button", { name: "Train", exact: true }).click();
  });

  test("Model filter dropdown is visible with label", async ({ page }) => {
    const label = page.locator("label", { hasText: "Model:" });
    await expect(label).toBeVisible();
  });

  test("model badges appear on embedding set panel child rows", async ({
    page,
  }) => {
    // Wait for embedding sets to load — the positive panel should have folder rows
    const posLabel = page.locator("text=Positive Embedding Sets");
    await expect(posLabel).toBeVisible();

    // The panels start collapsed. Find and click the first chevron to expand.
    // The panel is inside the Train Binary Classifier card
    const card = page.getByRole("heading", { name: "Train Binary Classifier" }).locator("..").locator("..");
    const chevrons = card.locator('button:has(svg.lucide-chevron-right)');
    const chevronCount = await chevrons.count();

    if (chevronCount === 0) {
      test.skip(true, "No embedding set folders to expand");
      return;
    }

    await chevrons.first().click();

    // After expanding, child label rows should contain model version text as badges
    // The embedding model name should appear somewhere in the expanded content
    const modelBadge = card.locator("text=surfperch-tensorflow2").first();
    const hasBadge = await modelBadge
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasBadge) {
      // Try the other known model name
      const altBadge = card.locator("text=perch_v2").first();
      await expect(altBadge).toBeVisible({ timeout: 2_000 });
    }
  });
});
