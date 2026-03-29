import { test, expect } from "@playwright/test";

test.describe("Vocalization Labeling Tab", () => {
  test("renders detection job picker", async ({ page }) => {
    await page.goto("/app/vocalization/labeling");
    await expect(page.getByText("Vocalization Labeling")).toBeVisible();
    await expect(page.getByText("Source")).toBeVisible();
    await expect(
      page.getByText("Select a detection job..."),
    ).toBeVisible();
  });

  test("shows embedding status when job selected", async ({ page }) => {
    await page.goto("/app/vocalization/labeling");

    // Check if there are any completed detection jobs
    const trigger = page.locator(
      '[role="combobox"]:has-text("Select a detection job...")',
    );
    await trigger.click();

    const options = page.locator('[role="option"]');
    const count = await options.count();

    if (count === 0) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    // Select the first job
    await options.first().click();

    // Embedding status panel should appear
    await expect(
      page.getByText(/Embeddings/).first(),
    ).toBeVisible({ timeout: 5000 });
  });
});
