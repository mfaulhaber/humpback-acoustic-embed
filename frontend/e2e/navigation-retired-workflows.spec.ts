import { expect, test, type Page } from "@playwright/test";

async function setupDetectionPageMocks(page: Page) {
  await page.route("**/classifier/hydrophones", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    }),
  );
  await page.route("**/classifier/models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    }),
  );
  await page.route("**/admin/models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    }),
  );
  await page.route("**/call-parsing/region-jobs", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    }),
  );
}

test.describe("Retired top-level workflow navigation", () => {
  test.beforeEach(async ({ page }) => {
    await setupDetectionPageMocks(page);
  });

  test("root and unknown routes redirect to Call Parsing Detection", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/app\/call-parsing\/detection$/);
    await expect(page.locator("text=Region Detection")).toBeVisible();

    await page.goto("/app/audio");
    await expect(page).toHaveURL(/\/app\/call-parsing\/detection$/);

    await page.goto("/app/no-such-route");
    await expect(page).toHaveURL(/\/app\/call-parsing\/detection$/);
  });

  test("retired top-level links are absent and retained clustering remains", async ({ page }) => {
    await page.goto("/app/call-parsing/detection");

    const nav = page.locator("nav").first();
    await expect(nav.getByText("Audio", { exact: true })).toHaveCount(0);
    await expect(nav.getByText("Processing", { exact: true })).toHaveCount(0);
    await expect(nav.getByText("Search", { exact: true })).toHaveCount(0);
    await expect(nav.getByText("Label Processing", { exact: true })).toHaveCount(0);
    await expect(nav.getByText("Clustering", { exact: true })).toHaveCount(0);

    await expect(nav.getByText("Vocalization", { exact: true })).toBeVisible();
    await expect(nav.getByText("Call Parsing", { exact: true })).toBeVisible();

    await nav.getByText("Vocalization", { exact: true }).click();
    await expect(nav.locator("a", { hasText: "Clustering" })).toBeVisible();
  });

  test("brand link opens retained default route", async ({ page }) => {
    await page.goto("/app/admin");
    await page.getByRole("link", { name: /Humpback Acoustic/ }).click();
    await expect(page).toHaveURL(/\/app\/call-parsing\/detection$/);
  });
});
