import { test, expect } from "@playwright/test";

test.describe("Label Processing page", () => {
  test("navigates to label processing page and renders form", async ({ page }) => {
    await page.goto("/app/label-processing");
    await expect(page.locator("text=Create Label Processing Job")).toBeVisible();
    await expect(page.locator("text=Classifier Model")).toBeVisible();
    await expect(page.locator("text=Annotation Folder")).toBeVisible();
    await expect(page.locator("text=Audio Folder")).toBeVisible();
    await expect(page.locator("text=Output Root")).toBeVisible();
  });

  test("start processing button is disabled without inputs", async ({ page }) => {
    await page.goto("/app/label-processing");
    const startBtn = page.getByRole("button", { name: /Start Processing/ });
    await expect(startBtn).toBeDisabled();
  });

  test("advanced parameters section is collapsible", async ({ page }) => {
    await page.goto("/app/label-processing");
    const trigger = page.locator("text=Advanced Parameters");
    await expect(trigger).toBeVisible();
    // Should be collapsed by default
    await expect(page.locator("text=Peak Threshold")).not.toBeVisible();
    // Open it
    await trigger.click();
    await expect(page.locator("text=Peak Threshold")).toBeVisible();
    await expect(page.locator("text=Smoothing Window")).toBeVisible();
    await expect(page.locator("text=Background Threshold")).toBeVisible();
  });

  test("side nav has label processing link", async ({ page }) => {
    await page.goto("/app/audio");
    const navLink = page.locator("nav a", { hasText: "Label Processing" });
    await expect(navLink).toBeVisible();
    await navLink.click();
    await expect(page).toHaveURL(/\/app\/label-processing/);
  });

  test("preview button is disabled without folder inputs", async ({ page }) => {
    await page.goto("/app/label-processing");
    const previewBtn = page.getByRole("button", { name: /Preview/ });
    await expect(previewBtn).toBeDisabled();
  });

  test("breadcrumb shows Label Processing", async ({ page }) => {
    await page.goto("/app/label-processing");
    await expect(page.locator("text=Label Processing").first()).toBeVisible();
  });
});
