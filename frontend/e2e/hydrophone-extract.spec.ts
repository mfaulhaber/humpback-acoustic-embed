import { test, expect } from "@playwright/test";

test.describe("Hydrophone extract activation", () => {
  test("Extract enables from saved labels on expanded completed job", async ({ page }) => {
    await page.goto("/app/classifier");
    await page.locator("button", { hasText: "Hydrophone" }).click();

    const table = page.locator("table").first();
    const hasTable = await table
      .waitFor({ timeout: 5_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasTable) {
      test.skip(true, "No hydrophone jobs table visible");
      return;
    }

    const completedRows = page.locator("table tbody tr").filter({ hasText: "complete" });
    const rowCount = await completedRows.count();
    if (rowCount === 0) {
      test.skip(true, "No completed hydrophone job rows");
      return;
    }

    let expanded = false;
    for (let i = 0; i < rowCount; i += 1) {
      const row = completedRows.nth(i);
      const expandBtn = row.locator("td:nth-child(2) button");
      if ((await expandBtn.count()) > 0) {
        await expandBtn.click();
        expanded = true;
        break;
      }
    }
    if (!expanded) {
      test.skip(true, "No expandable completed hydrophone rows");
      return;
    }

    const innerTable = page.locator("tr td[colspan] table");
    const hasInner = await innerTable
      .waitFor({ timeout: 10_000 })
      .then(() => true)
      .catch(() => false);
    if (!hasInner) {
      test.skip(true, "Expanded row has no detection content table");
      return;
    }

    await expect(innerTable.locator("thead")).toContainText("Duration (s)");

    const firstDataRow = innerTable.locator("tbody tr").first();
    const hasDataRow = (await firstDataRow.count()) > 0;
    if (!hasDataRow) {
      test.skip(true, "Expanded hydrophone job has no detections");
      return;
    }

    const detectionRangeCell = firstDataRow.locator("td:nth-child(2)");
    const rangeTitle = await detectionRangeCell.getAttribute("title");
    expect(rangeTitle).toContain(".wav");

    const humpbackCheckbox = firstDataRow.locator(
      'td:nth-child(5) input[type="checkbox"]',
    );
    const hasCheckbox = (await humpbackCheckbox.count()) > 0;
    if (!hasCheckbox) {
      test.skip(true, "Could not find label checkbox in expanded detection row");
      return;
    }

    const saveButton = page.locator("button", { hasText: "Save Labels" });
    const extractButton = page.locator("button", { hasText: "Extract Labeled Samples" });

    const initiallyChecked = await humpbackCheckbox.isChecked();
    if (initiallyChecked) {
      // Force a dirty change while ending at checked=true.
      await humpbackCheckbox.click();
      await humpbackCheckbox.click();
    } else {
      await humpbackCheckbox.click();
    }
    await expect(saveButton).toBeEnabled();
    await saveButton.click();
    await expect(saveButton).toBeDisabled({ timeout: 10_000 });

    await expect(extractButton).toBeEnabled({ timeout: 10_000 });
    await extractButton.click();

    await expect(
      page.getByText(/from 1 selected detection job/i),
    ).toBeVisible();
  });
});
