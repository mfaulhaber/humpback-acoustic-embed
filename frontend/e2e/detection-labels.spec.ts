import { test, expect } from "@playwright/test";

/**
 * Tests for detection label annotation (humpback, ship, background checkboxes).
 * Requires: backend running on :8000 with at least one completed detection job.
 */

// Helper to find a completed detection job
async function findCompletedJob(request: any) {
  const jobsRes = await request.get(
    "http://localhost:8000/classifier/detection-jobs",
  );
  expect(jobsRes.ok()).toBeTruthy();
  const jobs = await jobsRes.json();
  return jobs.find(
    (j: { status: string; output_tsv_path: string | null }) =>
      j.status === "complete" && j.output_tsv_path,
  );
}

test.describe("Detection labels API", () => {
  test("GET content returns label columns", async ({ request }) => {
    const job = await findCompletedJob(request);
    if (!job) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    const row = rows[0];
    // Label fields should be present (initially null if TSV has no label columns)
    expect("humpback" in row).toBe(true);
    expect("ship" in row).toBe(true);
    expect("background" in row).toBe(true);
    // avg_confidence should still be present
    expect("avg_confidence" in row).toBe(true);
  });

  test("PUT labels round-trip persists and resets", async ({ request }) => {
    const job = await findCompletedJob(request);
    if (!job) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    // Get initial content
    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    const firstRow = rows[0];

    // Save labels on the first row
    const putRes = await request.put(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/labels`,
      {
        data: [
          {
            filename: firstRow.filename,
            start_sec: firstRow.start_sec,
            end_sec: firstRow.end_sec,
            humpback: 1,
            ship: 0,
            background: null,
          },
        ],
      },
    );
    expect(putRes.ok()).toBeTruthy();
    const putBody = await putRes.json();
    expect(putBody.status).toBe("ok");
    expect(putBody.updated).toBe(1);

    // Verify labels persisted
    const verifyRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    expect(verifyRes.ok()).toBeTruthy();
    const verifyRows = await verifyRes.json();
    const updatedRow = verifyRows.find(
      (r: any) =>
        r.filename === firstRow.filename &&
        r.start_sec === firstRow.start_sec &&
        r.end_sec === firstRow.end_sec,
    );
    expect(updatedRow).toBeTruthy();
    expect(updatedRow.humpback).toBe(1);
    expect(updatedRow.ship).toBe(0);
    expect(updatedRow.background).toBeNull();

    // Reset labels back to null (cleanup)
    const resetRes = await request.put(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/labels`,
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
    expect(resetRes.ok()).toBeTruthy();

    // Verify reset
    const resetVerifyRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    const resetRows = await resetVerifyRes.json();
    const resetRow = resetRows.find(
      (r: any) =>
        r.filename === firstRow.filename &&
        r.start_sec === firstRow.start_sec &&
        r.end_sec === firstRow.end_sec,
    );
    expect(resetRow.humpback).toBeNull();
    expect(resetRow.ship).toBeNull();
    expect(resetRow.background).toBeNull();
  });
});

test.describe("Detection content table UI", () => {
  test("table columns show Confidence and label headers", async ({ page }) => {
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

    // Find a completed job row and expand it
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

    const expandBtn = completedRow.locator("td:nth-child(2) button");
    await expect(expandBtn).toBeVisible();
    await expandBtn.click();

    // Wait for inner table
    const innerTable = page.locator("tr td[colspan] table");
    await innerTable.waitFor({ timeout: 10_000 });

    // Check column headers
    const headers = innerTable.locator("thead th");
    const headerTexts = await headers.allTextContents();
    const trimmed = headerTexts.map((h) => h.trim());

    // Should have "Confidence" (not "Avg Confidence" or "Peak Confidence")
    expect(trimmed).toContain("Confidence");
    expect(trimmed).not.toContain("Avg Confidence");
    expect(trimmed).not.toContain("Peak Confidence");

    // Should have label columns
    expect(trimmed).toContain("Humpback");
    expect(trimmed).toContain("Ship");
    expect(trimmed).toContain("Background");
  });

  test("label checkboxes and Save button interaction", async ({ page }) => {
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

    // Verify Save button exists and is disabled initially
    const saveBtn = page.locator("button", { hasText: "Save Labels" });
    await expect(saveBtn).toBeVisible();
    await expect(saveBtn).toBeDisabled();

    // Verify Delete button exists and is disabled initially
    const deleteBtn = page.locator("button", { hasText: /^Delete/ });
    await expect(deleteBtn).toBeVisible();
    await expect(deleteBtn).toBeDisabled();

    // Expand a completed job
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

    const expandBtn = completedRow.locator("td:nth-child(2) button");
    await expandBtn.click();

    // Wait for inner table
    const innerTable = page.locator("tr td[colspan] table");
    await innerTable.waitFor({ timeout: 10_000 });

    // Click a Humpback checkbox in the first detection row
    // The checkboxes are in columns 6, 7, 8 (1-indexed) of the inner table
    const firstDataRow = innerTable.locator("tbody tr").first();
    const humpbackCheckbox = firstDataRow.locator(
      'td:nth-child(6) input[type="checkbox"]',
    );
    await expect(humpbackCheckbox).toBeVisible();
    await humpbackCheckbox.click();

    // Save button should now be enabled
    await expect(saveBtn).toBeEnabled();

    // Click Save
    await saveBtn.click();

    // Wait for Save button to return to disabled (save completed)
    await expect(saveBtn).toBeDisabled({ timeout: 10_000 });
  });
});
