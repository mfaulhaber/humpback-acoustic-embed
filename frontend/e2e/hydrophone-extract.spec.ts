import { test, expect } from "@playwright/test";

function parseCompactUtcMs(value: string): number | null {
  const match = value.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  if (!match) return null;
  return Date.UTC(
    Number(match[1]),
    Number(match[2]) - 1,
    Number(match[3]),
    Number(match[4]),
    Number(match[5]),
    Number(match[6]),
  );
}

function parseCompactRangeMs(value: string): { startMs: number; endMs: number } | null {
  const parts = value.trim().split("_");
  if (parts.length !== 2) return null;
  const startMs = parseCompactUtcMs(parts[0]);
  const endMs = parseCompactUtcMs(parts[1]);
  if (startMs === null || endMs === null || endMs <= startMs) return null;
  return { startMs, endMs };
}

test.describe("Hydrophone extract activation", () => {
  test("Extract enables from saved labels on expanded completed job", async ({ page }) => {
    await page.goto("/app/classifier/hydrophone");

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
    const clipRangeText = (await detectionRangeCell.locator(".clip-range").innerText()).trim();
    const clipRange = parseCompactRangeMs(clipRangeText);
    expect(clipRange).toBeTruthy();
    if (!clipRange) {
      test.skip(true, "Could not parse displayed clip range");
      return;
    }
    const durationText = (await firstDataRow.locator("td:nth-child(3)").innerText()).trim();
    const displayedDurationSec = Number.parseFloat(durationText);
    expect(Number.isFinite(displayedDurationSec)).toBe(true);
    expect(displayedDurationSec).toBeCloseTo((clipRange.endMs - clipRange.startMs) / 1000, 1);

    const rangeTitle = await detectionRangeCell.getAttribute("title");
    expect(rangeTitle).toContain("Z_");

    await firstDataRow.locator("td:nth-child(1) button").click();
    await page.waitForTimeout(300);
    const audioInfo = await page.evaluate(() => {
      const audio = document.querySelector("audio");
      if (!audio?.src) return null;
      const url = new URL(audio.src);
      return {
        filename: url.searchParams.get("filename"),
        startSec: Number.parseFloat(url.searchParams.get("start_sec") || "NaN"),
        durationSec: Number.parseFloat(url.searchParams.get("duration_sec") || "NaN"),
      };
    });
    expect(audioInfo).toBeTruthy();
    const filename = audioInfo?.filename;
    expect(filename).toBeTruthy();
    const fileStartMs = parseCompactUtcMs((filename ?? "").replace(".wav", ""));
    expect(fileStartMs).not.toBeNull();
    if (fileStartMs === null) {
      test.skip(true, "Playback filename is not compact UTC format");
      return;
    }
    const expectedStartSec = (clipRange.startMs - fileStartMs) / 1000;
    const expectedDurationSec = (clipRange.endMs - clipRange.startMs) / 1000;
    expect(audioInfo?.startSec).toBeCloseTo(expectedStartSec, 3);
    expect(audioInfo?.durationSec).toBeCloseTo(expectedDurationSec, 3);

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
