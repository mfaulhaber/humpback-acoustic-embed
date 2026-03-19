import { test, expect } from "@playwright/test";

test.describe("Extraction Settings API", () => {
  test("GET /classifier/extraction-settings returns default paths", async ({
    request,
  }) => {
    const resp = await request.get(
      "http://localhost:8000/classifier/extraction-settings",
    );
    expect(resp.ok()).toBe(true);
    const data = await resp.json();
    expect(data).toHaveProperty("positive_output_path");
    expect(data).toHaveProperty("negative_output_path");
    expect(typeof data.positive_output_path).toBe("string");
    expect(typeof data.negative_output_path).toBe("string");
  });
});

test.describe("Extraction UI", () => {
  test("Extract button is disabled with no selection", async ({ page }) => {
    await page.goto("/app/classifier/hydrophone");

    // Check if the Extract button exists and is disabled
    const extractButton = page.getByRole("button", {
      name: /extract labeled samples/i,
    });
    if (await extractButton.isVisible()) {
      await expect(extractButton).toBeDisabled();
    }
  });
});
