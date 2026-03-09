import { test, expect } from "@playwright/test";

test.describe("Retrain Workflow API", () => {
  test("retrain-info returns 404 for nonexistent model", async ({ request }) => {
    const resp = await request.get(
      "http://localhost:8000/classifier/models/nonexistent/retrain-info"
    );
    expect(resp.status()).toBe(404);
  });

  test("create retrain returns 400 for nonexistent model", async ({
    request,
  }) => {
    const resp = await request.post("http://localhost:8000/classifier/retrain", {
      data: {
        source_model_id: "nonexistent",
        new_model_name: "retrained",
      },
    });
    expect(resp.status()).toBe(400);
  });

  test("list retrain workflows returns array", async ({ request }) => {
    const resp = await request.get(
      "http://localhost:8000/classifier/retrain-workflows"
    );
    expect(resp.status()).toBe(200);
    const data = await resp.json();
    expect(Array.isArray(data)).toBe(true);
  });

  test("get retrain workflow returns 404 for nonexistent", async ({
    request,
  }) => {
    const resp = await request.get(
      "http://localhost:8000/classifier/retrain-workflows/nonexistent"
    );
    expect(resp.status()).toBe(404);
  });
});

test.describe("Retrain UI", () => {
  test("trained model row has retrain button when expanded", async ({
    page,
  }) => {
    await page.goto("/");

    // Navigate to Classifier tab then Train sub-tab
    const classifierTab = page.getByRole("button", { name: /classifier/i });
    if (await classifierTab.isVisible()) {
      await classifierTab.click();
    }

    // Check if there are any trained models
    const modelsHeading = page.getByText("Trained Models");
    if (!(await modelsHeading.isVisible({ timeout: 2000 }).catch(() => false))) {
      test.skip(true, "No Trained Models section visible");
      return;
    }

    // Try to find and expand a model row
    const modelRows = page.locator("table").last().locator("tbody tr");
    const rowCount = await modelRows.count();
    if (rowCount === 0) {
      test.skip(true, "No trained models available");
      return;
    }

    // Click to expand first model row
    await modelRows.first().click();

    // Should see the Retrain button in the expanded section
    const retrainButton = page.getByRole("button", { name: /retrain/i });
    await expect(retrainButton).toBeVisible({ timeout: 3000 });
  });
});
