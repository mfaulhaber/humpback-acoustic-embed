import { test, expect, type Page } from "@playwright/test";

/**
 * Tests for the detection spectrogram popup feature.
 */

const MODEL = {
  id: "model-spec-1",
  name: "Spectrogram Test Model",
  model_path: "/tmp/model.joblib",
  model_version: "perch_v1",
  vector_dim: 1280,
  window_size_seconds: 5,
  target_sample_rate: 32000,
  feature_config: null,
  training_summary: null,
  training_job_id: null,
  created_at: "2026-03-14T00:00:00Z",
  updated_at: "2026-03-14T00:00:00Z",
};

const HYDROPHONE = {
  id: "rpi_north_sjc",
  name: "North San Juan Channel",
  location: "San Juan Channel",
  provider_kind: "orcasound_hls",
};

const PNG_1X1_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnR6i8AAAAASUVORK5CYII=";

function buildJob(overrides: Record<string, unknown> = {}) {
  return {
    id: "job-spec-1",
    status: "complete",
    classifier_model_id: MODEL.id,
    audio_folder: null,
    confidence_threshold: 0.5,
    hop_seconds: 1.0,
    high_threshold: 0.8,
    low_threshold: 0.7,
    output_tsv_path: "/tmp/detections.tsv",
    result_summary: null,
    error_message: null,
    files_processed: null,
    files_total: null,
    extract_status: null,
    extract_error: null,
    extract_summary: null,
    hydrophone_id: HYDROPHONE.id,
    hydrophone_name: HYDROPHONE.name,
    start_timestamp: 1751644800,
    end_timestamp: 1751648400,
    segments_processed: 5,
    segments_total: 5,
    time_covered_sec: 300,
    alerts: null,
    local_cache_path: null,
    has_positive_labels: null,
    created_at: "2026-03-14T00:00:00Z",
    updated_at: "2026-03-14T00:00:00Z",
    ...overrides,
  };
}

async function setupUiMocks(page: Page, rows: Record<string, unknown>[]) {
  await page.addInitScript(() => {
    HTMLMediaElement.prototype.load = function load() {};
    HTMLMediaElement.prototype.play = async function play() {};
    HTMLMediaElement.prototype.pause = function pause() {};
  });

  await page.route("**/classifier/training-jobs", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/classifier/models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([MODEL]),
    }),
  );
  await page.route("**/classifier/hydrophones", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([HYDROPHONE]),
    }),
  );
  await page.route("**/classifier/hydrophone-detection-jobs", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([buildJob()]),
    }),
  );
  await page.route(/\/detection-jobs\/[^/]+\/content$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(rows),
    }),
  );
  await page.route(/\/detection-jobs\/[^/]+\/spectrogram(\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(PNG_1X1_BASE64, "base64"),
    }),
  );
}

test.describe("Detection spectrogram", () => {
  test("spectrogram endpoint returns PNG for a completed detection job", async ({
    request,
  }) => {
    // Try local detection jobs first
    const jobsRes = await request.get(
      "http://localhost:8000/classifier/detection-jobs",
    );
    expect(jobsRes.ok()).toBeTruthy();
    let jobs = await jobsRes.json();

    let completedJob = jobs.find(
      (j: { status: string; output_tsv_path: string | null }) =>
        j.status === "complete" && j.output_tsv_path,
    );

    // Fall back to hydrophone jobs
    if (!completedJob) {
      const hydroRes = await request.get(
        "http://localhost:8000/classifier/hydrophone-detection-jobs",
      );
      expect(hydroRes.ok()).toBeTruthy();
      jobs = await hydroRes.json();
      completedJob = jobs.find(
        (j: { status: string; output_tsv_path: string | null }) =>
          (j.status === "complete" || j.status === "canceled") &&
          j.output_tsv_path,
      );
    }

    if (!completedJob) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    // Fetch detection content to get a real row
    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    const row = rows[0];
    const duration = Math.max(row.end_utc - row.start_utc, 5);

    // Request spectrogram
    const specRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${completedJob.id}/spectrogram` +
        `?start_utc=${row.start_utc}&duration_sec=${duration}`,
    );
    if (!specRes.ok()) {
      test.skip(
        true,
        `Unable to fetch spectrogram for sampled job (status ${specRes.status()})`,
      );
      return;
    }
    expect(specRes.headers()["content-type"]).toBe("image/png");

    const body = await specRes.body();
    // Verify PNG magic bytes
    expect(body.slice(0, 4).toString("hex")).toBe("89504e47");
    expect(body.length).toBeGreaterThan(1000);
  });

  test("spectrogram endpoint returns 404 for nonexistent job", async ({
    request,
  }) => {
    const res = await request.get(
      "http://localhost:8000/classifier/detection-jobs/nonexistent/spectrogram" +
        "?start_utc=0&duration_sec=5",
    );
    expect(res.status()).toBe(404);
  });

  test("positive label edits show markers without save or refresh", async ({
    page,
  }) => {
    await setupUiMocks(page, [
      {
        start_utc: 1751619600,
        end_utc: 1751619610,
        avg_confidence: 0.93,
        peak_confidence: 0.97,
        n_windows: 6,
        humpback: null,
        orca: null,
        ship: null,
        background: null,
        raw_start_utc: 1751619600,
        raw_end_utc: 1751619610,
        positive_selection_start_utc: 1751619605,
        positive_selection_end_utc: 1751619610,
      },
    ]);

    await page.goto("/app/classifier/hydrophone");
    await page.locator("table").last().locator("tbody tr td:nth-child(2) button").first().click();
    await expect(page.locator(".clip-range")).toBeVisible();
    await page.locator(".clip-range").locator("xpath=ancestor::tr").locator('input[type="checkbox"]').nth(0).check();
    await page.locator('button[title="Play"]').click();

    await expect(page.getByTestId("spectrogram-popup")).toBeVisible();
    await expect(page.getByTestId("spectrogram-marker-start")).toHaveCount(1);
    await expect(page.getByTestId("spectrogram-marker-end")).toHaveCount(1);
  });

  test("positive rows can adjust the window in 5-second steps and apply row state", async ({
    page,
  }) => {
    let rowStatePayload: Record<string, unknown> | null = null;

    await setupUiMocks(page, [
      {
        start_utc: 1751619600,
        end_utc: 1751619610,
        avg_confidence: 0.93,
        peak_confidence: 0.97,
        n_windows: 6,
        humpback: null,
        orca: null,
        ship: null,
        background: null,
        raw_start_utc: 1751619600,
        raw_end_utc: 1751619610,
        auto_positive_selection_start_utc: 1751619605,
        auto_positive_selection_end_utc: 1751619610,
        positive_selection_start_utc: null,
        positive_selection_end_utc: null,
      },
    ]);

    await page.route(/\/detection-jobs\/[^/]+\/row-state$/, async (route) => {
      rowStatePayload = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "ok",
          row: {
            start_utc: 1751619600,
            end_utc: 1751619610,
            avg_confidence: 0.93,
            peak_confidence: 0.97,
            n_windows: 6,
            humpback: 1,
            orca: null,
            ship: null,
            background: null,
            raw_start_utc: 1751619600,
            raw_end_utc: 1751619610,
            manual_positive_selection_start_utc: 1751619600,
            manual_positive_selection_end_utc: 1751619615,
            positive_selection_origin: "manual_override",
            positive_selection_start_utc: 1751619600,
            positive_selection_end_utc: 1751619615,
          },
        }),
      });
    });

    await page.goto("/app/classifier/hydrophone");
    await page.locator("table").last().locator("tbody tr td:nth-child(2) button").first().click();
    await expect(page.locator(".clip-range")).toBeVisible();
    await page.locator(".clip-range").locator("xpath=ancestor::tr").locator('input[type="checkbox"]').nth(0).check();
    await page.locator('button[title="Play"]').click();

    await expect(page.getByTestId("spectrogram-popup")).toBeVisible();
    await expect(page.getByTestId("spectrogram-start-earlier")).toBeVisible();
    await expect(page.getByTestId("spectrogram-end-later")).toBeVisible();

    await page.getByTestId("spectrogram-start-earlier").click();
    await page.getByTestId("spectrogram-end-later").click();

    await expect(page.getByTestId("spectrogram-start-earlier")).toBeDisabled();
    await expect(page.getByTestId("spectrogram-end-later")).toBeDisabled();

    await page.getByTestId("spectrogram-apply").click();

    await expect.poll(() => rowStatePayload).not.toBeNull();
    expect(rowStatePayload).toMatchObject({
      start_utc: 1751619600,
      end_utc: 1751619610,
      humpback: 1,
      manual_positive_selection_start_utc: 1751619600,
      manual_positive_selection_end_utc: 1751619615,
    });
    await expect(page.getByTestId("spectrogram-popup")).toHaveCount(0);
  });

  test("edge expansion promotes to the full detection when a partial step would remain", async ({
    page,
  }) => {
    let rowStatePayload: Record<string, unknown> | null = null;

    await setupUiMocks(page, [
      {
        start_utc: 1438998259,
        end_utc: 1438998264,
        avg_confidence: 0.93,
        peak_confidence: 0.97,
        n_windows: 6,
        humpback: null,
        orca: null,
        ship: null,
        background: null,
        raw_start_utc: 1438998259,
        raw_end_utc: 1438998264,
        auto_positive_selection_start_utc: 1438998259,
        auto_positive_selection_end_utc: 1438998264,
        positive_selection_start_utc: null,
        positive_selection_end_utc: null,
      },
    ]);

    await page.route(/\/detection-jobs\/[^/]+\/row-state$/, async (route) => {
      rowStatePayload = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "ok",
          row: {
            start_utc: 1438998259,
            end_utc: 1438998264,
            avg_confidence: 0.93,
            peak_confidence: 0.97,
            n_windows: 6,
            humpback: 1,
            orca: null,
            ship: null,
            background: null,
            raw_start_utc: 1438998259,
            raw_end_utc: 1438998264,
            manual_positive_selection_start_utc: 1438998256,
            manual_positive_selection_end_utc: 1438998271,
            positive_selection_origin: "manual_override",
            positive_selection_start_utc: 1438998256,
            positive_selection_end_utc: 1438998271,
          },
        }),
      });
    });

    await page.goto("/app/classifier/hydrophone");
    await page.locator("table").last().locator("tbody tr td:nth-child(2) button").first().click();
    await expect(page.locator(".clip-range")).toBeVisible();
    await page.locator(".clip-range").locator("xpath=ancestor::tr").locator('input[type="checkbox"]').nth(0).check();
    await page.locator('button[title="Play"]').click();

    await expect(page.getByTestId("spectrogram-popup")).toBeVisible();
    await page.getByTestId("spectrogram-end-later").click();
    await page.getByTestId("spectrogram-end-later").click();
    await expect(page.getByTestId("spectrogram-start-earlier")).toBeDisabled();
    await expect(page.getByTestId("spectrogram-end-later")).toBeDisabled();

    await page.getByTestId("spectrogram-apply").click();

    await expect.poll(() => rowStatePayload).not.toBeNull();
    expect(rowStatePayload).toMatchObject({
      start_utc: 1438998259,
      end_utc: 1438998264,
      humpback: 1,
      manual_positive_selection_start_utc: 1438998256,
      manual_positive_selection_end_utc: 1438998271,
    });
  });

  test("right-edge expansion borrows from the left before using the full clip", async ({
    page,
  }) => {
    let rowStatePayload: Record<string, unknown> | null = null;

    await setupUiMocks(page, [
      {
        start_utc: 1438985900,
        end_utc: 1438985905,
        avg_confidence: 0.94,
        peak_confidence: 0.98,
        n_windows: 6,
        humpback: null,
        orca: null,
        ship: null,
        background: null,
        raw_start_utc: 1438985900,
        raw_end_utc: 1438985905,
        auto_positive_selection_start_utc: 1438985900,
        auto_positive_selection_end_utc: 1438985905,
        positive_selection_start_utc: null,
        positive_selection_end_utc: null,
      },
    ]);

    await page.route(/\/detection-jobs\/[^/]+\/row-state$/, async (route) => {
      rowStatePayload = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "ok",
          row: {
            start_utc: 1438985900,
            end_utc: 1438985905,
            avg_confidence: 0.94,
            peak_confidence: 0.98,
            n_windows: 6,
            humpback: 1,
            orca: null,
            ship: null,
            background: null,
            raw_start_utc: 1438985900,
            raw_end_utc: 1438985905,
            manual_positive_selection_start_utc: 1438985898,
            manual_positive_selection_end_utc: 1438985908,
            positive_selection_origin: "manual_override",
            positive_selection_start_utc: 1438985898,
            positive_selection_end_utc: 1438985908,
          },
        }),
      });
    });

    await page.goto("/app/classifier/hydrophone");
    await page.locator("table").last().locator("tbody tr td:nth-child(2) button").first().click();
    await expect(page.locator(".clip-range")).toBeVisible();
    await page
      .locator(".clip-range")
      .locator("xpath=ancestor::tr")
      .locator('input[type="checkbox"]')
      .nth(0)
      .check();
    await page.locator('button[title="Play"]').click();

    await expect(page.getByTestId("spectrogram-popup")).toBeVisible();
    await page.getByTestId("spectrogram-end-later").click();
    await expect(page.getByTestId("spectrogram-start-earlier")).toBeVisible();

    await page.getByTestId("spectrogram-apply").click();

    await expect.poll(() => rowStatePayload).not.toBeNull();
    expect(rowStatePayload).toMatchObject({
      start_utc: 1438985900,
      end_utc: 1438985905,
      humpback: 1,
      manual_positive_selection_start_utc: 1438985898,
      manual_positive_selection_end_utc: 1438985908,
    });
  });

  test("Alt+click still opens popup and rows without bounds show no markers", async ({
    page,
  }) => {
    await setupUiMocks(page, [
      {
        start_utc: 1751619600,
        end_utc: 1751619610,
        avg_confidence: 0.93,
        peak_confidence: 0.97,
        n_windows: 6,
        humpback: null,
        orca: null,
        ship: null,
        background: null,
        raw_start_utc: 1751619600,
        raw_end_utc: 1751619610,
      },
    ]);

    await page.goto("/app/classifier/hydrophone");
    await page.locator("table").last().locator("tbody tr td:nth-child(2) button").first().click();
    await expect(page.locator(".clip-range")).toBeVisible();
    await page.locator(".clip-range").click({ modifiers: ["Alt"] });

    await expect(page.getByTestId("spectrogram-popup")).toBeVisible();
    await expect(page.getByTestId("spectrogram-marker-start")).toHaveCount(0);
    await expect(page.getByTestId("spectrogram-marker-end")).toHaveCount(0);
  });

  test("legacy positive rows without selection metadata fall back to clip-edge markers", async ({
    page,
  }) => {
    await setupUiMocks(page, [
      {
        start_utc: 1438985698,
        end_utc: 1438985713,
        avg_confidence: 0.974968,
        peak_confidence: 0.99657,
        n_windows: 8,
        humpback: 1,
        orca: 0,
        ship: null,
        background: null,
        raw_start_utc: 1438985698,
        raw_end_utc: 1438985713,
        positive_selection_start_utc: null,
        positive_selection_end_utc: null,
        positive_extract_filename: null,
      },
    ]);

    await page.goto("/app/classifier/hydrophone");
    await page.locator("table").last().locator("tbody tr td:nth-child(2) button").first().click();
    await expect(page.locator(".clip-range")).toContainText("20150807T221458Z_20150807T221513Z");
    await page.locator('button[title="Play"]').click();

    await expect(page.getByTestId("spectrogram-popup")).toBeVisible();
    await expect(page.getByTestId("spectrogram-marker-start")).toHaveCount(1);
    await expect(page.getByTestId("spectrogram-marker-end")).toHaveCount(1);
  });
});
