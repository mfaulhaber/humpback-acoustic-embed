import { expect, test, type Page } from "@playwright/test";

const REGION_JOB = {
  id: "rj-complete-1",
  status: "complete",
  audio_file_id: null,
  hydrophone_id: "rpi_orcasound_lab",
  start_timestamp: 1751644800,
  end_timestamp: 1751648400,
  model_config_id: null,
  classifier_model_id: null,
  config_json: null,
  parent_run_id: null,
  error_message: null,
  chunks_total: null,
  chunks_completed: null,
  windows_detected: null,
  trace_row_count: 3600,
  region_count: 4,
  created_at: "2026-04-26T00:00:00Z",
  updated_at: "2026-04-26T00:30:00Z",
  started_at: "2026-04-26T00:00:01Z",
  completed_at: "2026-04-26T00:30:00Z",
};

const SEG_JOB = {
  id: "seg-complete-1",
  status: "complete",
  region_detection_job_id: REGION_JOB.id,
  segmentation_model_id: null,
  config_json: null,
  parent_run_id: null,
  event_count: 12,
  compute_device: null,
  gpu_fallback_reason: null,
  error_message: null,
  created_at: "2026-04-26T01:00:00Z",
  updated_at: "2026-04-26T01:10:00Z",
  started_at: "2026-04-26T01:00:01Z",
  completed_at: "2026-04-26T01:10:00Z",
};

const QUEUED_JOB = {
  id: "cej-queued-1",
  status: "queued",
  event_segmentation_job_id: SEG_JOB.id,
  model_version: "surfperch-tensorflow2",
  window_size_seconds: 5.0,
  hop_seconds: 1.0,
  pad_seconds: 2.0,
  target_sample_rate: 32000,
  feature_config_json: null,
  encoding_signature: "abc123",
  vector_dim: null,
  total_events: null,
  merged_spans: null,
  total_windows: null,
  parquet_path: null,
  error_message: null,
  created_at: "2026-04-27T00:00:00Z",
  updated_at: "2026-04-27T00:00:00Z",
};

const COMPLETE_JOB = {
  ...QUEUED_JOB,
  id: "cej-complete-1",
  status: "complete",
  vector_dim: 1280,
  total_events: 4,
  merged_spans: 3,
  total_windows: 240,
  parquet_path:
    "/tmp/data/continuous_embeddings/cej-complete-1/embeddings.parquet",
  encoding_signature: "complete-sig",
};

const FAILED_JOB = {
  ...QUEUED_JOB,
  id: "cej-failed-1",
  status: "failed",
  error_message: "embedder crashed mid-run",
  encoding_signature: "failed-sig",
};

const COMPLETE_DETAIL = {
  job: COMPLETE_JOB,
  manifest: {
    job_id: COMPLETE_JOB.id,
    model_version: COMPLETE_JOB.model_version,
    vector_dim: 1280,
    window_size_seconds: 5.0,
    hop_seconds: 1.0,
    pad_seconds: 10.0,
    target_sample_rate: 32000,
    total_events: 4,
    merged_spans: 3,
    total_windows: 240,
    spans: [
      {
        merged_span_id: 0,
        start_timestamp: 90.0,
        end_timestamp: 130.0,
        window_count: 36,
        event_id: "evt-0",
        region_id: "r1",
      },
      {
        merged_span_id: 1,
        start_timestamp: 200.0,
        end_timestamp: 240.0,
        window_count: 36,
        event_id: "evt-1",
        region_id: "r1",
      },
      {
        merged_span_id: 2,
        start_timestamp: 320.0,
        end_timestamp: 480.0,
        window_count: 168,
        event_id: "evt-2",
        region_id: "r2",
      },
    ],
  },
};

const FAILED_DETAIL = { job: FAILED_JOB, manifest: null };

interface MockState {
  jobs: typeof QUEUED_JOB[];
}

async function setupMocks(page: Page, state: MockState) {
  await page.route("**/call-parsing/region-jobs", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([REGION_JOB]),
    }),
  );

  await page.route("**/call-parsing/segmentation-jobs**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([SEG_JOB]),
    }),
  );

  await page.route("**/hydrophones", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([{ id: "rpi_orcasound_lab", name: "Orcasound Lab" }]),
    }),
  );

  await page.route(
    "**/sequence-models/continuous-embeddings**",
    (route) => {
      const url = route.request().url();
      const method = route.request().method();
      const idMatch = url.match(/\/continuous-embeddings\/([^/?#]+)/);

      if (idMatch) {
        const id = idMatch[1];
        if (url.includes("/cancel")) {
          const job = state.jobs.find((j) => j.id === id);
          if (!job) {
            return route.fulfill({ status: 404 });
          }
          job.status = "canceled";
          return route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(job),
          });
        }
        if (id === COMPLETE_JOB.id) {
          return route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(COMPLETE_DETAIL),
          });
        }
        if (id === FAILED_JOB.id) {
          return route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(FAILED_DETAIL),
          });
        }
        return route.fulfill({ status: 404 });
      }

      if (method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(state.jobs),
        });
      }

      if (method === "POST") {
        state.jobs = [QUEUED_JOB, ...state.jobs];
        return route.fulfill({
          status: 201,
          contentType: "application/json",
          body: JSON.stringify(QUEUED_JOB),
        });
      }
      return route.fulfill({ status: 405 });
    },
  );
}

test.describe("Sequence Models — Continuous Embedding", () => {
  test("nav reaches the Continuous Embedding page", async ({ page }) => {
    const state: MockState = { jobs: [QUEUED_JOB, COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/continuous-embedding");
    await expect(page.getByTestId("cej-jobs-page")).toBeVisible();
    await expect(page.getByText("Active Jobs")).toBeVisible();
    await expect(page.getByText("Previous Jobs")).toBeVisible();
  });

  test("create form posts and shows new job in Active", async ({ page }) => {
    const state: MockState = { jobs: [] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/continuous-embedding");

    await page
      .getByTestId("cej-seg-job-select")
      .selectOption(SEG_JOB.id);
    await page.getByTestId("cej-create-submit").click();

    const queuedRow = page.locator("tr", {
      hasText: SEG_JOB.id.slice(0, 8),
    }).first();
    await expect(queuedRow).toBeVisible();
    await expect(queuedRow.getByText("queued")).toBeVisible();
  });

  test("complete detail page shows manifest stats", async ({ page }) => {
    const state: MockState = { jobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(
      `/app/sequence-models/continuous-embedding/${COMPLETE_JOB.id}`,
    );

    await expect(page.getByTestId("cej-detail-page")).toBeVisible();
    await expect(page.getByTestId("cej-detail-status")).toHaveText("complete");
    await expect(page.getByTestId("cej-detail-spans-table")).toBeVisible();
    await expect(
      page.getByTestId("cej-detail-spans-table").locator("tbody tr"),
    ).toHaveCount(3);
  });

  test("failed job surfaces error message on detail", async ({ page }) => {
    const state: MockState = { jobs: [FAILED_JOB] };
    await setupMocks(page, state);
    await page.goto(
      `/app/sequence-models/continuous-embedding/${FAILED_JOB.id}`,
    );

    await expect(page.getByTestId("cej-detail-status")).toHaveText("failed");
    await expect(
      page.getByTestId("cej-detail-error-message"),
    ).toContainText("embedder crashed mid-run");
  });
});
