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

const SEG_JOB_OTHER = {
  ...SEG_JOB,
  id: "seg-complete-2",
  event_count: 8,
  created_at: "2026-04-26T02:00:00Z",
  updated_at: "2026-04-26T02:10:00Z",
  started_at: "2026-04-26T02:00:01Z",
  completed_at: "2026-04-26T02:10:00Z",
};

const CRNN_EMBEDDING_JOB = {
  id: "cej-crnn-complete-1",
  status: "complete",
  event_segmentation_job_id: SEG_JOB.id,
  event_source_mode: "raw",
  model_version: "crnn-region-v1",
  window_size_seconds: null,
  hop_seconds: null,
  pad_seconds: null,
  target_sample_rate: 16000,
  feature_config_json: null,
  encoding_signature: "crnn-embedding-sig",
  vector_dim: 64,
  total_events: null,
  merged_spans: null,
  total_windows: null,
  parquet_path:
    "/tmp/data/continuous_embeddings/cej-crnn-complete-1/embeddings.parquet",
  error_message: null,
  region_detection_job_id: REGION_JOB.id,
  chunk_size_seconds: 2.0,
  chunk_hop_seconds: 1.0,
  crnn_checkpoint_sha256: "abc123",
  crnn_segmentation_model_id: "crnn-seg-v1",
  projection_kind: "pca",
  projection_dim: 64,
  total_regions: 4,
  total_chunks: 128,
  created_at: "2026-04-27T00:00:00Z",
  updated_at: "2026-04-27T00:10:00Z",
};

const CRNN_EMBEDDING_JOB_OTHER = {
  ...CRNN_EMBEDDING_JOB,
  id: "cej-crnn-complete-2",
  event_segmentation_job_id: SEG_JOB_OTHER.id,
  encoding_signature: "crnn-embedding-sig-2",
  total_chunks: 96,
  created_at: "2026-04-27T02:00:00Z",
  updated_at: "2026-04-27T02:10:00Z",
};

const QUEUED_JOB = {
  id: "eej-queued-1",
  status: "queued",
  event_segmentation_job_id: SEG_JOB.id,
  event_source_mode: "raw",
  continuous_embedding_job_id: CRNN_EMBEDDING_JOB.id,
  continuous_embedding_signature: CRNN_EMBEDDING_JOB.encoding_signature,
  tokenizer_version: "crnn-event-encoder-v1",
  pooling_config_json:
    '{"enabled_pools":["mean_pool","top_k_pool","start_pool","middle_pool","end_pool"],"top_k_fraction":0.25,"min_overlap_fraction":0.25,"min_chunks_per_event":1}',
  descriptor_config_json:
    '{"target_sample_rate":16000,"n_fft":1024,"hop_length":512,"eps":1e-12}',
  preprocessing_config_json:
    '{"l2_normalize_pools":true,"pca_dim":128,"embedding_weight":1.0,"descriptor_weight":1.0}',
  k_values_json: "[50,100,200]",
  random_seed: 0,
  tokenization_signature: "queued-token-sig",
  event_vector_dim: null,
  total_events: null,
  encoded_events: null,
  skipped_events: null,
  event_vectors_path: null,
  event_tokens_path: null,
  token_sequences_path: null,
  manifest_path: null,
  report_path: null,
  error_message: null,
  created_at: "2026-04-27T01:00:00Z",
  updated_at: "2026-04-27T01:00:00Z",
};

const COMPLETE_JOB = {
  ...QUEUED_JOB,
  id: "eej-complete-1",
  status: "complete",
  tokenization_signature: "complete-token-sig",
  event_vector_dim: 136,
  total_events: 12,
  encoded_events: 11,
  skipped_events: 1,
  event_vectors_path: "/tmp/event-encoders/eej-complete-1/event_vectors.parquet",
  event_tokens_path: "/tmp/event-encoders/eej-complete-1/event_tokens.parquet",
  token_sequences_path:
    "/tmp/event-encoders/eej-complete-1/token_sequences.parquet",
  manifest_path: "/tmp/event-encoders/eej-complete-1/manifest.json",
  report_path: "/tmp/event-encoders/eej-complete-1/report.json",
};

const FAILED_JOB = {
  ...QUEUED_JOB,
  id: "eej-failed-1",
  status: "failed",
  tokenization_signature: "failed-token-sig",
  error_message: "event encoder could not encode any events",
};

const COMPLETE_DETAIL = {
  job: COMPLETE_JOB,
  manifest: {
    job_id: COMPLETE_JOB.id,
    tokenizer_version: COMPLETE_JOB.tokenizer_version,
    event_segmentation_job_id: SEG_JOB.id,
    continuous_embedding_job_id: CRNN_EMBEDDING_JOB.id,
    continuous_embedding_signature: CRNN_EMBEDDING_JOB.encoding_signature,
    valid_k_values: [50, 100],
    invalid_k_values: [200],
    event_vector_dim: 136,
    total_events: 12,
    encoded_events: 11,
    skipped_events: 1,
  },
  report: {
    summary: {
      total_events: 12,
      encoded_events: 11,
      skipped_events: 1,
      valid_k_values: [50, 100],
      invalid_k_values: [200],
    },
    tokenization: {
      "50": {
        inertia: 12.345,
        token_counts: { "0": 3, "1": 8 },
      },
      "100": {
        inertia: 4.321,
        token_counts: { "0": 2, "1": 9 },
      },
    },
    token_examples: {
      "50": {
        T17: [{ event_id: "evt-17", distance_to_centroid: 0.1 }],
        T42: [{ event_id: "evt-42", distance_to_centroid: 0.2 }],
      },
      "100": {
        T05: [{ event_id: "evt-05", distance_to_centroid: 0.3 }],
      },
    },
    descriptor_summary: {
      duration: { mean: 1.25, min: 0.8, max: 2.1 },
      spectral_entropy: { mean: 0.42, min: 0.3, max: 0.6 },
    },
    sequence_preview: {
      "50": ["T17", "T42", "T17", "T08"],
      "100": ["T05", "T31", "T05", "T12"],
    },
  },
};

const FAILED_DETAIL = { job: FAILED_JOB, manifest: null, report: null };

interface MockState {
  jobs: typeof QUEUED_JOB[];
  lastCreateBody?: CreatePayload;
}

interface CreatePayload {
  preprocessing?: {
    l2_normalize_pools?: boolean;
  };
}

async function setupMocks(page: Page, state: MockState) {
  await page.route("**/call-parsing/segmentation-jobs**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([SEG_JOB, SEG_JOB_OTHER]),
    }),
  );

  await page.route(
    "**/sequence-models/continuous-embeddings**",
    (route) => {
      const method = route.request().method();
      if (method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([CRNN_EMBEDDING_JOB, CRNN_EMBEDDING_JOB_OTHER]),
        });
      }
      return route.fulfill({ status: 405 });
    },
  );

  await page.route("**/sequence-models/event-encoders**", async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const idMatch = url.match(/\/event-encoders\/([^/?#]+)/);

    if (idMatch) {
      const id = idMatch[1];
      if (url.includes("/cancel")) {
        const job = state.jobs.find((j) => j.id === id);
        if (!job) return route.fulfill({ status: 404 });
        job.status = "canceled";
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(job),
        });
      }
      if (method === "DELETE") {
        state.jobs = state.jobs.filter((j) => j.id !== id);
        return route.fulfill({ status: 204 });
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
      state.lastCreateBody =
        (await route.request().postDataJSON()) as CreatePayload;
      state.jobs = [QUEUED_JOB, ...state.jobs];
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(QUEUED_JOB),
      });
    }

    return route.fulfill({ status: 405 });
  });
}

async function segmentationOptionValues(page: Page): Promise<string[]> {
  return page
    .getByTestId("eej-seg-job-select")
    .locator("option")
    .evaluateAll((options) =>
      options.map((option) => (option as HTMLOptionElement).value),
    );
}

test.describe("Sequence Models - Event Encoder", () => {
  test("nav reaches the Event Encoder page", async ({ page }) => {
    const state: MockState = { jobs: [QUEUED_JOB, COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/event-encoder");
    await expect(page.getByTestId("eej-jobs-page")).toBeVisible();
    await expect(page.getByText("New Event Encoder Job")).toBeVisible();
    await expect(page.getByText("Previous Jobs")).toBeVisible();
  });

  test("create form posts and shows new job in Active", async ({ page }) => {
    const state: MockState = { jobs: [] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/event-encoder");

    await expect(page.getByTestId("eej-continuous-job-select")).toBeEnabled();
    await page
      .getByTestId("eej-continuous-job-select")
      .selectOption(CRNN_EMBEDDING_JOB.id);
    await expect(page.getByTestId("eej-seg-job-select")).toHaveValue(
      SEG_JOB.id,
    );
    await page.getByTestId("eej-advanced-toggle").click();
    await expect(page.getByTestId("eej-l2-normalize-pools")).toBeChecked();
    await page.getByTestId("eej-l2-normalize-pools").click();
    await page.getByTestId("eej-create-submit").click();

    expect(
      state.lastCreateBody?.preprocessing?.l2_normalize_pools,
    ).toBe(false);

    const queuedRow = page.locator("tr", {
      hasText: SEG_JOB.id.slice(0, 8),
    }).first();
    await expect(queuedRow).toBeVisible();
    await expect(queuedRow.getByText("queued")).toBeVisible();
  });

  test("create form filters segmentation choices by selected embedding", async ({
    page,
  }) => {
    const state: MockState = { jobs: [] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/event-encoder");

    await expect(page.getByTestId("eej-seg-job-select")).toBeDisabled();
    await page
      .getByTestId("eej-continuous-job-select")
      .selectOption(CRNN_EMBEDDING_JOB.id);
    await expect(page.getByTestId("eej-seg-job-select")).toBeEnabled();
    await expect(page.getByTestId("eej-seg-job-select")).toHaveValue(
      SEG_JOB.id,
    );
    expect(await segmentationOptionValues(page)).toEqual(["", SEG_JOB.id]);

    await page
      .getByTestId("eej-continuous-job-select")
      .selectOption(CRNN_EMBEDDING_JOB_OTHER.id);
    await expect(page.getByTestId("eej-seg-job-select")).toHaveValue(
      SEG_JOB_OTHER.id,
    );
    expect(await segmentationOptionValues(page)).toEqual(["", SEG_JOB_OTHER.id]);
  });

  test("complete detail page shows report stats", async ({ page }) => {
    const state: MockState = { jobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/event-encoder/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("eej-detail-page")).toBeVisible();
    await expect(page.getByTestId("eej-detail-status")).toHaveText("complete");
    await expect(page.getByTestId("eej-sequence-preview")).toContainText("T17");
    await expect(page.getByTestId("eej-tokenization-table")).toBeVisible();
    await expect(page.getByTestId("eej-exemplar-table")).toContainText(
      "evt-17",
    );
    await expect(page.getByTestId("eej-descriptor-table")).toContainText(
      "spectral_entropy",
    );
  });

  test("failed job surfaces error message on detail", async ({ page }) => {
    const state: MockState = { jobs: [FAILED_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/event-encoder/${FAILED_JOB.id}`);

    await expect(page.getByTestId("eej-detail-status")).toHaveText("failed");
    await expect(page.getByTestId("eej-detail-error-message")).toContainText(
      "could not encode",
    );
  });
});
