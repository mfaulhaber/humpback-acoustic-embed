import { expect, test, type Page } from "@playwright/test";

const CEJ_COMPLETE = {
  id: "cej-complete-hmm",
  status: "complete",
  region_detection_job_id: "rj-1",
  model_version: "surfperch-tensorflow2",
  window_size_seconds: 5.0,
  hop_seconds: 1.0,
  pad_seconds: 10.0,
  target_sample_rate: 32000,
  feature_config_json: null,
  encoding_signature: "cej-hmm-sig",
  vector_dim: 1280,
  total_regions: 4,
  merged_spans: 2,
  total_windows: 120,
  parquet_path: "/tmp/data/continuous_embeddings/cej-complete-hmm/embeddings.parquet",
  error_message: null,
  created_at: "2026-04-27T00:00:00Z",
  updated_at: "2026-04-27T00:10:00Z",
};

const QUEUED_JOB = {
  id: "hmm-queued-1",
  status: "queued",
  continuous_embedding_job_id: CEJ_COMPLETE.id,
  n_states: 4,
  pca_dims: 50,
  pca_whiten: false,
  l2_normalize: true,
  covariance_type: "diag",
  n_iter: 100,
  random_seed: 42,
  min_sequence_length_frames: 10,
  tol: 0.0001,
  library: "hmmlearn",
  train_log_likelihood: null,
  n_train_sequences: null,
  n_train_frames: null,
  n_decoded_sequences: null,
  artifact_dir: null,
  error_message: null,
  created_at: "2026-04-27T01:00:00Z",
  updated_at: "2026-04-27T01:00:00Z",
};

const COMPLETE_JOB = {
  ...QUEUED_JOB,
  id: "hmm-complete-1",
  status: "complete",
  train_log_likelihood: -12345.6,
  n_train_sequences: 2,
  n_train_frames: 100,
  n_decoded_sequences: 2,
  artifact_dir: "/tmp/data/hmm_sequences/hmm-complete-1",
};

const SUMMARY = [
  { state: 0, occupancy: 0.6, mean_dwell_frames: 5.0, dwell_histogram: [2, 3, 1] },
  { state: 1, occupancy: 0.25, mean_dwell_frames: 3.0, dwell_histogram: [4, 1] },
  { state: 2, occupancy: 0.1, mean_dwell_frames: 2.5, dwell_histogram: [3, 2] },
  { state: 3, occupancy: 0.05, mean_dwell_frames: 1.5, dwell_histogram: [5] },
];

const TRANSITIONS = {
  n_states: 4,
  matrix: [
    [0.7, 0.2, 0.05, 0.05],
    [0.1, 0.6, 0.2, 0.1],
    [0.05, 0.15, 0.7, 0.1],
    [0.1, 0.1, 0.1, 0.7],
  ],
};

const DWELL = {
  n_states: 4,
  histograms: {
    "0": [2, 3, 1],
    "1": [4, 1],
    "2": [3, 2],
    "3": [5],
  },
};

const STATES = {
  total: 4,
  offset: 0,
  limit: 5000,
  items: [
    { merged_span_id: 0, window_index_in_span: 0, viterbi_state: 0, start_timestamp: 100.0, end_timestamp: 105.0, state_posterior: [0.9, 0.05, 0.03, 0.02], max_state_probability: 0.9, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, source_region_ids: ["r1"] },
    { merged_span_id: 0, window_index_in_span: 1, viterbi_state: 1, start_timestamp: 101.0, end_timestamp: 106.0, state_posterior: [0.1, 0.8, 0.05, 0.05], max_state_probability: 0.8, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, source_region_ids: ["r1"] },
    { merged_span_id: 1, window_index_in_span: 0, viterbi_state: 2, start_timestamp: 200.0, end_timestamp: 205.0, state_posterior: [0.05, 0.05, 0.85, 0.05], max_state_probability: 0.85, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, source_region_ids: ["r2"] },
    { merged_span_id: 1, window_index_in_span: 1, viterbi_state: 3, start_timestamp: 201.0, end_timestamp: 206.0, state_posterior: [0.02, 0.03, 0.05, 0.9], max_state_probability: 0.9, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, source_region_ids: ["r2"] },
  ],
};

const OVERLAY = {
  total: 4,
  items: [
    { merged_span_id: 0, window_index_in_span: 0, start_timestamp: 100.0, end_timestamp: 105.0, pca_x: 1.0, pca_y: 2.0, umap_x: 0.5, umap_y: 0.8, viterbi_state: 0, max_state_probability: 0.9 },
    { merged_span_id: 0, window_index_in_span: 1, start_timestamp: 101.0, end_timestamp: 106.0, pca_x: 1.5, pca_y: 2.5, umap_x: 0.6, umap_y: 0.9, viterbi_state: 1, max_state_probability: 0.8 },
    { merged_span_id: 1, window_index_in_span: 0, start_timestamp: 200.0, end_timestamp: 205.0, pca_x: -1.0, pca_y: -0.5, umap_x: -0.3, umap_y: 0.1, viterbi_state: 2, max_state_probability: 0.85 },
    { merged_span_id: 1, window_index_in_span: 1, start_timestamp: 201.0, end_timestamp: 206.0, pca_x: -0.8, pca_y: -0.2, umap_x: -0.1, umap_y: 0.2, viterbi_state: 3, max_state_probability: 0.9 },
  ],
};

const LABEL_DISTRIBUTION = {
  n_states: 4,
  total_windows: 4,
  states: {
    "0": { song: 1 },
    "1": { call: 1 },
    "2": { unlabeled: 1 },
    "3": { song: 1 },
  },
};

const EXEMPLARS = {
  n_states: 4,
  states: {
    "0": [
      { merged_span_id: 0, window_index_in_span: 0, audio_file_id: 1, start_timestamp: 100.0, end_timestamp: 105.0, max_state_probability: 0.9, exemplar_type: "high_confidence" },
    ],
    "1": [
      { merged_span_id: 0, window_index_in_span: 1, audio_file_id: 1, start_timestamp: 101.0, end_timestamp: 106.0, max_state_probability: 0.8, exemplar_type: "high_confidence" },
    ],
    "2": [
      { merged_span_id: 1, window_index_in_span: 0, audio_file_id: 1, start_timestamp: 200.0, end_timestamp: 205.0, max_state_probability: 0.85, exemplar_type: "mean_nearest" },
    ],
    "3": [],
  },
};

interface MockState {
  hmmJobs: typeof QUEUED_JOB[];
  regionJobId?: string;
  audioUrls?: string[];
}

async function setupMocks(page: Page, state: MockState) {
  await page.route("**/sequence-models/continuous-embeddings**", (route) => {
    const method = route.request().method();
    if (method === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([CEJ_COMPLETE]),
      });
    }
    return route.fulfill({ status: 405 });
  });

  await page.route("**/sequence-models/hmm-sequences**", (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const idMatch = url.match(/\/hmm-sequences\/([^/?#]+)/);

    if (idMatch) {
      const id = idMatch[1];

      if (url.includes("/cancel")) {
        const job = state.hmmJobs.find((j) => j.id === id);
        if (!job) return route.fulfill({ status: 404 });
        job.status = "canceled";
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(job),
        });
      }

      if (url.includes("/states")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(STATES),
        });
      }

      if (url.includes("/transitions")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(TRANSITIONS),
        });
      }

      if (url.includes("/dwell")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(DWELL),
        });
      }

      if (url.includes("/overlay")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(OVERLAY),
        });
      }

      if (url.includes("/label-distribution")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(LABEL_DISTRIBUTION),
        });
      }

      if (url.includes("/exemplars")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(EXEMPLARS),
        });
      }

      if (url.includes("/generate-interpretations") && method === "POST") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ status: "ok", job_id: id }),
        });
      }

      // Detail endpoint
      const job = state.hmmJobs.find((j) => j.id === id);
      if (!job) return route.fulfill({ status: 404 });
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job,
          region_detection_job_id:
            state.regionJobId ?? CEJ_COMPLETE.region_detection_job_id,
          region_start_timestamp: 100.0,
          region_end_timestamp: 700.0,
          summary: job.status === "complete" ? SUMMARY : null,
        }),
      });
    }

    if (method === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(state.hmmJobs),
      });
    }

    if (method === "POST") {
      state.hmmJobs = [QUEUED_JOB, ...state.hmmJobs];
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(QUEUED_JOB),
      });
    }

    return route.fulfill({ status: 405 });
  });

  // Stub tile and audio-slice endpoints used by the timeline viewer
  await page.route("**/call-parsing/region-jobs/*/tile**", (route) => {
    // Return a 1x1 transparent PNG
    const pixel = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "base64",
    );
    return route.fulfill({ status: 200, contentType: "image/png", body: pixel });
  });

  await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) => {
    state.audioUrls?.push(route.request().url());
    return route.fulfill({
      status: 200,
      contentType: "audio/mpeg",
      body: Buffer.alloc(0),
    });
  });
}

test.describe("Sequence Models — HMM Sequence", () => {
  test("nav reaches the HMM Sequence page", async ({ page }) => {
    const state: MockState = { hmmJobs: [QUEUED_JOB, COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/hmm-sequence");
    await expect(page.getByTestId("hmm-jobs-page")).toBeVisible();
    await expect(page.getByText("Active Jobs")).toBeVisible();
    await expect(page.getByText("Previous Jobs")).toBeVisible();
  });

  test("create form is constrained to completed continuous-embedding jobs", async ({
    page,
  }) => {
    const state: MockState = { hmmJobs: [] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/hmm-sequence");

    const select = page.getByTestId("hmm-source-select");
    await expect(select).toBeVisible();
    const options = select.locator("option");
    // Placeholder + completed CEJ
    await expect(options).toHaveCount(2);
    await expect(options.nth(1)).toContainText(CEJ_COMPLETE.id.slice(0, 8));
  });

  test("detail page renders all chart containers on a complete job", async ({
    page,
  }) => {
    const state: MockState = { hmmJobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/hmm-sequence/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("hmm-detail-page")).toBeVisible();
    await expect(page.getByTestId("hmm-detail-status")).toHaveText("complete");
    await expect(page.getByTestId("hmm-timeline-viewer")).toBeVisible();
    await expect(page.getByTestId("hmm-state-timeline")).toBeVisible();
    await expect(page.getByTestId("hmm-pca-umap-scatter")).toBeVisible();
    await expect(page.getByTestId("hmm-transition-heatmap")).toBeVisible();
    await expect(page.getByTestId("hmm-label-distribution")).toBeVisible();
    await expect(page.getByTestId("hmm-dwell-histograms")).toBeVisible();
    await expect(page.getByTestId("hmm-exemplar-gallery")).toBeVisible();
  });

  test("span selector switches between merged spans", async ({ page }) => {
    const state: MockState = { hmmJobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/hmm-sequence/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("hmm-detail-page")).toBeVisible();
    const selector = page.getByTestId("hmm-span-selector");
    await expect(selector).toBeVisible();
    const options = selector.locator("option");
    await expect(options).toHaveCount(2);
    await expect(options.nth(0)).toHaveText("Span 0");
    await expect(options.nth(1)).toHaveText("Span 1");

    // Switch to span 1
    await selector.selectOption("1");
    await expect(page.getByTestId("hmm-state-timeline")).toBeVisible();
  });

  test("HMM State Timeline Viewer panel renders with spectrogram and state bar", async ({
    page,
  }) => {
    const state: MockState = { hmmJobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/hmm-sequence/${COMPLETE_JOB.id}`);

    // Timeline viewer panel is visible
    await expect(page.getByTestId("hmm-timeline-viewer")).toBeVisible();

    // Spectrogram viewport present within the panel
    const viewerPanel = page.getByTestId("hmm-timeline-viewer");
    const viewport = viewerPanel.getByTestId("spectrogram-viewport");
    await expect(viewport).toBeVisible();
    await expect
      .poll(async () => (await viewport.boundingBox())?.height ?? 0)
      .toBeGreaterThan(100);

    // HMMStateBar canvas present
    await expect(viewerPanel.getByTestId("hmm-state-bar")).toBeVisible();

    // Span nav is visible with correct label
    await expect(page.getByTestId("hmm-span-nav")).toBeVisible();
    await expect(page.getByTestId("hmm-span-label")).toContainText("Span 1/2");

    // Prev disabled at first span, Next enabled
    await expect(page.getByTestId("hmm-span-prev")).toBeDisabled();
    await expect(page.getByTestId("hmm-span-next")).toBeEnabled();

    // User-selected zoom survives span navigation.
    const zoomOneMinute = viewerPanel.getByRole("button", { name: "1m" });
    await zoomOneMinute.click();
    await expect(zoomOneMinute).toHaveClass(/text-primary/);

    // Click next span — label updates
    await page.getByTestId("hmm-span-next").click();
    await expect(page.getByTestId("hmm-span-label")).toContainText("Span 2/2");
    await expect(zoomOneMinute).toHaveClass(/text-primary/);

    // Now next is disabled, prev is enabled
    await expect(page.getByTestId("hmm-span-next")).toBeDisabled();
    await expect(page.getByTestId("hmm-span-prev")).toBeEnabled();

    // Zoom preset buttons are present (ZoomSelector renders buttons)
    const zoomButtons = viewerPanel.locator("button").filter({ hasText: /^\d+[smh]$/ });
    await expect(zoomButtons.first()).toBeVisible();
  });
});
