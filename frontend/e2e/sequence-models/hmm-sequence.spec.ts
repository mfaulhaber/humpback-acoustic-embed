import { expect, test, type Page } from "@playwright/test";

const CEJ_COMPLETE = {
  id: "cej-complete-hmm",
  status: "complete",
  event_segmentation_job_id: "seg-1",
  model_version: "surfperch-tensorflow2",
  window_size_seconds: 5.0,
  hop_seconds: 1.0,
  pad_seconds: 2.0,
  target_sample_rate: 32000,
  feature_config_json: null,
  encoding_signature: "cej-hmm-sig",
  vector_dim: 1280,
  total_events: 4,
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
  min_sequence_length_frames: 3,
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
    { merged_span_id: 0, window_index_in_span: 0, viterbi_state: 0, start_timestamp: 100.0, end_timestamp: 105.0, state_posterior: [0.9, 0.05, 0.03, 0.02], max_state_probability: 0.9, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, event_id: "evt-0" },
    { merged_span_id: 0, window_index_in_span: 1, viterbi_state: 1, start_timestamp: 101.0, end_timestamp: 106.0, state_posterior: [0.1, 0.8, 0.05, 0.05], max_state_probability: 0.8, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, event_id: "evt-0" },
    { merged_span_id: 1, window_index_in_span: 0, viterbi_state: 2, start_timestamp: 200.0, end_timestamp: 205.0, state_posterior: [0.05, 0.05, 0.85, 0.05], max_state_probability: 0.85, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, event_id: "evt-1" },
    { merged_span_id: 1, window_index_in_span: 1, viterbi_state: 3, start_timestamp: 201.0, end_timestamp: 206.0, state_posterior: [0.02, 0.03, 0.05, 0.9], max_state_probability: 0.9, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, event_id: "evt-1" },
  ],
};

const OVERLAY = {
  total: 4,
  items: [
    { sequence_id: "0", position_in_sequence: 0, start_timestamp: 100.0, end_timestamp: 105.0, pca_x: 1.0, pca_y: 2.0, umap_x: 0.5, umap_y: 0.8, viterbi_state: 0, max_state_probability: 0.9 },
    { sequence_id: "0", position_in_sequence: 1, start_timestamp: 101.0, end_timestamp: 106.0, pca_x: 1.5, pca_y: 2.5, umap_x: 0.6, umap_y: 0.9, viterbi_state: 1, max_state_probability: 0.8 },
    { sequence_id: "1", position_in_sequence: 0, start_timestamp: 200.0, end_timestamp: 205.0, pca_x: -1.0, pca_y: -0.5, umap_x: -0.3, umap_y: 0.1, viterbi_state: 2, max_state_probability: 0.85 },
    { sequence_id: "1", position_in_sequence: 1, start_timestamp: 201.0, end_timestamp: 206.0, pca_x: -0.8, pca_y: -0.2, umap_x: -0.1, umap_y: 0.2, viterbi_state: 3, max_state_probability: 0.9 },
  ],
};

const LABEL_DISTRIBUTION = {
  n_states: 4,
  total_windows: 4,
  states: {
    "0": { all: { song: 1 } },
    "1": { all: { call: 1 } },
    "2": { all: { unlabeled: 1 } },
    "3": { all: { song: 1 } },
  },
};

const EXEMPLARS = {
  n_states: 4,
  states: {
    "0": [
      { sequence_id: "0", position_in_sequence: 0, audio_file_id: 1, start_timestamp: 100.0, end_timestamp: 105.0, max_state_probability: 0.9, exemplar_type: "high_confidence", extras: {} },
    ],
    "1": [
      { sequence_id: "0", position_in_sequence: 1, audio_file_id: 1, start_timestamp: 101.0, end_timestamp: 106.0, max_state_probability: 0.8, exemplar_type: "high_confidence", extras: {} },
    ],
    "2": [
      { sequence_id: "1", position_in_sequence: 0, audio_file_id: 1, start_timestamp: 200.0, end_timestamp: 205.0, max_state_probability: 0.85, exemplar_type: "mean_nearest", extras: {} },
    ],
    "3": [],
  },
};

const CRNN_EXEMPLARS = {
  n_states: 4,
  states: {
    "0": [
      { sequence_id: "zzz-first", position_in_sequence: 0, audio_file_id: 1, start_timestamp: 100.0, end_timestamp: 100.25, max_state_probability: 0.9, exemplar_type: "high_confidence", extras: { tier: "event_core" } },
    ],
    "1": [
      { sequence_id: "aaa-second", position_in_sequence: 1, audio_file_id: 1, start_timestamp: 200.25, end_timestamp: 200.5, max_state_probability: 0.8, exemplar_type: "high_confidence", extras: { tier: "near_event" } },
    ],
    "2": [
      { sequence_id: "mmm-third", position_in_sequence: 0, audio_file_id: 1, start_timestamp: 300.0, end_timestamp: 300.25, max_state_probability: 0.85, exemplar_type: "mean_nearest", extras: { tier: "background" } },
    ],
    "3": [],
  },
};

const MOTIF_JOB = {
  id: "motif-1",
  status: "complete",
  hmm_sequence_job_id: COMPLETE_JOB.id,
  source_kind: "surfperch",
  min_ngram: 2,
  max_ngram: 8,
  minimum_occurrences: 5,
  minimum_event_sources: 2,
  frequency_weight: 0.4,
  event_source_weight: 0.3,
  event_core_weight: 0.2,
  low_background_weight: 0.1,
  call_probability_weight: null,
  config_signature: "motif-sig",
  total_groups: 2,
  total_collapsed_tokens: 6,
  total_candidate_occurrences: 4,
  total_motifs: 1,
  artifact_dir: "/tmp/data/motif_extractions/motif-1",
  error_message: null,
  created_at: "2026-04-30T00:00:00Z",
  updated_at: "2026-04-30T00:01:00Z",
};

const MOTIFS = {
  total: 1,
  offset: 0,
  limit: 100,
  items: [
    {
      motif_key: "2-3",
      states: [2, 3],
      length: 2,
      occurrence_count: 5,
      event_source_count: 2,
      audio_source_count: 2,
      group_count: 2,
      event_core_fraction: 0.8,
      background_fraction: 0.1,
      mean_call_probability: null,
      mean_duration_seconds: 2.0,
      median_duration_seconds: 2.0,
      rank_score: 0.95,
      example_occurrence_ids: ["occ-1"],
    },
  ],
};

const MOTIF_OCCURRENCES = {
  total: 1,
  offset: 0,
  limit: 100,
  items: [
    {
      occurrence_id: "occ-1",
      motif_key: "2-3",
      states: [2, 3],
      source_kind: "surfperch",
      group_key: "1",
      event_source_key: "evt-1",
      audio_source_key: "1",
      token_start_index: 0,
      token_end_index: 1,
      raw_start_index: 0,
      raw_end_index: 1,
      start_timestamp: 202.0,
      end_timestamp: 203.0,
      duration_seconds: 1.0,
      event_core_fraction: 1.0,
      background_fraction: 0.0,
      mean_call_probability: null,
      anchor_event_id: "evt-1",
      anchor_timestamp: 203.0,
      relative_start_seconds: -1.0,
      relative_end_seconds: 0.0,
      anchor_strategy: "event_midpoint",
    },
  ],
};

interface MockState {
  hmmJobs: typeof QUEUED_JOB[];
  motifJobs?: typeof MOTIF_JOB[];
  regionJobId?: string;
  audioUrls?: string[];
}

async function setupMocks(page: Page, state: MockState) {
  await page.route("**/sequence-models/continuous-embeddings**", (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const idMatch = url.match(/\/continuous-embeddings\/([^/?#]+)/);
    if (idMatch) {
      const id = idMatch[1];
      if (id === CEJ_COMPLETE.id) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            job: CEJ_COMPLETE,
            manifest: {
              job_id: CEJ_COMPLETE.id,
              model_version: CEJ_COMPLETE.model_version,
              vector_dim: 1280,
              window_size_seconds: 5.0,
              hop_seconds: 1.0,
              pad_seconds: 2.0,
              target_sample_rate: 32000,
              total_events: 4,
              merged_spans: 2,
              total_windows: 120,
              spans: [
                {
                  merged_span_id: 0,
                  start_timestamp: 100.0,
                  end_timestamp: 106.0,
                  window_count: 2,
                  event_id: "evt-0",
                  region_id: "r1",
                },
                {
                  merged_span_id: 1,
                  start_timestamp: 200.0,
                  end_timestamp: 206.0,
                  window_count: 2,
                  event_id: "evt-1",
                  region_id: "r1",
                },
              ],
            },
          }),
        });
      }
      return route.fulfill({ status: 404 });
    }
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
          region_detection_job_id: state.regionJobId ?? "region-job-1",
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

  await page.route("**/sequence-models/motif-extractions**", (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const idMatch = url.match(/\/motif-extractions\/([^/?#]+)/);
    state.motifJobs ??= [];

    if (idMatch) {
      const id = idMatch[1];
      const job = state.motifJobs.find((j) => j.id === id);
      if (!job) return route.fulfill({ status: 404 });

      if (url.includes("/occurrences")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(MOTIF_OCCURRENCES),
        });
      }
      if (url.includes("/motifs")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(MOTIFS),
        });
      }
      if (url.includes("/cancel") && method === "POST") {
        job.status = "canceled";
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(job),
        });
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ job, manifest: null }),
      });
    }

    if (method === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(state.motifJobs),
      });
    }
    if (method === "POST") {
      const newJob = {
        ...MOTIF_JOB,
        id: "motif-created",
        status: "queued",
      };
      state.motifJobs = [newJob, ...state.motifJobs];
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(newJob),
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
    await expect(page.getByTestId("motif-extraction-panel")).toBeVisible();
    await expect(page.getByTestId("hmm-state-timeline")).toBeVisible();
    await expect(page.getByTestId("hmm-pca-umap-scatter")).toBeVisible();
    await expect(page.getByTestId("hmm-transition-heatmap")).toBeVisible();
    await expect(page.getByTestId("hmm-label-distribution")).toBeVisible();
    await expect(page.getByTestId("hmm-dwell-histograms")).toBeVisible();
    await expect(page.getByTestId("hmm-exemplar-gallery")).toBeVisible();
  });

  test("motifs panel creates and renders completed motif jobs", async ({ page }) => {
    const state: MockState = { hmmJobs: [COMPLETE_JOB], motifJobs: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/hmm-sequence/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("motif-create-form")).toBeVisible();
    await page.getByText("Advanced").click();
    await expect(page.getByLabel("frequency")).toBeVisible();
    await page.getByTestId("motif-create-submit").click();
    await expect(page.getByTestId("motif-running")).toContainText("queued");

    state.motifJobs = [MOTIF_JOB];
    await page.reload();
    await expect(page.getByTestId("motif-table")).toBeVisible();
    await expect(page.getByText("2-3")).toBeVisible();
    await expect(page.getByTestId("motif-example-alignment")).toBeVisible();
    await page.getByRole("button", { name: "Jump" }).click();
    await expect(page.getByTestId("hmm-span-label")).toContainText("Event 2/2");
    await expect(page.getByTestId("timeline-center-time")).toContainText("00:03:22 UTC");
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

    // Two-layer overlay container is in place: clipped band layer keeps
    // overlays from bleeding past the canvas edge; sibling tooltip layer
    // is unclipped so DetectionOverlay/VocalizationOverlay tooltips
    // remain readable past the boundary.
    const band = viewerPanel.getByTestId("overlay-band-layer");
    await expect(band).toBeAttached();
    const bandOverflow = await band.evaluate((el) => getComputedStyle(el).overflow);
    expect(bandOverflow).toBe("hidden");
    await expect(viewerPanel.getByTestId("overlay-tooltip-layer")).toBeAttached();

    // Span nav is visible with correct label
    await expect(page.getByTestId("hmm-span-nav")).toBeVisible();
    await expect(page.getByTestId("hmm-span-label")).toContainText("Event 1/2");

    // Prev disabled at first span, Next enabled
    await expect(page.getByTestId("hmm-span-prev")).toBeDisabled();
    await expect(page.getByTestId("hmm-span-next")).toBeEnabled();

    // User-selected zoom survives span navigation.
    const zoomOneMinute = viewerPanel.getByRole("button", { name: "1m" });
    await zoomOneMinute.click();
    await expect(zoomOneMinute).toHaveClass(/text-primary/);

    // Click next span — label updates
    await page.getByTestId("hmm-span-next").click();
    await expect(page.getByTestId("hmm-span-label")).toContainText("Event 2/2");
    await expect(zoomOneMinute).toHaveClass(/text-primary/);

    // Now next is disabled, prev is enabled
    await expect(page.getByTestId("hmm-span-next")).toBeDisabled();
    await expect(page.getByTestId("hmm-span-prev")).toBeEnabled();

    // Zoom preset buttons are present (ZoomSelector renders buttons)
    const zoomButtons = viewerPanel.locator("button").filter({ hasText: /^\d+[smh]$/ });
    await expect(zoomButtons.first()).toBeVisible();
  });

  test("CRNN-source detail page renders one navigable span per region", async ({
    page,
  }) => {
    // Regression for: CRNN states.parquet uses region_id, not merged_span_id.
    // Frontend must group/filter by region_id so all regions are reachable.
    const CRNN_CEJ = {
      ...CEJ_COMPLETE,
      id: "cej-crnn-1",
      model_version: "crnn-region-bigru-v1",
      region_detection_job_id: "region-job-crnn",
      event_segmentation_job_id: null,
      total_regions: 3,
      total_chunks: 6,
    };
    const CRNN_JOB = {
      ...COMPLETE_JOB,
      id: "hmm-crnn-1",
      continuous_embedding_job_id: CRNN_CEJ.id,
      training_mode: "event_balanced",
    };
    // Region IDs deliberately chosen so lexicographic order ("aaa", "mmm",
    // "zzz") differs from chronological order ("zzz" first, "aaa" second,
    // "mmm" third). Catches a regression where spanIds is sorted as strings.
    const CRNN_STATES = {
      total: 6,
      offset: 0,
      limit: 5000,
      items: [
        { region_id: "zzz-first", chunk_index_in_region: 0, viterbi_state: 0, start_timestamp: 100.0, end_timestamp: 100.25, state_posterior: [0.9, 0.05, 0.03, 0.02], max_state_probability: 0.9, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, tier: "event_core" },
        { region_id: "zzz-first", chunk_index_in_region: 1, viterbi_state: 1, start_timestamp: 100.25, end_timestamp: 100.5, state_posterior: [0.1, 0.8, 0.05, 0.05], max_state_probability: 0.8, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, tier: "event_core" },
        { region_id: "aaa-second", chunk_index_in_region: 0, viterbi_state: 2, start_timestamp: 200.0, end_timestamp: 200.25, state_posterior: [0.05, 0.05, 0.85, 0.05], max_state_probability: 0.85, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, tier: "near_event" },
        { region_id: "aaa-second", chunk_index_in_region: 1, viterbi_state: 3, start_timestamp: 200.25, end_timestamp: 200.5, state_posterior: [0.02, 0.03, 0.05, 0.9], max_state_probability: 0.9, was_used_for_training: true, audio_file_id: 1, is_in_pad: false, tier: "near_event" },
        { region_id: "mmm-third", chunk_index_in_region: 0, viterbi_state: 0, start_timestamp: 300.0, end_timestamp: 300.25, state_posterior: [0.7, 0.1, 0.1, 0.1], max_state_probability: 0.7, was_used_for_training: false, audio_file_id: 1, is_in_pad: false, tier: "background" },
        { region_id: "mmm-third", chunk_index_in_region: 1, viterbi_state: 1, start_timestamp: 300.25, end_timestamp: 300.5, state_posterior: [0.1, 0.7, 0.1, 0.1], max_state_probability: 0.7, was_used_for_training: false, audio_file_id: 1, is_in_pad: false, tier: "background" },
      ],
    };

    await page.route("**/sequence-models/continuous-embeddings/cej-crnn-1**", (route) => {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job: CRNN_CEJ,
          manifest: {
            job_id: CRNN_CEJ.id,
            model_version: CRNN_CEJ.model_version,
            source_kind: "region_crnn",
            vector_dim: 512,
            target_sample_rate: 32000,
            region_detection_job_id: CRNN_CEJ.region_detection_job_id,
            chunk_size_seconds: 0.25,
            chunk_hop_seconds: 0.25,
            total_regions: 3,
            total_chunks: 6,
            regions: [
              { region_id: "zzz-first", start_timestamp: 100.0, end_timestamp: 100.5, chunk_count: 2 },
              { region_id: "aaa-second", start_timestamp: 200.0, end_timestamp: 200.5, chunk_count: 2 },
              { region_id: "mmm-third", start_timestamp: 300.0, end_timestamp: 300.5, chunk_count: 2 },
            ],
          },
        }),
      });
    });

    await page.route("**/sequence-models/hmm-sequences/hmm-crnn-1**", (route) => {
      const url = route.request().url();
      if (url.includes("/states")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(CRNN_STATES),
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
          body: JSON.stringify(CRNN_EXEMPLARS),
        });
      }
      // Detail endpoint
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job: CRNN_JOB,
          region_detection_job_id: "region-job-crnn",
          region_start_timestamp: 100.0,
          region_end_timestamp: 300.5,
          summary: SUMMARY,
          source_kind: "region_crnn",
          tier_composition: [
            { state: 0, event_core: 0.5, near_event: 0.0, background: 0.5 },
            { state: 1, event_core: 0.5, near_event: 0.0, background: 0.5 },
            { state: 2, event_core: 0.0, near_event: 1.0, background: 0.0 },
            { state: 3, event_core: 0.0, near_event: 1.0, background: 0.0 },
          ],
        }),
      });
    });

    await page.route("**/call-parsing/region-jobs/*/tile**", (route) => {
      const pixel = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "base64",
      );
      return route.fulfill({ status: 200, contentType: "image/png", body: pixel });
    });
    await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) => {
      return route.fulfill({
        status: 200,
        contentType: "audio/mpeg",
        body: Buffer.alloc(0),
      });
    });

    await page.goto(`/app/sequence-models/hmm-sequence/${CRNN_JOB.id}`);
    await expect(page.getByTestId("hmm-detail-page")).toBeVisible();
    await expect(page.getByTestId("hmm-detail-source-kind")).toHaveText("CRNN");

    // Three distinct regions → three navigable items (not one).
    await expect(page.getByTestId("hmm-span-label")).toContainText("Region 1/3");
    const selector = page.getByTestId("hmm-span-selector");
    await expect(selector).toBeVisible();
    const options = selector.locator("option");
    await expect(options).toHaveCount(3);

    // Order is chronological (start_timestamp), not lexicographic on
    // region_id — "zzz-first" comes before "aaa-second".
    await expect(options.nth(0)).toContainText("zzz-firs");
    await expect(options.nth(1)).toContainText("aaa-seco");
    await expect(options.nth(2)).toContainText("mmm-thir");

    // Stepping forward must reach the third region.
    await page.getByTestId("hmm-span-next").click();
    await expect(page.getByTestId("hmm-span-label")).toContainText("Region 2/3");
    await page.getByTestId("hmm-span-next").click();
    await expect(page.getByTestId("hmm-span-label")).toContainText("Region 3/3");
    await expect(page.getByTestId("hmm-span-next")).toBeDisabled();

    // Region-level nav is suppressed for CRNN (each span IS a region).
    await expect(page.getByTestId("hmm-region-nav")).toHaveCount(0);

    // Tier composition strip renders for CRNN.
    await expect(page.getByTestId("hmm-tier-composition-strip")).toBeVisible();

    // PCA/UMAP overlay panel is visible (Phase 1 unblocks the CRNN source).
    await expect(page.getByTestId("hmm-pca-umap-scatter")).toBeVisible();

    // Exemplar gallery renders tier badges for CRNN-source records.
    const badges = page.getByTestId("exemplar-tier-badge");
    await expect(badges.first()).toBeVisible();
    const badgeText = (await badges.first().textContent())?.trim();
    expect(["event_core", "near_event", "background"]).toContain(badgeText);

    // Label Distribution card renders for CRNN sources under ADR-060.
    // The mock returns unified-shape JSON (state → tier → label → count);
    // the chart's useMemo collapses tiers before plotting.
    await expect(page.getByTestId("hmm-generate-interpretations")).toBeVisible();
    await expect(page.getByTestId("hmm-label-distribution")).toBeVisible();
  });

  test("SurfPerch detail page renders overlay and exemplar cards without tier badge", async ({
    page,
  }) => {
    const state: MockState = { hmmJobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/hmm-sequence/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("hmm-pca-umap-scatter")).toBeVisible();
    await expect(page.getByTestId("hmm-exemplar-gallery")).toBeVisible();
    // SurfPerch records have no extras.tier — no badges should render.
    await expect(page.getByTestId("exemplar-tier-badge")).toHaveCount(0);
  });
});
