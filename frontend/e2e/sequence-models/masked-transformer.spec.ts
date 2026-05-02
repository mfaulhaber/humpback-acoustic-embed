import { expect, test, type Page, type Route } from "@playwright/test";

const CEJ_CRNN_COMPLETE = {
  id: "cej-crnn-mt",
  status: "complete",
  event_segmentation_job_id: "seg-1",
  model_version: "crnn-call-parsing-pytorch",
  window_size_seconds: null,
  hop_seconds: null,
  pad_seconds: null,
  target_sample_rate: 32000,
  feature_config_json: null,
  encoding_signature: "cej-mt-sig",
  vector_dim: 64,
  total_events: null,
  merged_spans: null,
  total_windows: null,
  parquet_path: "/tmp/data/continuous_embeddings/cej-crnn-mt/embeddings.parquet",
  error_message: null,
  region_detection_job_id: "rdj-mt-1",
  chunk_size_seconds: 0.25,
  chunk_hop_seconds: 0.25,
  crnn_checkpoint_sha256: "deadbeef",
  crnn_segmentation_model_id: "seg-model-1",
  projection_kind: "identity",
  projection_dim: 64,
  total_regions: 2,
  total_chunks: 12,
  created_at: "2026-04-29T00:00:00Z",
  updated_at: "2026-04-29T00:10:00Z",
};

const CEJ_SURFPERCH_COMPLETE = {
  ...CEJ_CRNN_COMPLETE,
  id: "cej-surfperch-mt",
  model_version: "surfperch-tensorflow2",
  encoding_signature: "cej-surf-sig",
  region_detection_job_id: null,
  chunk_size_seconds: null,
  chunk_hop_seconds: null,
  crnn_checkpoint_sha256: null,
  crnn_segmentation_model_id: null,
  projection_kind: null,
  projection_dim: null,
  total_regions: null,
  total_chunks: null,
};

const QUEUED_JOB = {
  id: "mt-queued",
  status: "queued",
  status_reason: null,
  continuous_embedding_job_id: CEJ_CRNN_COMPLETE.id,
  training_signature: "sig-q",
  preset: "default",
  mask_fraction: 0.2,
  span_length_min: 2,
  span_length_max: 6,
  dropout: 0.1,
  mask_weight_bias: true,
  cosine_loss_weight: 0.0,
  max_epochs: 30,
  early_stop_patience: 3,
  val_split: 0.1,
  seed: 42,
  k_values: [50, 100],
  chosen_device: null,
  fallback_reason: null,
  final_train_loss: null,
  final_val_loss: null,
  total_epochs: null,
  job_dir: null,
  total_sequences: null,
  total_chunks: null,
  error_message: null,
  created_at: "2026-04-29T01:00:00Z",
  updated_at: "2026-04-29T01:00:00Z",
};

const COMPLETE_JOB = {
  ...QUEUED_JOB,
  id: "mt-complete",
  status: "complete",
  training_signature: "sig-c",
  chosen_device: "mps",
  fallback_reason: null,
  final_train_loss: 0.12,
  final_val_loss: 0.18,
  total_epochs: 8,
  total_sequences: 2,
  total_chunks: 12,
  job_dir: "/tmp/data/masked_transformer_jobs/mt-complete",
};

const COMPLETE_JOB_DETAIL = {
  job: COMPLETE_JOB,
  region_detection_job_id: "rdj-mt-1",
  region_start_timestamp: 100.0,
  region_end_timestamp: 600.0,
  tier_composition: null,
  source_kind: "region_crnn",
};

const LOSS_CURVE = {
  epochs: [1, 2, 3, 4, 5],
  train_loss: [0.5, 0.4, 0.3, 0.25, 0.22],
  val_loss: [0.55, 0.45, 0.35, 0.3, 0.28],
  val_metrics: { final_val_loss: 0.28 },
};

const TOKENS = {
  total: 4,
  offset: 0,
  limit: 5000,
  items: [
    {
      sequence_id: "r1",
      position: 0,
      label: 5,
      confidence: 0.7,
      start_timestamp: 100.0,
      end_timestamp: 100.25,
      tier: "event_core",
      audio_file_id: 1,
    },
    {
      sequence_id: "r1",
      position: 1,
      label: 7,
      confidence: 0.5,
      start_timestamp: 100.25,
      end_timestamp: 100.5,
      tier: "background",
      audio_file_id: 1,
    },
    {
      sequence_id: "r2",
      position: 0,
      label: 5,
      confidence: 0.6,
      start_timestamp: 200.0,
      end_timestamp: 200.25,
      tier: "event_core",
      audio_file_id: 2,
    },
    {
      sequence_id: "r2",
      position: 1,
      label: 8,
      confidence: 0.4,
      start_timestamp: 200.25,
      end_timestamp: 200.5,
      tier: "background",
      audio_file_id: 2,
    },
  ],
};

const RUN_LENGTHS = { k: 100, tau: 1.5, run_lengths: { "5": [1, 1], "7": [1], "8": [1] } };

const OVERLAY = {
  total: 4,
  items: [
    {
      sequence_id: "r1",
      position_in_sequence: 0,
      start_timestamp: 100.0,
      end_timestamp: 100.25,
      pca_x: 0.1,
      pca_y: 0.2,
      umap_x: 1.0,
      umap_y: 2.0,
      viterbi_state: 5,
      max_state_probability: 0.7,
    },
    {
      sequence_id: "r1",
      position_in_sequence: 1,
      start_timestamp: 100.25,
      end_timestamp: 100.5,
      pca_x: -0.1,
      pca_y: -0.2,
      umap_x: -1.0,
      umap_y: -2.0,
      viterbi_state: 7,
      max_state_probability: 0.5,
    },
  ],
};

const EXEMPLARS = {
  n_states: 100,
  states: {
    "5": [
      {
        sequence_id: "r1",
        position_in_sequence: 0,
        audio_file_id: 1,
        start_timestamp: 100.0,
        end_timestamp: 100.25,
        max_state_probability: 0.7,
        exemplar_type: "high_confidence",
        extras: { tier: "event_core" },
      },
    ],
    "7": [
      {
        sequence_id: "r1",
        position_in_sequence: 1,
        audio_file_id: 1,
        start_timestamp: 100.25,
        end_timestamp: 100.5,
        max_state_probability: 0.5,
        exemplar_type: "high_confidence",
        extras: { tier: "background" },
      },
    ],
  },
};

const LABEL_DIST = {
  n_states: 100,
  total_windows: 4,
  states: {
    "5": { event_core: { song: 2 }, background: {}, near_event: {} },
    "7": { event_core: {}, background: { call: 1 }, near_event: {} },
  },
};

const MOTIF_JOB_HMM = {
  id: "motif-hmm",
  status: "complete",
  parent_kind: "hmm",
  hmm_sequence_job_id: "hmm-x",
  masked_transformer_job_id: null,
  k: null,
  source_kind: "region_crnn",
  min_ngram: 2,
  max_ngram: 8,
  minimum_occurrences: 5,
  minimum_event_sources: 2,
  frequency_weight: 0.4,
  event_source_weight: 0.3,
  event_core_weight: 0.2,
  low_background_weight: 0.1,
  call_probability_weight: null,
  config_signature: "sig-1",
  total_groups: 2,
  total_collapsed_tokens: 8,
  total_candidate_occurrences: 4,
  total_motifs: 0,
  artifact_dir: "/tmp/data/motif_extractions/motif-hmm",
  error_message: null,
  created_at: "2026-04-29T02:00:00Z",
  updated_at: "2026-04-29T02:01:00Z",
};

interface MockState {
  jobs: typeof COMPLETE_JOB[];
  motifJobs?: typeof MOTIF_JOB_HMM[];
  capturedMotifListUrls?: string[];
  capturedMotifCreates?: Array<Record<string, unknown>>;
}

async function setupMocks(page: Page, state: MockState): Promise<void> {
  await page.route("**/sequence-models/continuous-embeddings**", (route: Route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([CEJ_CRNN_COMPLETE, CEJ_SURFPERCH_COMPLETE]),
      });
    }
    return route.fulfill({ status: 405 });
  });

  await page.route(
    "**/sequence-models/masked-transformers",
    (route: Route) => {
      const method = route.request().method();
      if (method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(state.jobs),
        });
      }
      if (method === "POST") {
        const body = route.request().postDataJSON();
        const created = {
          ...QUEUED_JOB,
          id: "mt-created",
          continuous_embedding_job_id: body.continuous_embedding_job_id,
          preset: body.preset ?? "default",
          k_values: body.k_values ?? [100],
        };
        state.jobs = [created, ...state.jobs];
        return route.fulfill({
          status: 201,
          contentType: "application/json",
          body: JSON.stringify(created),
        });
      }
      return route.fulfill({ status: 405 });
    },
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}`,
    (route: Route) => {
      const method = route.request().method();
      if (method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(COMPLETE_JOB_DETAIL),
        });
      }
      return route.fulfill({ status: 405 });
    },
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/loss-curve**`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(LOSS_CURVE),
      }),
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/tokens**`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(TOKENS),
      }),
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/run-lengths**`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(RUN_LENGTHS),
      }),
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/overlay**`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(OVERLAY),
      }),
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/exemplars**`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(EXEMPLARS),
      }),
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/label-distribution**`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(LABEL_DIST),
      }),
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/reconstruction-error**`,
    (route) =>
      route.fulfill({
        status: 404,
        contentType: "application/json",
        body: JSON.stringify({ detail: "not found" }),
      }),
  );

  await page.route("**/sequence-models/motif-extractions**", async (route: Route) => {
    const method = route.request().method();
    if (method === "GET") {
      state.capturedMotifListUrls?.push(route.request().url());
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(state.motifJobs ?? []),
      });
    }
    if (method === "POST") {
      const body = route.request().postDataJSON();
      state.capturedMotifCreates?.push(body);
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          ...MOTIF_JOB_HMM,
          id: "motif-mt",
          parent_kind: body.parent_kind,
          hmm_sequence_job_id: body.hmm_sequence_job_id ?? null,
          masked_transformer_job_id: body.masked_transformer_job_id ?? null,
          k: body.k ?? null,
        }),
      });
    }
    return route.fulfill({ status: 405 });
  });

  // Stub timeline tile + audio routes to keep the detail page quiet.
  await page.route("**/call-parsing/region-jobs/*/tile**", (route) => {
    const pixel = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "base64",
    );
    return route.fulfill({ status: 200, contentType: "image/png", body: pixel });
  });
  await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "audio/mpeg",
      body: Buffer.alloc(0),
    }),
  );
}

test.describe("Sequence Models — Masked Transformer", () => {
  test("nav reaches the Masked Transformer jobs page", async ({ page }) => {
    const state: MockState = { jobs: [QUEUED_JOB, COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/masked-transformer");
    await expect(page.getByTestId("masked-transformer-jobs-page")).toBeVisible();
    await expect(page.getByText("Active Jobs")).toBeVisible();
    await expect(page.getByText("Previous Jobs")).toBeVisible();
  });

  test("create form filters to CRNN-source completed CE jobs", async ({ page }) => {
    const state: MockState = { jobs: [] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/masked-transformer");

    const select = page.getByTestId("mt-source-select");
    await expect(select).toBeVisible();
    const options = select.locator("option");
    // Placeholder + the single CRNN-source completed CEJ.
    await expect(options).toHaveCount(2);
    await expect(options.nth(1)).toContainText(CEJ_CRNN_COMPLETE.id.slice(0, 8));
  });

  test("create form parses k_values CSV and submits", async ({ page }) => {
    const state: MockState = { jobs: [] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/masked-transformer");

    await page
      .getByTestId("mt-source-select")
      .selectOption(CEJ_CRNN_COMPLETE.id);
    await page.getByTestId("mt-k-values").fill("50, 100");
    await page.getByTestId("mt-preset-small").locator("input").check();
    await page.getByTestId("mt-create-submit").click();
    await expect(page).toHaveURL(/\/masked-transformer\/mt-created/);
  });

  test("create form validation rejects k below 2", async ({ page }) => {
    const state: MockState = { jobs: [] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/masked-transformer");
    await page
      .getByTestId("mt-source-select")
      .selectOption(CEJ_CRNN_COMPLETE.id);
    await page.getByTestId("mt-k-values").fill("1, 5");
    const submit = page.getByTestId("mt-create-submit");
    await expect(submit).toBeDisabled();
  });

  test("jobs table displays k_values + device badge + actions", async ({ page }) => {
    const state: MockState = { jobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/masked-transformer");

    await expect(
      page.getByTestId(`mt-job-k-values-${COMPLETE_JOB.id}`),
    ).toContainText("50, 100");
    await expect(page.getByTestId(`mt-job-row-${COMPLETE_JOB.id}`)).toContainText(
      "mps",
    );
    await expect(page.getByTestId(`mt-job-open-${COMPLETE_JOB.id}`)).toBeVisible();
  });

  test("detail page renders header, loss curve, timeline, exemplars, label distribution", async ({ page }) => {
    const state: MockState = { jobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("masked-transformer-detail-page")).toBeVisible();
    await expect(page.getByTestId("mt-detail-device-badge")).toContainText("mps");
    await expect(page.getByTestId("k-picker")).toBeVisible();
    await expect(page.getByTestId("loss-curve-chart")).toBeVisible();
    await expect(page.getByTestId("token-run-length-histograms")).toBeVisible();
    await expect(page.getByTestId("mt-exemplar-gallery")).toBeVisible();
    await expect(page.getByTestId("mt-label-distribution")).toBeVisible();
    await expect(page.getByTestId("motif-extraction-panel")).toBeVisible();
  });

  test("k-picker switches the URL search-param without remount", async ({ page }) => {
    const state: MockState = { jobs: [COMPLETE_JOB] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    // Default = first k.
    await expect(
      page.getByTestId("k-picker-tab-50"),
    ).toHaveAttribute("aria-selected", "true");

    await page.getByTestId("k-picker-tab-100").click();
    await expect(page).toHaveURL(/[?&]k=100/);
    await expect(
      page.getByTestId("k-picker-tab-100"),
    ).toHaveAttribute("aria-selected", "true");
  });

  test("motif panel pre-fills parent_kind=masked_transformer + k", async ({ page }) => {
    const state: MockState = {
      jobs: [COMPLETE_JOB],
      motifJobs: [],
      capturedMotifListUrls: [],
      capturedMotifCreates: [],
    };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await page.getByTestId("k-picker-tab-100").click();
    await page.getByTestId("motif-create-submit").click();

    expect(state.capturedMotifCreates?.length).toBeGreaterThanOrEqual(1);
    const body = state.capturedMotifCreates?.[0] as
      | Record<string, unknown>
      | undefined;
    expect(body?.parent_kind).toBe("masked_transformer");
    expect(body?.masked_transformer_job_id).toBe(COMPLETE_JOB.id);
    expect(body?.k).toBe(100);
    expect(
      state.capturedMotifListUrls?.some((url) => {
        const parsed = new URL(url);
        return (
          parsed.searchParams.get("parent_kind") === "masked_transformer" &&
          parsed.searchParams.get("masked_transformer_job_id") === COMPLETE_JOB.id &&
          parsed.searchParams.get("k") === "100"
        );
      }),
    ).toBe(true);
  });
});
