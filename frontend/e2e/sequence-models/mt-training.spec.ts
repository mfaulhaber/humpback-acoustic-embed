import { expect, test, type Page, type Route } from "@playwright/test";

const CEJ_ONE = {
  id: "cej-training-one",
  status: "complete",
  event_segmentation_job_id: "seg-training-one",
  event_source_mode: "raw",
  model_version: "crnn-call-parsing-pytorch",
  window_size_seconds: null,
  hop_seconds: null,
  pad_seconds: null,
  target_sample_rate: 32000,
  feature_config_json: null,
  encoding_signature: "sig-cej-one",
  vector_dim: 64,
  total_events: null,
  merged_spans: null,
  total_windows: null,
  parquet_path: "/tmp/cej-one.parquet",
  error_message: null,
  region_detection_job_id: "rdj-one",
  chunk_size_seconds: 0.25,
  chunk_hop_seconds: 0.25,
  crnn_checkpoint_sha256: "ckpt-shared",
  crnn_segmentation_model_id: "seg-model-one",
  projection_kind: "identity",
  projection_dim: 64,
  total_regions: 2,
  total_chunks: 12,
  created_at: "2026-05-06T12:00:00Z",
  updated_at: "2026-05-06T12:05:00Z",
};

const CEJ_TWO = {
  ...CEJ_ONE,
  id: "cej-training-two",
  event_segmentation_job_id: "seg-training-two",
  encoding_signature: "sig-cej-two",
  region_detection_job_id: "rdj-two",
  crnn_segmentation_model_id: "seg-model-two",
  total_chunks: 24,
};

const CLASSIFY_ONE = {
  id: "cls-training-one",
  status: "complete",
  event_segmentation_job_id: "seg-training-one",
  model_name: "north-labels",
  n_events_classified: 12,
  artifact_dir: "/tmp/classify-one",
  error_message: null,
  created_at: "2026-05-06T12:10:00Z",
  updated_at: "2026-05-06T12:11:00Z",
};

const CLASSIFY_TWO = {
  ...CLASSIFY_ONE,
  id: "cls-training-two",
  event_segmentation_job_id: "seg-training-two",
  model_name: "south-labels",
  n_events_classified: 24,
};

const COMPLETE_JOB = {
  id: "mt-training-complete",
  status: "complete",
  status_reason: null,
  continuous_embedding_job_id: CEJ_ONE.id,
  event_classification_job_id: CLASSIFY_ONE.id,
  source_count: 2,
  training_signature: "mt-training-sig",
  preset: "default",
  mask_fraction: 0.2,
  span_length_min: 2,
  span_length_max: 6,
  dropout: 0.1,
  mask_weight_bias: true,
  cosine_loss_weight: 0.0,
  batch_size: 8,
  retrieval_head_enabled: true,
  retrieval_dim: 128,
  retrieval_hidden_dim: 512,
  retrieval_l2_normalize: true,
  retrieval_head_arch: "mlp",
  sequence_construction_mode: "region",
  event_centered_fraction: 0.0,
  pre_event_context_sec: null,
  post_event_context_sec: null,
  contrastive_loss_weight: 0.0,
  contrastive_temperature: 0.07,
  contrastive_label_source: "none",
  contrastive_min_events_per_label: 4,
  contrastive_min_regions_per_label: 2,
  require_cross_region_positive: true,
  related_label_policy_json: null,
  contrastive_sampler_enabled: true,
  contrastive_labels_per_batch: 4,
  contrastive_events_per_label: 4,
  contrastive_max_unlabeled_fraction: 0.25,
  contrastive_region_balance: true,
  training_freeze_mode: "none",
  source_masked_transformer_job_id: null,
  negative_label_family_policy_json: null,
  max_epochs: 30,
  early_stop_patience: 3,
  val_split: 0.1,
  seed: 42,
  k_values: [50, 100],
  chosen_device: "mps",
  fallback_reason: null,
  final_train_loss: 0.12,
  final_val_loss: 0.18,
  total_epochs: 8,
  job_dir: "/tmp/mt-training-complete",
  total_sequences: 3,
  total_chunks: 36,
  error_message: null,
  created_at: "2026-05-06T12:20:00Z",
  updated_at: "2026-05-06T12:30:00Z",
};

const SOURCES = [
  {
    id: "source-one",
    masked_transformer_job_id: COMPLETE_JOB.id,
    source_order: 0,
    continuous_embedding_job_id: CEJ_ONE.id,
    event_classification_job_id: CLASSIFY_ONE.id,
    source_alias: "north",
    created_at: "2026-05-06T12:20:00Z",
    updated_at: "2026-05-06T12:20:00Z",
  },
  {
    id: "source-two",
    masked_transformer_job_id: COMPLETE_JOB.id,
    source_order: 1,
    continuous_embedding_job_id: CEJ_TWO.id,
    event_classification_job_id: CLASSIFY_TWO.id,
    source_alias: "south",
    created_at: "2026-05-06T12:20:00Z",
    updated_at: "2026-05-06T12:20:00Z",
  },
];

const COMPLETE_DETAIL = {
  job: COMPLETE_JOB,
  sources: SOURCES,
  region_detection_job_id: null,
  region_start_timestamp: null,
  region_end_timestamp: null,
  tier_composition: null,
  source_kind: "region_crnn",
};

const LOSS_CURVE = {
  epochs: [1, 2],
  train_loss: [0.5, 0.25],
  val_loss: [0.55, 0.3],
  val_metrics: { final_val_loss: 0.3 },
};

const RUN_LENGTHS = {
  k: 100,
  tau: 1.5,
  run_lengths: { "4": [1, 2, 3], "8": [2] },
};

const OVERLAY = {
  total: 1,
  items: [
    {
      sequence_id: "0:region-one",
      position_in_sequence: 0,
      start_timestamp: 10.0,
      end_timestamp: 10.25,
      pca_x: 0.1,
      pca_y: 0.2,
      umap_x: 1.0,
      umap_y: 2.0,
      viterbi_state: 4,
      max_state_probability: 0.9,
    },
  ],
};

const EXEMPLARS = {
  n_states: 100,
  states: {
    "4": [
      {
        sequence_id: "0:region-one",
        position_in_sequence: 0,
        audio_file_id: 1,
        start_timestamp: 10.0,
        end_timestamp: 10.25,
        max_state_probability: 0.9,
        exemplar_type: "high_confidence",
        extras: { tier: "event_core", event_types: ["Moan"] },
      },
    ],
  },
};

const ANALYSIS_REPORT = {
  job: { job_id: COMPLETE_JOB.id, k: 100 },
  options: { include_geometry_report: true },
  artifacts: { contextual_path: "/tmp/contextual.parquet" },
  label_coverage: { labeled_query_count: 10 },
  results: {
    unrestricted: {
      raw_l2: { same_human_label: 0.7, adjacent_1s: 0.1 },
    },
  },
  event_level_results: {
    unrestricted: {
      raw_l2: { exact_human_label_set: 0.6 },
    },
  },
  representative_good_queries: [
    {
      query_idx: 1,
      query_human_types: "Moan",
      verdict: "good",
      same_human_label_rate: 0.7,
      adjacent_1s_rate: 0.1,
    },
  ],
  representative_risky_queries: [],
  query_rows: [],
  neighbor_rows: [],
  geometry_report: {
    spaces: {
      "contextual.raw_l2": {
        available: true,
        mean_vector_band: "healthy",
        effective_rank_band: "plausible",
        warnings: [],
      },
      "retrieval.raw_l2": {
        available: false,
        reason: "retrieval_artifact_unavailable",
        mean_vector_band: null,
        effective_rank_band: null,
        warnings: [],
      },
    },
    summary: { lambda_sweeps_blocked: false },
  },
};

interface MockState {
  jobs: Array<Record<string, unknown>>;
  capturedCreates: Array<Record<string, unknown>>;
  capturedAnalysis: Array<Record<string, unknown>>;
  latestReport: Record<string, unknown> | null;
}

async function setupMocks(page: Page, state: MockState): Promise<void> {
  await page.route("**/sequence-models/continuous-embeddings**", (route: Route) => {
    if (route.request().method() !== "GET") return route.fulfill({ status: 405 });
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([CEJ_ONE, CEJ_TWO]),
    });
  });

  await page.route(
    "**/call-parsing/classification-jobs/by-segmentation**",
    (route: Route) => {
      const url = new URL(route.request().url());
      const segmentationId = url.searchParams.get("event_segmentation_job_id");
      const jobs =
        segmentationId === "seg-training-two" ? [CLASSIFY_TWO] : [CLASSIFY_ONE];
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(jobs),
      });
    },
  );

  await page.route("**/sequence-models/masked-transformers", (route: Route) => {
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
      state.capturedCreates.push(body);
      const created = {
        ...COMPLETE_JOB,
        id: "mt-created",
        status: "queued",
        chosen_device: null,
        source_count: body.sources?.length ?? 1,
        continuous_embedding_job_id:
          body.sources?.[0]?.continuous_embedding_job_id ?? CEJ_ONE.id,
        event_classification_job_id:
          body.sources?.[0]?.event_classification_job_id ?? CLASSIFY_ONE.id,
        k_values: body.k_values ?? [100],
        total_chunks: null,
      };
      state.jobs = [created, ...state.jobs];
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(created),
      });
    }
    return route.fulfill({ status: 405 });
  });

  await page.route("**/sequence-models/masked-transformers/mt-created", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ...COMPLETE_DETAIL,
        job: state.jobs.find((job) => job.id === "mt-created"),
        sources: SOURCES.map((source, index) => ({
          ...source,
          masked_transformer_job_id: "mt-created",
          continuous_embedding_job_id:
            index === 0 ? CEJ_ONE.id : CEJ_TWO.id,
          event_classification_job_id:
            index === 0 ? CLASSIFY_ONE.id : CLASSIFY_TWO.id,
        })),
      }),
    }),
  );

  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(COMPLETE_DETAIL),
      }),
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
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/nearest-neighbor-report`,
    (route) => {
      state.capturedAnalysis.push(route.request().postDataJSON());
      state.latestReport = ANALYSIS_REPORT;
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(ANALYSIS_REPORT),
      });
    },
  );
  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}/nearest-neighbor-report/latest`,
    (route) =>
      route.fulfill({
        status: state.latestReport ? 200 : 404,
        contentType: "application/json",
        body: JSON.stringify(state.latestReport ?? { detail: "not found" }),
      }),
  );
}

test.describe("Sequence Models - MT Training", () => {
  test("nav and breadcrumbs expose MT Training and MT Motif routes", async ({ page }) => {
    const state: MockState = {
      jobs: [COMPLETE_JOB],
      capturedCreates: [],
      capturedAnalysis: [],
      latestReport: null,
    };
    await setupMocks(page, state);

    await page.goto("/app/sequence-models/mt-training");

    await expect(page.getByTestId("mt-training-jobs-page")).toBeVisible();
    await expect(page.getByText("MT Training").first()).toBeVisible();
    await expect(page.getByText("MT Motif").first()).toBeVisible();
    await expect(page.getByTestId(`mt-training-source-count-${COMPLETE_JOB.id}`)).toContainText("2");
  });

  test("legacy masked-transformer route redirects to MT Motif", async ({ page }) => {
    const state: MockState = {
      jobs: [COMPLETE_JOB],
      capturedCreates: [],
      capturedAnalysis: [],
      latestReport: null,
    };
    await setupMocks(page, state);

    await page.goto("/app/sequence-models/masked-transformer");

    await expect(page).toHaveURL(/\/app\/sequence-models\/mt-motif$/);
    await expect(page.getByTestId("masked-transformer-jobs-page")).toBeVisible();
  });

  test("create form submits multiple source pairs with contrastive disabled", async ({ page }) => {
    const state: MockState = {
      jobs: [],
      capturedCreates: [],
      capturedAnalysis: [],
      latestReport: null,
    };
    await setupMocks(page, state);
    await page.goto("/app/sequence-models/mt-training");

    await page.getByTestId("mt-training-source-0").selectOption(CEJ_ONE.id);
    await expect(page.getByTestId("mt-training-classify-0")).toHaveValue(
      CLASSIFY_ONE.id,
    );
    await page.getByTestId("mt-training-add-source").click();
    await page.getByTestId("mt-training-source-1").selectOption(CEJ_TWO.id);
    await expect(page.getByTestId("mt-training-classify-1")).toHaveValue(
      CLASSIFY_TWO.id,
    );
    await expect(page.getByTestId("mt-contrastive-enabled")).toHaveCount(0);
    await page.getByTestId("mt-training-submit").click();

    await expect(page).toHaveURL(/\/app\/sequence-models\/mt-training\/mt-created$/);
    expect(state.capturedCreates[0]).toMatchObject({
      sources: [
        {
          continuous_embedding_job_id: CEJ_ONE.id,
          event_classification_job_id: CLASSIFY_ONE.id,
          source_alias: null,
        },
        {
          continuous_embedding_job_id: CEJ_TWO.id,
          event_classification_job_id: CLASSIFY_TWO.id,
          source_alias: null,
        },
      ],
      contrastive_loss_weight: 0,
      contrastive_label_source: "none",
      training_freeze_mode: "none",
    });
  });

  test("detail Analyze button posts full report options and opens report tables", async ({ page }) => {
    const state: MockState = {
      jobs: [COMPLETE_JOB],
      capturedCreates: [],
      capturedAnalysis: [],
      latestReport: null,
    };
    await setupMocks(page, state);

    await page.goto(`/app/sequence-models/mt-training/${COMPLETE_JOB.id}?k=100`);

    await expect(page.getByTestId("mt-training-detail-page")).toBeVisible();
    await expect(page.getByTestId("mt-training-source-table")).toContainText(CEJ_TWO.id);
    await expect(page.getByTestId("mt-timeline-viewer")).toHaveCount(0);
    await expect(page.getByTestId("mt-label-distribution")).toHaveCount(0);
    await expect(page.getByTestId("motif-extraction-panel")).toHaveCount(0);

    await page.getByTestId("mt-training-analysis-button").click();

    await expect(page).toHaveURL(
      new RegExp(`/app/sequence-models/mt-training/${COMPLETE_JOB.id}/analysis`),
    );
    expect(state.capturedAnalysis[0]).toMatchObject({
      k: 100,
      include_event_level: true,
      include_geometry_report: true,
      include_query_rows: true,
      include_neighbor_rows: false,
      retrieval_modes: [
        "unrestricted",
        "exclude_same_event",
        "exclude_same_event_and_region",
      ],
    });
    await expect(page.getByTestId("mt-analysis-summary-panel")).toBeVisible();
    await expect(page.getByText("Recommended Spaces")).toBeVisible();
    await expect(page.getByText("Label Coverage")).toBeVisible();
    await expect(page.getByText("Aggregate Retrieval Metrics")).toBeVisible();
    await expect(page.getByText("Geometry Diagnostics")).toBeVisible();
  });
});
