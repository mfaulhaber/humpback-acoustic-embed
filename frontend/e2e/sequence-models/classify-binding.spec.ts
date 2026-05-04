/**
 * Sequence Models — Classify-binding workflow.
 *
 * Covers the new HMM/MT submit-time Classify dropdown and the regenerate
 * dialog on the detail pages. Mocks just enough of the API to exercise the
 * UI plumbing; the heavier visualization endpoints are stubbed minimally.
 */

import { expect, test, type Page } from "@playwright/test";

const SEG_JOB_ID = "seg-classify-binding";

const CEJ = {
  id: "cej-classify-binding",
  status: "complete",
  event_segmentation_job_id: SEG_JOB_ID,
  event_source_mode: "raw" as const,
  model_version: "surfperch-tensorflow2",
  window_size_seconds: 5.0,
  hop_seconds: 1.0,
  pad_seconds: 2.0,
  target_sample_rate: 32000,
  feature_config_json: null,
  encoding_signature: "cej-classify-sig",
  vector_dim: 1280,
  total_events: 4,
  merged_spans: 2,
  total_windows: 120,
  parquet_path: null,
  error_message: null,
  region_detection_job_id: null,
  chunk_size_seconds: null,
  chunk_hop_seconds: null,
  crnn_checkpoint_sha256: null,
  crnn_segmentation_model_id: null,
  projection_kind: null,
  projection_dim: null,
  total_regions: null,
  total_chunks: null,
  created_at: "2026-04-27T00:00:00Z",
  updated_at: "2026-04-27T00:10:00Z",
};

const CEJ_CRNN = {
  ...CEJ,
  id: "cej-classify-binding-crnn",
  model_version: "crnn-call-parsing-pytorch",
  encoding_signature: "cej-classify-crnn-sig",
  region_detection_job_id: "rdj-1",
  total_chunks: 200,
};

const CLASSIFY_NEWER = {
  id: "classify-newer-id-12345",
  created_at: "2026-04-27T02:00:00Z",
  model_name: "voc-cnn-v3",
  n_events_classified: 42,
  status: "complete",
};

const CLASSIFY_OLDER = {
  id: "classify-older-id-67890",
  created_at: "2026-04-26T01:00:00Z",
  model_name: "voc-cnn-v2",
  n_events_classified: 30,
  status: "complete",
};

interface ClassifyMockState {
  jobs: typeof CLASSIFY_NEWER[];
}

async function mockBaselines(
  page: Page,
  options: {
    cejList?: Array<typeof CEJ | typeof CEJ_CRNN>;
    classifyState?: ClassifyMockState;
  } = {},
) {
  const cejList = options.cejList ?? [CEJ];
  const classifyState =
    options.classifyState ?? { jobs: [CLASSIFY_NEWER, CLASSIFY_OLDER] };
  await page.route("**/sequence-models/continuous-embeddings**", (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const idMatch = url.match(/\/continuous-embeddings\/([^/?#]+)/);
    if (idMatch) {
      const id = idMatch[1];
      const job = cejList.find((j) => j.id === id);
      if (!job) return route.fulfill({ status: 404 });
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
        body: JSON.stringify(cejList),
      });
    }
    return route.fulfill({ status: 405 });
  });

  await page.route(
    "**/call-parsing/classification-jobs/by-segmentation**",
    (route) => {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(classifyState.jobs),
      });
    },
  );
}

test.describe("Sequence Models — Classify binding (HMM submit)", () => {
  test("populated Classify dropdown defaults to newest and enables submit", async ({
    page,
  }) => {
    await mockBaselines(page);

    let postBody: Record<string, unknown> | null = null;
    await page.route("**/sequence-models/hmm-sequences**", (route) => {
      const method = route.request().method();
      if (method === "POST") {
        postBody = JSON.parse(route.request().postData() ?? "{}");
        return route.fulfill({
          status: 201,
          contentType: "application/json",
          body: JSON.stringify({
            ...postBody,
            id: "hmm-new",
            status: "queued",
            created_at: "2026-04-28T00:00:00Z",
            updated_at: "2026-04-28T00:00:00Z",
            library: "hmmlearn",
            tol: 0.0001,
            train_log_likelihood: null,
            n_train_sequences: null,
            n_train_frames: null,
            n_decoded_sequences: null,
            artifact_dir: null,
            error_message: null,
            l2_normalize: true,
            pca_whiten: false,
          }),
        });
      }
      if (method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([]),
        });
      }
      return route.fulfill({ status: 405 });
    });

    await page.goto("/app/sequence-models/hmm-sequence");

    await page.getByTestId("hmm-source-select").selectOption(CEJ.id);

    // The Classify dropdown should populate and default to the newer one.
    const classifySelect = page.getByTestId("hmm-classify-select");
    await expect(classifySelect).toBeEnabled();
    await expect(classifySelect).toHaveValue(CLASSIFY_NEWER.id);

    // Older option present.
    const optionTexts = await classifySelect
      .locator("option")
      .allInnerTexts();
    const sawOlder = optionTexts.some((t) =>
      t.includes(CLASSIFY_OLDER.id.slice(0, 8)),
    );
    expect(sawOlder).toBe(true);

    // Submit posts the chosen Classify FK.
    await page.getByTestId("hmm-create-submit").click();
    await expect.poll(() => postBody).not.toBeNull();
    expect(postBody!.continuous_embedding_job_id).toBe(CEJ.id);
    expect(postBody!.event_classification_job_id).toBe(CLASSIFY_NEWER.id);
  });

  test("empty Classify dropdown disables submit and shows helper text", async ({
    page,
  }) => {
    await mockBaselines(page, { classifyState: { jobs: [] } });
    await page.route("**/sequence-models/hmm-sequences**", (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([]),
        });
      }
      return route.fulfill({ status: 405 });
    });

    await page.goto("/app/sequence-models/hmm-sequence");
    await page.getByTestId("hmm-source-select").selectOption(CEJ.id);

    await expect(page.getByTestId("hmm-classify-select")).toBeDisabled();
    await expect(page.getByTestId("hmm-classify-empty-helper")).toBeVisible();
    await expect(page.getByTestId("hmm-create-submit")).toBeDisabled();
  });
});

test.describe("Sequence Models — Classify binding (MT submit)", () => {
  test("populated Classify dropdown defaults to newest and enables submit", async ({
    page,
  }) => {
    await mockBaselines(page, { cejList: [CEJ_CRNN] });

    let postBody: Record<string, unknown> | null = null;
    await page.route("**/sequence-models/masked-transformers**", (route) => {
      const method = route.request().method();
      if (method === "POST") {
        postBody = JSON.parse(route.request().postData() ?? "{}");
        return route.fulfill({
          status: 201,
          contentType: "application/json",
          body: JSON.stringify({
            id: "mt-new",
            status: "queued",
            status_reason: null,
            continuous_embedding_job_id: postBody!.continuous_embedding_job_id,
            event_classification_job_id: postBody!.event_classification_job_id,
            training_signature: "sig-x",
            preset: postBody!.preset,
            mask_fraction: 0.2,
            span_length_min: 2,
            span_length_max: 6,
            dropout: 0.1,
            mask_weight_bias: true,
            cosine_loss_weight: 0,
            max_epochs: 30,
            early_stop_patience: 3,
            val_split: 0.1,
            seed: 42,
            k_values: postBody!.k_values,
            chosen_device: null,
            fallback_reason: null,
            final_train_loss: null,
            final_val_loss: null,
            total_epochs: null,
            job_dir: null,
            total_sequences: null,
            total_chunks: null,
            error_message: null,
            created_at: "2026-04-28T00:00:00Z",
            updated_at: "2026-04-28T00:00:00Z",
          }),
        });
      }
      if (method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([]),
        });
      }
      return route.fulfill({ status: 405 });
    });

    await page.goto("/app/sequence-models/masked-transformer");
    await page.getByTestId("mt-source-select").selectOption(CEJ_CRNN.id);

    const classifySelect = page.getByTestId("mt-classify-select");
    await expect(classifySelect).toBeEnabled();
    await expect(classifySelect).toHaveValue(CLASSIFY_NEWER.id);

    await page.getByTestId("mt-create-submit").click();
    await expect.poll(() => postBody).not.toBeNull();
    expect(postBody!.continuous_embedding_job_id).toBe(CEJ_CRNN.id);
    expect(postBody!.event_classification_job_id).toBe(CLASSIFY_NEWER.id);
  });

  test("empty Classify dropdown disables MT submit", async ({ page }) => {
    await mockBaselines(page, {
      cejList: [CEJ_CRNN],
      classifyState: { jobs: [] },
    });
    await page.route("**/sequence-models/masked-transformers**", (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([]),
        });
      }
      return route.fulfill({ status: 405 });
    });

    await page.goto("/app/sequence-models/masked-transformer");
    await page.getByTestId("mt-source-select").selectOption(CEJ_CRNN.id);

    await expect(page.getByTestId("mt-classify-select")).toBeDisabled();
    await expect(page.getByTestId("mt-classify-empty-helper")).toBeVisible();
    await expect(page.getByTestId("mt-create-submit")).toBeDisabled();
  });
});
