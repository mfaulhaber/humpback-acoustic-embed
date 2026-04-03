import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";

/**
 * Tests for the classifier Training tab UI changes:
 * - Positive and Negative embedding set panels with select-all inside panel
 * - Always-visible (disabled) delete buttons on Training Jobs and Trained Models
 * - Training job API accepts negative_embedding_set_ids
 */

const E2E_DIR = dirname(fileURLToPath(import.meta.url));
const AUTORESEARCH_FIXTURE_DIR = resolve(
  E2E_DIR,
  "../../scripts/autoresearch/output/explicit-negatives",
);
const VENDORED_MANIFEST = JSON.parse(
  readFileSync(resolve(AUTORESEARCH_FIXTURE_DIR, "manifest.json"), "utf8"),
) as Record<string, unknown>;
const VENDORED_PHASE1_BEST_RUN = JSON.parse(
  readFileSync(
    resolve(AUTORESEARCH_FIXTURE_DIR, "phase1/best_run.json"),
    "utf8",
  ),
) as Record<string, unknown>;
const VENDORED_PHASE1_COMPARISON = JSON.parse(
  readFileSync(
    resolve(AUTORESEARCH_FIXTURE_DIR, "phase1/lr-v12-comparison.json"),
    "utf8",
  ),
) as Record<string, unknown>;
const VENDORED_PHASE1_TOP_FALSE_POSITIVES = JSON.parse(
  readFileSync(
    resolve(AUTORESEARCH_FIXTURE_DIR, "phase1/top_false_positives.json"),
    "utf8",
  ),
) as unknown[];

test.describe("Classifier Training tab", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/app/classifier/training");
  });

  test("renders positive and negative embedding set panels", async ({
    page,
  }) => {
    // Both panels should be inside the Train Binary Classifier card
    const card = page.locator("text=Train Binary Classifier").locator("..");

    // Look for the panel labels inside bordered panels
    const posLabel = page.locator("text=Positive Embedding Sets");
    const negLabel = page.locator("text=Negative Embedding Sets");

    await expect(posLabel).toBeVisible();
    await expect(negLabel).toBeVisible();
  });

  test("select-all checkbox is inside each panel", async ({ page }) => {
    // Each panel is a bordered div containing a checkbox + label on the first row
    // The positive panel has a checkbox next to "Positive Embedding Sets"
    const panels = page.locator(".border.rounded.p-2");

    // Should have at least 2 panels (positive and negative)
    const panelCount = await panels.count();
    expect(panelCount).toBeGreaterThanOrEqual(2);

    // Each panel's first child row should contain a checkbox
    for (let i = 0; i < 2; i++) {
      const panel = panels.nth(i);
      const firstRow = panel.locator(".border-b").first();
      const checkbox = firstRow.locator('button[role="checkbox"]');
      await expect(checkbox).toBeVisible();
    }
  });

  test("Train Classifier button disabled when no selections", async ({
    page,
  }) => {
    const trainBtn = page.locator("button", { hasText: "Train Classifier" });
    await expect(trainBtn).toBeVisible();
    await expect(trainBtn).toBeDisabled();
  });

  test("Model Name input is present", async ({ page }) => {
    const nameInput = page.locator('input[placeholder="e.g. humpback-detector-v1"]');
    await expect(nameInput).toBeVisible();
  });

  test("no folder browser or folder path input present", async ({ page }) => {
    // The old negative audio folder input should be gone
    const folderInput = page.locator(
      'input[placeholder="/path/to/negative/audio"]',
    );
    await expect(folderInput).toHaveCount(0);

    // No FolderOpen icon button for negative folder
    const folderBrowseBtn = page.locator('button[title="Browse folders"]');
    await expect(folderBrowseBtn).toHaveCount(0);
  });
});

test.describe("Advanced Options", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/app/classifier/training");
  });

  test("advanced options collapsible is present and expands", async ({
    page,
  }) => {
    const trigger = page.locator("button", { hasText: "Advanced Options" });
    await expect(trigger).toBeVisible();

    // Click to expand
    await trigger.click();

    // Should show classifier type select
    await expect(
      page.locator("text=Classifier Type"),
    ).toBeVisible();

    // Should show L2 Normalize checkbox
    await expect(
      page.locator("text=L2 Normalize Embeddings"),
    ).toBeVisible();

    // Should show Class Weight select
    await expect(page.locator("text=Class Weight")).toBeVisible();
  });

  test("regularization C shown only for logistic regression", async ({
    page,
  }) => {
    const trigger = page.locator("button", { hasText: "Advanced Options" });
    await trigger.click();

    // Default is logistic regression — regularization should be visible
    await expect(
      page.locator("text=Regularization (C)"),
    ).toBeVisible();
  });
});

test.describe("Trained Models table columns", () => {
  test("table has Precision and F1 columns when models exist", async ({
    page,
  }) => {
    await page.goto("/app/classifier/training");

    const modelsHeading = page.locator("text=Trained Models");
    const hasModels = await modelsHeading
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasModels) {
      test.skip(true, "No trained models to verify table columns");
      return;
    }

    // Check for new column headers
    const table = modelsHeading.locator("..").locator("..").locator("table");
    await expect(table.locator("th", { hasText: "Precision" })).toBeVisible();
    await expect(table.locator("th", { hasText: "F1" })).toBeVisible();
  });
});

test.describe("Training API overlap validation", () => {
  test("rejects same embedding set in both positive and negative", async ({
    request,
  }) => {
    const resp = await request.post(
      "http://localhost:8000/classifier/training-jobs",
      {
        data: {
          name: "overlap-test",
          positive_embedding_set_ids: ["shared-id", "pos-only"],
          negative_embedding_set_ids: ["shared-id", "neg-only"],
        },
      },
    );
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.detail).toContain("both positive and negative");
  });
});

test.describe("Training Jobs delete button visibility", () => {
  test("delete button always visible but disabled when nothing selected", async ({
    page,
  }) => {
    await page.goto("/app/classifier/training");

    // Check for Training Jobs section
    const jobsHeading = page.locator("text=Training Jobs");
    const hasJobs = await jobsHeading
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasJobs) {
      test.skip(true, "No training jobs to test delete button visibility");
      return;
    }

    // Delete button should be visible even with nothing selected
    const deleteBtn = jobsHeading
      .locator("..")
      .locator("..")
      .locator("button", { hasText: /^Delete/ });
    await expect(deleteBtn).toBeVisible();
    await expect(deleteBtn).toBeDisabled();
  });
});

test.describe("Trained Models delete button visibility", () => {
  test("delete button always visible but disabled when nothing selected", async ({
    page,
  }) => {
    await page.goto("/app/classifier/training");

    // Check for Trained Models section
    const modelsHeading = page.locator("text=Trained Models");
    const hasModels = await modelsHeading
      .waitFor({ timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (!hasModels) {
      test.skip(true, "No trained models to test delete button visibility");
      return;
    }

    // Delete button should be visible even with nothing selected
    const deleteBtn = modelsHeading
      .locator("..")
      .locator("..")
      .locator("button", { hasText: /^Delete/ });
    await expect(deleteBtn).toBeVisible();
    await expect(deleteBtn).toBeDisabled();
  });
});

test.describe("Training job API", () => {
  test("create training job validates negative_embedding_set_ids", async ({
    request,
  }) => {
    // Should reject when negative embedding sets don't exist
    const resp = await request.post(
      "http://localhost:8000/classifier/training-jobs",
      {
        data: {
          name: "playwright-test",
          positive_embedding_set_ids: ["nonexistent-pos"],
          negative_embedding_set_ids: ["nonexistent-neg"],
        },
      },
    );
    expect(resp.status()).toBe(400);
  });

  test("create training job rejects empty negative_embedding_set_ids", async ({
    request,
  }) => {
    const resp = await request.post(
      "http://localhost:8000/classifier/training-jobs",
      {
        data: {
          name: "playwright-test",
          positive_embedding_set_ids: ["some-id"],
          negative_embedding_set_ids: [],
        },
      },
    );
    expect(resp.status()).toBe(400);
  });

  test("training job response includes negative_embedding_set_ids field", async ({
    request,
  }) => {
    // List existing training jobs and check response shape
    const resp = await request.get(
      "http://localhost:8000/classifier/training-jobs",
    );
    expect(resp.ok()).toBeTruthy();
    const jobs = await resp.json();

    if (jobs.length === 0) {
      test.skip(true, "No training jobs to verify response shape");
      return;
    }

    const job = jobs[0];
    expect(job).toHaveProperty("positive_embedding_set_ids");
    expect(job).toHaveProperty("negative_embedding_set_ids");
    expect(Array.isArray(job.negative_embedding_set_ids)).toBe(true);
    // Should NOT have the old field
    expect(job).not.toHaveProperty("negative_audio_folder");
  });
});

function buildCandidateSummary(overrides: Record<string, unknown>) {
  return {
    id: "candidate-base",
    name: "Candidate Base",
    status: "blocked",
    phase: "phase1",
    objective_name: "default",
    threshold: 0.5,
    comparison_target: "LR-v12",
    source_model_id: "model-lr-v12",
    source_model_name: "LR-v12",
    is_reproducible_exact: false,
    promoted_config: {
      classifier: "logreg",
      feature_norm: "l2",
      pca_dim: 128,
      prob_calibration: "platt",
      context_pooling: "mean3",
    },
    best_run_metrics: {
      precision: 0.98,
      recall: 0.94,
      fp_rate: 0.01,
    },
    split_metrics: {
      test: {
        autoresearch: { precision: 0.987, recall: 0.938, fp_rate: 0.0 },
        production: { precision: 0.959, recall: 0.942, fp_rate: 0.024 },
      },
    },
    metric_deltas: {
      test: {
        precision: 0.028,
        recall: -0.004,
        fp_rate: -0.024,
        fp: -92,
      },
    },
    replay_summary: {
      available_hard_negatives: 12,
      replayed_hard_negatives: 0,
      used_replay_manifest: false,
    },
    source_counts: {
      example_count: 1135,
      split_counts: { train: 464, val: 408, test: 263 },
    },
    warnings: [
      "PCA-backed candidates are not yet supported for promotion",
      "Probability calibration is not yet supported by the production trainer",
    ],
    training_job_id: null,
    new_model_id: null,
    error_message: null,
    created_at: "2026-04-03T17:02:17.095949Z",
    updated_at: "2026-04-03T17:02:17.095949Z",
    ...overrides,
  };
}

function buildCandidateDetail(overrides: Record<string, unknown>) {
  const summary = buildCandidateSummary(overrides);
  return {
    ...summary,
    artifact_paths: {
      manifest_path: "/tmp/fixture/manifest.json",
      best_run_path: "/tmp/fixture/phase1/best_run.json",
      comparison_path: "/tmp/fixture/phase1/comparison.json",
      top_false_positives_path: "/tmp/fixture/phase1/top_false_positives.json",
    },
    source_model_metadata: {
      name: "LR-v12",
      model_version: "surfperch-tensorflow2",
    },
    top_false_positives_preview: {
      imported: [
        {
          row_id: "fp-imported-1",
          audio_file_id: "audio-1",
          source_type: "detection_job",
          confidence: 0.998,
        },
      ],
      test: {
        autoresearch: [
          {
            row_id: "fp-auto-1",
            audio_file_id: "audio-1",
            source_type: "detection_job",
            autoresearch_score: 0.992,
          },
        ],
      },
    },
    prediction_disagreements_preview: {
      test: [
        {
          row_id: "disagree-1",
          audio_file_id: "audio-1",
          autoresearch_score: 0.91,
          production_score: 0.42,
          label: "ship",
        },
      ],
    },
  };
}

function buildVendoredPhase1CandidateSummary(overrides: Record<string, unknown>) {
  const comparison = VENDORED_PHASE1_COMPARISON;
  const bestRun = VENDORED_PHASE1_BEST_RUN;
  const manifest = VENDORED_MANIFEST;
  const manifestSummary =
    (comparison.manifest_summary as Record<string, unknown> | undefined) ?? {};
  const splitCounts =
    (manifestSummary.split_counts as Record<string, Record<string, number>> | undefined) ??
    {};
  const examples = Array.isArray(manifest.examples) ? manifest.examples : [];

  return buildCandidateSummary({
    name: "Blocked Candidate",
    status: "blocked",
    phase: "phase1",
    objective_name: (comparison.objective_name as string | undefined) ?? "default",
    threshold:
      ((bestRun.metrics as Record<string, unknown> | undefined)?.threshold as
        | number
        | undefined) ?? 0.5,
    comparison_target:
      ((comparison.production as Record<string, unknown> | undefined)?.name as
        | string
        | undefined) ?? "LR-v12",
    source_model_id:
      ((comparison.production as Record<string, unknown> | undefined)?.id as
        | string
        | undefined) ?? "model-lr-v12",
    source_model_name:
      ((comparison.production as Record<string, unknown> | undefined)?.name as
        | string
        | undefined) ?? "LR-v12",
    is_reproducible_exact: false,
    promoted_config: bestRun.config,
    best_run_metrics: bestRun.metrics,
    split_metrics: Object.fromEntries(
      Object.entries(
        (comparison.splits as Record<string, Record<string, unknown>> | undefined) ??
          {},
      ).map(([splitName, payload]) => [
        splitName,
        {
          autoresearch: (payload.autoresearch as Record<string, unknown>)?.metrics,
          production: (payload.production as Record<string, unknown>)?.metrics,
        },
      ]),
    ),
    metric_deltas: Object.fromEntries(
      Object.entries(
        (comparison.splits as Record<string, Record<string, unknown>> | undefined) ??
          {},
      ).map(([splitName, payload]) => [splitName, payload.delta]),
    ),
    replay_summary: manifestSummary.replay,
    source_counts: {
      example_count: examples.length,
      split_counts: Object.fromEntries(
        Object.entries(splitCounts).map(([splitName, split]) => [
          splitName,
          split.total,
        ]),
      ),
    },
    warnings: [
      "PCA-backed candidates are not yet supported for promotion",
      "Probability calibration is not yet supported by the production trainer",
      "Only context_pooling='center' is promotable today; pooled neighbor contexts are not yet reproduced by the production trainer",
    ],
    ...overrides,
  });
}

function buildVendoredPhase1CandidateDetail(overrides: Record<string, unknown>) {
  const summary = buildVendoredPhase1CandidateSummary(overrides);
  const comparison = VENDORED_PHASE1_COMPARISON;
  const testSplit = (comparison.splits as Record<string, Record<string, unknown>>).test;

  return {
    ...summary,
    artifact_paths: {
      manifest_path: resolve(AUTORESEARCH_FIXTURE_DIR, "manifest.json"),
      best_run_path: resolve(AUTORESEARCH_FIXTURE_DIR, "phase1/best_run.json"),
      comparison_path: resolve(
        AUTORESEARCH_FIXTURE_DIR,
        "phase1/lr-v12-comparison.json",
      ),
      top_false_positives_path: resolve(
        AUTORESEARCH_FIXTURE_DIR,
        "phase1/top_false_positives.json",
      ),
    },
    source_model_metadata: comparison.production ?? null,
    top_false_positives_preview: {
      imported: VENDORED_PHASE1_TOP_FALSE_POSITIVES.slice(0, 10),
      test: {
        autoresearch: (
          (testSplit.autoresearch as Record<string, unknown>).top_false_positives as
            | unknown[]
            | undefined
        )?.slice(0, 10),
        production: (
          (testSplit.production as Record<string, unknown>).top_false_positives as
            | unknown[]
            | undefined
        )?.slice(0, 10),
      },
    },
    prediction_disagreements_preview: {
      test: (
        testSplit.prediction_disagreements as unknown[] | undefined
      )?.slice(0, 10),
    },
  };
}

function parseRouteBody(route: import("@playwright/test").Route) {
  const raw = route.request().postData() ?? "{}";
  try {
    return JSON.parse(raw) as Record<string, string>;
  } catch {
    return {};
  }
}

async function mockAutoresearchCandidateApi(page: import("@playwright/test").Page) {
  const blockedId = "candidate-blocked";
  const promotableId = "candidate-promotable";

  let candidates = [
    buildVendoredPhase1CandidateSummary({
      id: blockedId,
      name: "Blocked Candidate",
      status: "blocked",
    }),
    buildCandidateSummary({
      id: promotableId,
      name: "Promotable Candidate",
      status: "promotable",
      phase: "phase2",
      is_reproducible_exact: true,
      promoted_config: {
        classifier: "logreg",
        feature_norm: "standard",
        class_weight_pos: 1.0,
        class_weight_neg: 1.0,
        context_pooling: "center",
      },
      warnings: [],
      metric_deltas: {
        test: {
          precision: 0.014,
          recall: 0.011,
          fp_rate: -0.009,
        },
      },
    }),
  ];

  const details = new Map<string, Record<string, unknown>>([
    [
      blockedId,
      buildVendoredPhase1CandidateDetail({
        id: blockedId,
        name: "Blocked Candidate",
        status: "blocked",
      }),
    ],
    [
      promotableId,
      buildCandidateDetail({
        id: promotableId,
        name: "Promotable Candidate",
        status: "promotable",
        phase: "phase2",
        is_reproducible_exact: true,
        promoted_config: {
          classifier: "logreg",
          feature_norm: "standard",
          class_weight_pos: 1.0,
          class_weight_neg: 1.0,
          context_pooling: "center",
        },
        warnings: [],
        metric_deltas: {
          test: {
            precision: 0.014,
            recall: 0.011,
            fp_rate: -0.009,
          },
        },
      }),
    ],
  ]);

  await page.route("**/audio/", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "[]",
    });
  });
  await page.route("**/processing/embedding-sets", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "[]",
    });
  });
  await page.route("**/classifier/training-jobs", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "[]",
    });
  });
  await page.route("**/classifier/models", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "[]",
    });
  });
  await page.route("**/classifier/retrain-workflows", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "[]",
    });
  });

  await page.route("**/classifier/autoresearch-candidates**", async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;

    if (pathname === "/classifier/autoresearch-candidates/import") {
      const body = parseRouteBody(route);
      const importedId = "candidate-imported";
      const importedSummary = buildVendoredPhase1CandidateSummary({
        id: importedId,
        name: body.name || "Imported Candidate",
        status: "blocked",
      });
      const importedDetail = {
        ...buildVendoredPhase1CandidateDetail({
          id: importedId,
          name: body.name || "Imported Candidate",
          status: "blocked",
        }),
        artifact_paths: {
          manifest_path: body.manifest_path ?? resolve(AUTORESEARCH_FIXTURE_DIR, "manifest.json"),
          best_run_path: body.best_run_path ?? resolve(AUTORESEARCH_FIXTURE_DIR, "phase1/best_run.json"),
          comparison_path:
            body.comparison_path ??
            resolve(AUTORESEARCH_FIXTURE_DIR, "phase1/lr-v12-comparison.json"),
          top_false_positives_path:
            body.top_false_positives_path ??
            resolve(AUTORESEARCH_FIXTURE_DIR, "phase1/top_false_positives.json"),
        },
      };
      candidates = [importedSummary, ...candidates];
      details.set(importedId, importedDetail);

      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(importedDetail),
      });
      return;
    }

    if (pathname === "/classifier/autoresearch-candidates") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(candidates),
      });
      return;
    }

    if (pathname.endsWith("/training-jobs")) {
      const candidateId = pathname.split("/").slice(-2, -1)[0] ?? "";
      const body = parseRouteBody(route);
      const detail = details.get(candidateId);
      expect(body.new_model_name).toBe("candidate-backed-ui");
      expect(detail).toBeTruthy();

      candidates = candidates.map((candidate) =>
        candidate.id === candidateId
          ? { ...candidate, status: "training", training_job_id: "training-job-1" }
          : candidate,
      );

      if (detail) {
        details.set(candidateId, {
          ...detail,
          status: "training",
          training_job_id: "training-job-1",
        });
      }

      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          id: "training-job-1",
          status: "queued",
          name: body.new_model_name,
          positive_embedding_set_ids: [],
          negative_embedding_set_ids: [],
          model_version: "surfperch-tensorflow2",
          window_size_seconds: 3,
          target_sample_rate: 16000,
          feature_config: null,
          parameters: { classifier_type: "logistic_regression" },
          classifier_model_id: null,
          error_message: null,
          source_mode: "autoresearch_candidate",
          source_candidate_id: candidateId,
          source_model_id: "model-lr-v12",
          manifest_path: "/tmp/fixture/manifest.json",
          training_split_name: "train",
          promoted_config: { classifier: "logreg" },
          source_comparison_context: {
            candidate_id: candidateId,
            candidate_name: "Promotable Candidate",
          },
          created_at: "2026-04-03T17:02:17.095949Z",
          updated_at: "2026-04-03T17:02:17.095949Z",
        }),
      });
      return;
    }

    const candidateId = pathname.split("/").pop() ?? "";
    const detail = details.get(candidateId);
    await route.fulfill({
      status: detail ? 200 : 404,
      contentType: "application/json",
      body: detail ? JSON.stringify(detail) : JSON.stringify({ detail: "not found" }),
    });
  });
}

test.describe("Autoresearch candidates UI", () => {
  test("imports, reviews, and promotes candidates from the training tab", async ({
    page,
  }) => {
    await mockAutoresearchCandidateApi(page);
    await page.goto("/app/classifier/training");

    await expect(page.getByText("Autoresearch Candidates")).toBeVisible();
    await expect(page.getByText("Blocked Candidate")).toBeVisible();
    await expect(page.getByText("Promotable Candidate")).toBeVisible();
    await expect(page.getByText("Source model: LR-v12").first()).toBeVisible();
    await expect(page.getByText("Replay 0/12 hard negatives").first()).toBeVisible();

    await page.getByPlaceholder("optional display name").fill("Imported Fixture");
    await page.getByPlaceholder("/abs/path/manifest.json").fill("/fixtures/manifest.json");
    await page.getByPlaceholder("/abs/path/phase1/best_run.json").fill("/fixtures/phase1/best_run.json");
    await page.getByPlaceholder("/abs/path/comparison.json").fill("/fixtures/comparison.json");
    await page.getByPlaceholder("/abs/path/top_false_positives.json").fill("/fixtures/top_false_positives.json");
    await page.getByRole("button", { name: "Import Candidate" }).click();

    await expect(page.getByText("Imported Fixture")).toBeVisible();

    await page.getByRole("button", { name: /Blocked Candidate/ }).click();
    await expect(page.getByText("Promotion warnings")).toBeVisible();
    await expect(
      page.getByText("PCA-backed candidates are not yet supported for promotion"),
    ).toBeVisible();
    await expect(page.getByText("Promoted Config")).toBeVisible();
    await expect(page.getByText("Comparison Metrics")).toBeVisible();
    await expect(page.getByText("Prediction Disagreements (test)")).toBeVisible();

    await page.getByRole("button", { name: /Promotable Candidate/ }).click();
    await page.getByLabel("New Model Name").fill("candidate-backed-ui");
    await page.getByRole("button", { name: "Start Candidate Training" }).click();

    await expect(page.getByText("Candidate training is in progress.")).toBeVisible();
  });
});
