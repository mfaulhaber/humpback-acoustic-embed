import { expect, test, type Page, type Route } from "@playwright/test";

const COMPLETE_JOB = {
  id: "mt-motif-ux",
  status: "complete",
  status_reason: null,
  continuous_embedding_job_id: "cej-mt-ux",
  training_signature: "sig-ux",
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
  k_values: [50],
  chosen_device: "mps",
  fallback_reason: null,
  final_train_loss: 0.12,
  final_val_loss: 0.18,
  total_epochs: 8,
  total_sequences: 2,
  total_chunks: 12,
  job_dir: "/tmp/data/masked_transformer_jobs/mt-motif-ux",
  error_message: null,
  created_at: "2026-04-29T01:00:00Z",
  updated_at: "2026-04-29T01:10:00Z",
};

const COMPLETE_JOB_DETAIL = {
  job: COMPLETE_JOB,
  region_detection_job_id: "rdj-mt-ux",
  region_start_timestamp: 0.0,
  region_end_timestamp: 600.0,
  tier_composition: null,
  source_kind: "region_crnn",
};

const LOSS_CURVE = {
  epochs: [1, 2],
  train_loss: [0.5, 0.4],
  val_loss: [0.55, 0.45],
  val_metrics: { final_val_loss: 0.45 },
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
  ],
};

const RUN_LENGTHS = { k: 50, tau: 1.5, run_lengths: { "5": [1, 1], "7": [1] } };
const OVERLAY = { total: 0, items: [] };
const EXEMPLARS = { n_states: 50, states: {} };
const LABEL_DIST = { n_states: 50, total_windows: 0, states: {} };

const MOTIF_JOB = {
  id: "motif-mt-ux",
  status: "complete",
  parent_kind: "masked_transformer",
  hmm_sequence_job_id: null,
  masked_transformer_job_id: COMPLETE_JOB.id,
  k: 50,
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
  config_signature: "motif-ux-sig",
  total_groups: 2,
  total_collapsed_tokens: 200,
  total_candidate_occurrences: 100,
  total_motifs: 1,
  artifact_dir: "/tmp/data/motif_extractions/motif-mt-ux",
  error_message: null,
  created_at: "2026-04-29T02:00:00Z",
  updated_at: "2026-04-29T02:01:00Z",
};

function summary(motif_key: string, length: number, rank: number) {
  return {
    motif_key,
    states: motif_key.split("-").map((s) => Number.parseInt(s, 10)),
    length,
    occurrence_count: 8,
    event_source_count: 3,
    audio_source_count: 2,
    group_count: 1,
    event_core_fraction: 0.8,
    background_fraction: 0.1,
    mean_call_probability: null,
    mean_duration_seconds: 0.5,
    median_duration_seconds: 0.5,
    rank_score: rank,
    example_occurrence_ids: [],
  };
}

const MOTIFS = {
  total: 4,
  offset: 0,
  limit: 100,
  items: [
    // Primary length-2 motif used by the legacy single-mode tests below.
    {
      motif_key: "5-7",
      states: [5, 7],
      length: 2,
      occurrence_count: 25,
      event_source_count: 5,
      audio_source_count: 3,
      group_count: 2,
      event_core_fraction: 0.8,
      background_fraction: 0.1,
      mean_call_probability: null,
      mean_duration_seconds: 0.5,
      median_duration_seconds: 0.5,
      rank_score: 0.95,
      example_occurrence_ids: ["occ-0"],
    },
    // Additional length-2 / length-3 / length-4 motifs feed the Token
    // Count selector tests.
    summary("2-9", 2, 0.85),
    summary("5-7-2", 3, 0.9),
    summary("5-7-2-9", 4, 0.88),
  ],
};

// 25 occurrences, all inside the default 5m viewport (centered on 100s),
// so the spec can assert that ≥20 alignment rows render and that overlay
// bands appear in-viewport.
const MOTIF_OCCURRENCES = {
  total: 25,
  offset: 0,
  limit: 100,
  items: Array.from({ length: 25 }, (_, i) => {
    const start = 60 + i * 4;
    const end = start + 0.5;
    return {
      occurrence_id: `occ-${i}`,
      motif_key: "5-7",
      states: [5, 7],
      source_kind: "region_crnn",
      group_key: "g0",
      event_source_key: `evt-${i}`,
      audio_source_key: "1",
      token_start_index: i * 2,
      token_end_index: i * 2 + 1,
      raw_start_index: i * 2,
      raw_end_index: i * 2 + 1,
      start_timestamp: start,
      end_timestamp: end,
      duration_seconds: end - start,
      event_core_fraction: 1.0,
      background_fraction: 0.0,
      mean_call_probability: null,
      anchor_event_id: `evt-${i}`,
      anchor_timestamp: (start + end) / 2,
      relative_start_seconds: -0.25,
      relative_end_seconds: 0.25,
      anchor_strategy: "event_midpoint",
    };
  }),
};

/**
 * Smaller per-motif occurrence fixtures keyed by motif_key. Used by the
 * Token Count selector tests so different motifs of the same length
 * carry different colors and different time ranges.
 */
function makeOccurrences(motifKey: string, count: number, baseStart: number) {
  return {
    total: count,
    offset: 0,
    limit: 100,
    items: Array.from({ length: count }, (_, i) => {
      const start = baseStart + i * 8;
      const end = start + 0.6;
      return {
        occurrence_id: `${motifKey}-occ-${i}`,
        motif_key: motifKey,
        states: motifKey.split("-").map((s) => Number.parseInt(s, 10)),
        source_kind: "region_crnn",
        group_key: "g0",
        event_source_key: `${motifKey}-evt-${i}`,
        audio_source_key: "1",
        token_start_index: i,
        token_end_index: i + 1,
        raw_start_index: i,
        raw_end_index: i + 1,
        start_timestamp: start,
        end_timestamp: end,
        duration_seconds: end - start,
        event_core_fraction: 1.0,
        background_fraction: 0.0,
        mean_call_probability: null,
        anchor_event_id: `${motifKey}-evt-${i}`,
        anchor_timestamp: (start + end) / 2,
        relative_start_seconds: -0.3,
        relative_end_seconds: 0.3,
        anchor_strategy: "event_midpoint",
      };
    }),
  };
}

const PER_MOTIF_OCCURRENCES: Record<string, ReturnType<typeof makeOccurrences>> = {
  "5-7": MOTIF_OCCURRENCES,
  "2-9": makeOccurrences("2-9", 4, 70),
  "5-7-2": makeOccurrences("5-7-2", 4, 80),
  "5-7-2-9": makeOccurrences("5-7-2-9", 4, 90),
};

interface MockState {
  audioSliceUrls: string[];
}

async function setupMocks(page: Page, state: MockState): Promise<void> {
  await page.route(
    `**/sequence-models/masked-transformers/${COMPLETE_JOB.id}`,
    (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(COMPLETE_JOB_DETAIL),
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

  await page.route("**/sequence-models/motif-extractions**", (route: Route) => {
    const url = route.request().url();
    const method = route.request().method();
    const occMatch = url.match(/\/motifs\/([^/]+)\/occurrences/);
    if (occMatch) {
      const motifKey = decodeURIComponent(occMatch[1]);
      const body =
        PER_MOTIF_OCCURRENCES[motifKey] ?? MOTIF_OCCURRENCES;
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    }
    if (url.match(/\/motif-extractions\/[^/?]+\/motifs/)) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOTIFS),
      });
    }
    if (method === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([MOTIF_JOB]),
      });
    }
    return route.fulfill({ status: 405 });
  });

  await page.route("**/call-parsing/region-jobs/*/tile**", (route) => {
    const pixel = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "base64",
    );
    return route.fulfill({ status: 200, contentType: "image/png", body: pixel });
  });
  await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) => {
    state.audioSliceUrls.push(route.request().url());
    return route.fulfill({
      status: 200,
      contentType: "audio/mpeg",
      body: Buffer.alloc(0),
    });
  });
}

test.describe("Masked Transformer Motif UX", () => {
  test("renders motif highlight bands in timeline column", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("masked-transformer-detail-page")).toBeVisible();
    await expect(page.getByTestId("motif-table")).toBeVisible();

    // Selected motif auto-populates from the first row; bands should
    // appear in the spectrogram overlay layer for occurrences in-view.
    await expect(page.getByTestId("mt-motif-highlight-layer")).toBeVisible();
    const bands = page.getByTestId("mt-motif-highlight-band");
    expect(await bands.count()).toBeGreaterThan(0);

    // Default active occurrence is index 0 — exactly one band carries it.
    const activeBands = page.locator(
      '[data-testid="mt-motif-highlight-band"][data-active="true"]',
    );
    await expect(activeBands).toHaveCount(1);
    await expect(activeBands.first()).toHaveAttribute(
      "data-occurrence-index",
      "0",
    );
  });

  test("Jump moves the active band to the clicked occurrence", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("motif-example-row-0")).toBeVisible();
    // Row 2's Jump button — choose a row whose data-active should flip to true.
    const row2 = page.getByTestId("motif-example-row-2");
    await row2.getByRole("button", { name: "Jump" }).click();

    const activeBand = page.locator(
      '[data-testid="mt-motif-highlight-band"][data-active="true"]',
    );
    await expect(activeBand).toHaveCount(1);
    await expect(activeBand).toHaveAttribute("data-occurrence-index", "2");
  });

  test("Play requests motif-bounded audio with no padding", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("motif-example-row-3")).toBeVisible();
    await page
      .getByTestId("motif-example-row-3")
      .getByRole("button", { name: "Play" })
      .click();

    await expect.poll(() => state.audioSliceUrls.length).toBeGreaterThan(0);
    const playUrl = new URL(state.audioSliceUrls[state.audioSliceUrls.length - 1]);
    const occ = MOTIF_OCCURRENCES.items[3];
    expect(Number(playUrl.searchParams.get("start_timestamp"))).toBeCloseTo(
      occ.start_timestamp,
      3,
    );
    // Exact (end - start), no ±1s padding.
    expect(Number(playUrl.searchParams.get("duration_sec"))).toBeCloseTo(
      occ.end_timestamp - occ.start_timestamp,
      3,
    );

    // Clicking Play also marks that occurrence active.
    const activeBand = page.locator(
      '[data-testid="mt-motif-highlight-band"][data-active="true"]',
    );
    await expect(activeBand).toHaveAttribute("data-occurrence-index", "3");
  });

  test("conf and recon strips are hidden", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("masked-transformer-detail-page")).toBeVisible();
    await expect(page.getByTestId("mt-token-confidence-strip")).toHaveCount(0);
    await expect(page.getByTestId("mt-reconstruction-error-strip")).toHaveCount(0);
  });

  test("Token Count selector starts unset and toggles byLength mode", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("motif-token-count-selector")).toBeVisible();
    // Selector starts unset; single-mode highlight should be active for
    // the auto-picked first motif.
    await expect(page.getByTestId("motif-token-count-2")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    await expect(page.getByTestId("motif-token-count-3")).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    // Click length-3 → byLength mode renders highlight bands for the
    // length-3 motif's occurrences (per the per-motif fixture).
    await page.getByTestId("motif-token-count-3").click();
    await expect(page.getByTestId("motif-token-count-3")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    const bands = page.getByTestId("mt-motif-highlight-band");
    await expect.poll(() => bands.count()).toBeGreaterThan(0);
    // Every byLength-mode band carries a data-motif-key attribute.
    const firstBandKey = await bands
      .first()
      .getAttribute("data-motif-key");
    expect(firstBandKey).toBe("5-7-2");

    // Clicking the active value again returns to single-motif mode.
    await page.getByTestId("motif-token-count-3").click();
    await expect(page.getByTestId("motif-token-count-3")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  test("byLength mode prev/next walks the visible occurrence set", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("motif-token-count-selector")).toBeVisible();
    await page.getByTestId("motif-token-count-2").click();

    // Active occurrence should start at index 0; clicking next bumps it.
    const activeBand = page.locator(
      '[data-testid="mt-motif-highlight-band"][data-active="true"]',
    );
    await expect(activeBand).toHaveCount(1);
    const initialIdx = await activeBand.getAttribute("data-occurrence-index");
    await page.getByTestId("motif-timeline-legend-next").click();
    const nextIdx = await page
      .locator('[data-testid="mt-motif-highlight-band"][data-active="true"]')
      .getAttribute("data-occurrence-index");
    expect(nextIdx).not.toBe(initialIdx);
  });

  test("Picking a motif row exits byLength mode", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await page.getByTestId("motif-token-count-3").click();
    await expect(page.getByTestId("motif-token-count-3")).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    // Click any row in the motif table; selector should clear.
    await page.locator('[data-testid="motif-table"] tbody tr').first().click();
    await expect(page.getByTestId("motif-token-count-3")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  test("byLength legend Play requests audio for the active occurrence", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await page.getByTestId("motif-token-count-2").click();
    await expect(page.getByTestId("motif-timeline-legend-play")).toBeVisible();
    state.audioSliceUrls.length = 0;
    await page.getByTestId("motif-timeline-legend-play").click();

    await expect.poll(() => state.audioSliceUrls.length).toBeGreaterThan(0);
  });

  test("alignment list shows up to 20 rows and scrolls", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("motif-example-list")).toBeVisible();
    // 25 occurrences exist; the list caps at 20 rows.
    await expect(page.getByTestId("motif-example-row-19")).toBeVisible();
    await expect(page.getByTestId("motif-example-row-20")).toHaveCount(0);

    const list = page.getByTestId("motif-example-list");
    const overflowY = await list.evaluate(
      (el) => getComputedStyle(el).overflowY,
    );
    expect(overflowY).toBe("auto");
  });

  test("overlay band layer clips children; tooltip layer is unclipped", async ({ page }) => {
    const state: MockState = { audioSliceUrls: [] };
    await setupMocks(page, state);
    await page.goto(`/app/sequence-models/masked-transformer/${COMPLETE_JOB.id}`);

    await expect(page.getByTestId("masked-transformer-detail-page")).toBeVisible();
    await expect(page.getByTestId("mt-motif-highlight-layer")).toBeVisible();

    const band = page.getByTestId("overlay-band-layer");
    const tooltip = page.getByTestId("overlay-tooltip-layer");
    await expect(band).toBeAttached();
    await expect(tooltip).toBeAttached();

    const bandOverflow = await band.evaluate((el) => getComputedStyle(el).overflow);
    expect(bandOverflow).toBe("hidden");

    const tooltipOverflow = await tooltip.evaluate((el) => getComputedStyle(el).overflow);
    expect(tooltipOverflow).not.toBe("hidden");
  });
});
