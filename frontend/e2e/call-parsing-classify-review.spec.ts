import { expect, test, type Page } from "@playwright/test";

const HYDROPHONE = {
  id: "rpi_north_sjc",
  name: "North San Juan Channel",
  location: "San Juan Channel",
  provider_kind: "orcasound_hls",
};

const REGION_JOB = {
  id: "rj-1",
  status: "complete",
  audio_file_id: null,
  hydrophone_id: HYDROPHONE.id,
  start_timestamp: 1751644800,
  end_timestamp: 1751648400,
  model_config_id: "mc-1",
  classifier_model_id: "cm-1",
  config_json: "{}",
  parent_run_id: null,
  error_message: null,
  chunks_total: null,
  chunks_completed: null,
  windows_detected: null,
  trace_row_count: 3600,
  region_count: 1,
  created_at: "2026-04-12T01:00:00Z",
  updated_at: "2026-04-12T01:30:00Z",
  started_at: "2026-04-12T01:00:01Z",
  completed_at: "2026-04-12T01:30:00Z",
};

const SEG_JOB = {
  id: "sj-1",
  status: "complete",
  region_detection_job_id: REGION_JOB.id,
  segmentation_model_id: "sm-1",
  config_json: "{}",
  parent_run_id: null,
  event_count: 4,
  error_message: null,
  created_at: "2026-04-12T04:00:00Z",
  updated_at: "2026-04-12T04:05:00Z",
  started_at: "2026-04-12T04:00:01Z",
  completed_at: "2026-04-12T04:05:00Z",
};

const CLASSIFY_JOB = {
  id: "cj-1",
  status: "complete",
  event_segmentation_job_id: SEG_JOB.id,
  vocalization_model_id: "vm-1",
  typed_event_count: 4,
  error_message: null,
  created_at: "2026-04-12T05:00:00Z",
  updated_at: "2026-04-12T05:05:00Z",
  started_at: "2026-04-12T05:00:01Z",
  completed_at: "2026-04-12T05:05:00Z",
};

const REGIONS = [
  {
    region_id: "reg-aaaa",
    start_sec: 95.0,
    end_sec: 130.0,
    padded_start_sec: 94.0,
    padded_end_sec: 131.0,
    max_score: 0.95,
    mean_score: 0.88,
    n_windows: 5,
  },
];

// Four test events covering all four badge states:
//   ev-inference: above-threshold Moan (typeSource=inference)
//   ev-corrected-pos: will be corrected to Buzz via palette click (typeSource=correction)
//   ev-corrected-neg: will be marked negative via (Negative) click
//   ev-unlabeled: only below-threshold scores, no correction (typeSource=null)
const TYPED_EVENTS = [
  {
    event_id: "ev-inference",
    region_id: "reg-aaaa",
    start_sec: 100.0,
    end_sec: 102.0,
    type_name: "Moan",
    score: 0.92,
    above_threshold: true,
  },
  {
    event_id: "ev-corrected-pos",
    region_id: "reg-aaaa",
    start_sec: 105.0,
    end_sec: 107.0,
    type_name: "Buzz",
    score: 0.85,
    above_threshold: true,
  },
  {
    event_id: "ev-corrected-neg",
    region_id: "reg-aaaa",
    start_sec: 110.0,
    end_sec: 112.0,
    type_name: "Moan",
    score: 0.88,
    above_threshold: true,
  },
  {
    event_id: "ev-unlabeled",
    region_id: "reg-aaaa",
    start_sec: 115.0,
    end_sec: 117.0,
    type_name: "Cry",
    score: 0.42,
    above_threshold: false,
  },
];

const VOC_TYPES = [
  { id: "vt-moan", name: "Moan", description: null },
  { id: "vt-buzz", name: "Buzz", description: null },
  { id: "vt-cry", name: "Cry", description: null },
];

async function setupMocks(page: Page) {
  await page.route("**/classifier/hydrophones", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([HYDROPHONE]),
    }),
  );
  await page.route("**/call-parsing/region-jobs", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([REGION_JOB]),
    }),
  );
  await page.route("**/call-parsing/segmentation-jobs*", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([SEG_JOB]),
      });
    }
    return route.fulfill({ status: 405, body: "" });
  });
  await page.route("**/call-parsing/classification-jobs*", (route) => {
    const url = route.request().url();
    if (url.includes("/typed-events")) return route.continue();
    if (url.includes("/corrections")) return route.continue();
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([CLASSIFY_JOB]),
      });
    }
    return route.fulfill({ status: 405, body: "" });
  });
  await page.route(
    "**/call-parsing/classification-jobs/*/typed-events",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(TYPED_EVENTS),
      }),
  );
  // Initially no type corrections; we'll update on POST.
  let currentTypeCorrections: { event_id: string; type_name: string | null }[] = [];
  await page.route(
    "**/call-parsing/classification-jobs/*/corrections",
    (route) => {
      if (route.request().method() === "POST") {
        const body = route.request().postDataJSON() as {
          corrections: { event_id: string; type_name: string | null }[];
        };
        currentTypeCorrections = body.corrections;
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(
            currentTypeCorrections.map((c, i) => ({
              id: `tc-${i}`,
              event_classification_job_id: CLASSIFY_JOB.id,
              event_id: c.event_id,
              type_name: c.type_name,
              created_at: "2026-04-16T00:00:00Z",
            })),
          ),
        });
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(
          currentTypeCorrections.map((c, i) => ({
            id: `tc-${i}`,
            event_classification_job_id: CLASSIFY_JOB.id,
            event_id: c.event_id,
            type_name: c.type_name,
            created_at: "2026-04-16T00:00:00Z",
          })),
        ),
      });
    },
  );
  await page.route("**/call-parsing/region-jobs/*/regions", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(REGIONS),
    }),
  );
  await page.route(
    "**/call-parsing/segmentation-jobs/*/corrections",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: "[]",
      }),
  );
  await page.route("**/vocalization/types", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(VOC_TYPES),
    }),
  );
  await page.route("**/call-parsing/classifier-models", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  // 1x1 transparent PNG for tile requests
  const TINY_PNG = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB" +
      "Nl7BcQAAAABJRU5ErkJggg==",
    "base64",
  );
  await page.route("**/call-parsing/region-jobs/*/tile*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "image/png",
      body: TINY_PNG,
    }),
  );
  await page.route("**/classifier/models", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
}

async function openReview(page: Page) {
  await page.goto(
    `/app/call-parsing/classify?tab=review&reviewJobId=${CLASSIFY_JOB.id}`,
  );
  // Wait for the review workspace to have loaded events.
  await expect(page.getByText(/Event \d+ of \d+/)).toBeVisible();
}

async function goToEventByIndex(page: Page, oneBasedIndex: number) {
  const counter = page.getByText(/Event \d+ of \d+/);
  await expect(counter).toBeVisible();
  // Navigate from event 1 forward.
  for (let i = 1; i < oneBasedIndex; i++) {
    await page.keyboard.press("ArrowRight");
  }
}

test.describe("Classify Review — None palette indicator", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("None chip is visible when current event has no effective type", async ({
    page,
  }) => {
    await openReview(page);
    // Event 4 is ev-unlabeled (below-threshold only, no correction).
    await goToEventByIndex(page, 4);
    const none = page.getByTestId("palette-none-indicator");
    await expect(none).toBeVisible();
  });

  test("None chip hides but preserves layout when a type is active", async ({
    page,
  }) => {
    await openReview(page);
    // Event 1 has an inference type ("Moan"), so the None chip should be hidden.
    await goToEventByIndex(page, 1);
    const none = page.getByTestId("palette-none-indicator");
    // In the DOM but not visible to the user (visibility: hidden).
    await expect(none).toHaveCSS("visibility", "hidden");
    // The chip still occupies layout space — width > 0.
    const box = await none.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(0);
  });

  test("palette layout does not reflow when None chip toggles", async ({
    page,
  }) => {
    await openReview(page);
    // Locate the (Negative) chip — its position should be stable whether
    // the None chip is visible or hidden.
    const negativeChip = page.getByRole("button", { name: "(Negative)" });

    // Event 4 — unlabeled, None visible.
    await goToEventByIndex(page, 4);
    const withNoneVisible = await negativeChip.boundingBox();
    expect(withNoneVisible).not.toBeNull();

    // Event 1 — inference, None hidden.
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    const withNoneHidden = await negativeChip.boundingBox();
    expect(withNoneHidden).not.toBeNull();

    expect(withNoneHidden!.x).toBeCloseTo(withNoneVisible!.x, 1);
  });
});

test.describe("Classify Review — event badges", () => {
  test.beforeEach(async ({ page }) => {
    await setupMocks(page);
  });

  test("inference event shows bordered badge with uppercased 2-letter code", async ({
    page,
  }) => {
    await openReview(page);
    const badge = page.getByTestId("event-badge-ev-inference");
    await expect(badge).toBeVisible();
    await expect(badge).toHaveAttribute("data-source", "inference");
    await expect(badge).toHaveText("MO");
  });

  test("unlabeled event has no badge", async ({ page }) => {
    await openReview(page);
    const badge = page.getByTestId("event-badge-ev-unlabeled");
    await expect(badge).toHaveCount(0);
  });

  test("clicking a palette type flips an inference badge to solid correction style", async ({
    page,
  }) => {
    await openReview(page);
    // Select the corrected-pos event by navigating to it (event 2).
    await goToEventByIndex(page, 2);
    // Sanity: badge starts as inference.
    const badge = page.getByTestId("event-badge-ev-corrected-pos");
    await expect(badge).toHaveAttribute("data-source", "inference");

    // Click the Cry palette button to apply a correction.
    await page.getByRole("button", { name: "Cry" }).click();

    await expect(badge).toHaveAttribute("data-source", "correction");
    await expect(badge).toHaveText("CR");
  });

  test("inference badge has a white background, correction badge has the palette color", async ({
    page,
  }) => {
    await openReview(page);
    const inference = page.getByTestId("event-badge-ev-inference");
    await expect(inference).toHaveAttribute("data-source", "inference");
    await expect(inference).toHaveCSS(
      "background-color",
      "rgb(255, 255, 255)",
    );

    // Promote the inference label to a correction by clicking the highlighted
    // palette button — the background should swap from white to the palette
    // color.
    await goToEventByIndex(page, 1);
    await page.getByRole("button", { name: "Moan" }).click();
    await expect(inference).toHaveAttribute("data-source", "correction");
    await expect(inference).not.toHaveCSS(
      "background-color",
      "rgb(255, 255, 255)",
    );
  });

  test("clicking the already-highlighted inference type promotes it to a correction", async ({
    page,
  }) => {
    await openReview(page);
    // Event 1 is inference-only for "Moan".
    await goToEventByIndex(page, 1);
    const badge = page.getByTestId("event-badge-ev-inference");
    await expect(badge).toHaveAttribute("data-source", "inference");

    // Clicking the highlighted "Moan" palette button should promote the
    // prediction to a human correction and flag the workspace as dirty.
    await page.getByRole("button", { name: "Moan" }).click();
    await expect(badge).toHaveAttribute("data-source", "correction");
    await expect(page.getByText(/unsaved change/)).toBeVisible();

    // Clicking "Moan" again is idempotent — it must not add another pending
    // correction (count stays at 1).
    await page.getByRole("button", { name: "Moan" }).click();
    await expect(page.getByText("1 unsaved change")).toBeVisible();
  });

  test("clicking (Negative) sets solid red badge with em dash", async ({
    page,
  }) => {
    await openReview(page);
    // Event 3 — navigate.
    await goToEventByIndex(page, 3);
    const badge = page.getByTestId("event-badge-ev-corrected-neg");
    await expect(badge).toHaveAttribute("data-source", "inference");

    await page.getByRole("button", { name: "(Negative)" }).click();

    await expect(badge).toHaveAttribute("data-source", "negative");
    await expect(badge).toHaveText("—");
  });

  test("overlay band layer is clipped; tooltip layer is unclipped", async ({
    page,
  }) => {
    await openReview(page);
    const band = page.getByTestId("overlay-band-layer").first();
    await expect(band).toBeAttached();
    const overflow = await band.evaluate((el) => getComputedStyle(el).overflow);
    expect(overflow).toBe("hidden");
    await expect(page.getByTestId("overlay-tooltip-layer").first()).toBeAttached();
  });
});
