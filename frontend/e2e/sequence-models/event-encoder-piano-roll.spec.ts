import { expect, test, type Page } from "@playwright/test";

const SILENT_WAV = Buffer.from(
  "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA=",
  "base64",
);

const REGION_JOB_ID = "rj-piano-roll-1";
const JOB_START = 1_751_644_800;
const JOB_END = JOB_START + 300;

const COMPLETE_JOB = {
  id: "eej-piano-roll-1",
  status: "complete",
  event_segmentation_job_id: "seg-piano-roll-1",
  event_source_mode: "raw",
  continuous_embedding_job_id: "cej-piano-roll-1",
  continuous_embedding_signature: "piano-roll-embedding-sig",
  tokenizer_version: "crnn-event-encoder-v2",
  pooling_config_json: "{}",
  descriptor_config_json: "{}",
  preprocessing_config_json: "{}",
  k_values_json: "[50,100]",
  random_seed: 0,
  tokenization_signature: "piano-roll-token-sig",
  event_vector_dim: 136,
  total_events: 5,
  encoded_events: 5,
  skipped_events: 0,
  event_vectors_path: "/tmp/event-encoders/eej-piano-roll-1/event_vectors.parquet",
  event_tokens_path: "/tmp/event-encoders/eej-piano-roll-1/event_tokens.parquet",
  token_sequences_path:
    "/tmp/event-encoders/eej-piano-roll-1/token_sequences.parquet",
  manifest_path: "/tmp/event-encoders/eej-piano-roll-1/manifest.json",
  report_path: "/tmp/event-encoders/eej-piano-roll-1/report.json",
  error_message: null,
  created_at: "2026-05-08T01:00:00Z",
  updated_at: "2026-05-08T01:10:00Z",
};

const QUEUED_JOB = {
  ...COMPLETE_JOB,
  id: "eej-piano-roll-queued",
  status: "queued",
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
};

const COMPLETE_DETAIL = {
  job: COMPLETE_JOB,
  manifest: {
    job_id: COMPLETE_JOB.id,
    valid_k_values: [50, 100],
    total_events: 5,
    encoded_events: 5,
    skipped_events: 0,
  },
  report: {
    summary: {
      total_events: 5,
      encoded_events: 5,
      skipped_events: 0,
      valid_k_values: [50, 100],
    },
    sequence_preview: {
      "50": ["T03", "T07", "T03", "T07", "T03"],
    },
  },
};

const QUEUED_DETAIL = {
  job: QUEUED_JOB,
  manifest: null,
  report: null,
};

const TIMELINE_50 = {
  job_id: COMPLETE_JOB.id,
  event_segmentation_job_id: COMPLETE_JOB.event_segmentation_job_id,
  event_source_mode: "raw",
  continuous_embedding_job_id: COMPLETE_JOB.continuous_embedding_job_id,
  region_detection_job_id: REGION_JOB_ID,
  selected_k: 50,
  valid_k_values: [50, 100],
  descriptor_feature_names: [
    "duration",
    "peak_frequency",
    "ridge_log_frequency_slope",
    "gap_to_previous",
    "median_f0",
    "f0_range",
    "voicing_fraction",
    "pulse_rate",
  ],
  descriptor_feature_units: {
    duration: "seconds",
    peak_frequency: "Hz",
    ridge_log_frequency_slope: "octaves/s",
    gap_to_previous: "seconds",
    median_f0: "Hz",
    f0_range: "Hz",
    voicing_fraction: "ratio",
    pulse_rate: "Hz",
  },
  job_start_timestamp: JOB_START,
  job_end_timestamp: JOB_END,
  events: [
    timelineEvent("evt-a", 3, "T03", 10, 12, {
      median_f0: 480,
      f0_range: 120,
      peak_frequency: 520,
      voicing_fraction: 0.91,
      ridge_log_frequency_slope: 1.2,
      pulse_rate: 2.5,
      gap_to_previous: 0,
    }),
    timelineEvent("evt-b", 7, "T07", 28, 29.4, {
      median_f0: 880,
      f0_range: 180,
      peak_frequency: 980,
      voicing_fraction: 0.83,
      ridge_log_frequency_slope: -0.9,
      pulse_rate: 3.1,
      gap_to_previous: 16,
    }),
    timelineEvent("evt-c", 3, "T03", 42, 43.5, {
      median_f0: 520,
      f0_range: 90,
      peak_frequency: 560,
      voicing_fraction: 0.72,
      ridge_log_frequency_slope: 0.2,
      pulse_rate: 2.8,
      gap_to_previous: 12.6,
    }),
    timelineEvent("evt-d", 7, "T07", 75, 75.9, {
      median_f0: 0,
      f0_range: 0,
      peak_frequency: 1420,
      voicing_fraction: 0.12,
      ridge_log_frequency_slope: -1.7,
      pulse_rate: 0,
      gap_to_previous: 31.5,
    }),
    timelineEvent("evt-e", 3, "T03", 108, 110, {
      median_f0: 650,
      f0_range: 140,
      peak_frequency: 700,
      voicing_fraction: 0.88,
      ridge_log_frequency_slope: 0.7,
      pulse_rate: 2.2,
      gap_to_previous: 32.1,
    }),
  ],
};

const TIMELINE_100 = {
  ...TIMELINE_50,
  selected_k: 100,
  events: TIMELINE_50.events.map((event, index) => ({
    ...event,
    token_id: index % 2 === 0 ? 11 : 24,
    token_label: index % 2 === 0 ? "T11" : "T24",
  })),
};

interface MockState {
  timelineRequests: string[];
  audioRequests: string[];
}

function timelineEvent(
  eventId: string,
  tokenId: number,
  tokenLabel: string,
  startOffset: number,
  endOffset: number,
  descriptors: {
    median_f0: number;
    f0_range: number;
    peak_frequency: number;
    voicing_fraction: number;
    ridge_log_frequency_slope: number;
    pulse_rate: number;
    gap_to_previous: number;
  },
) {
  const duration = endOffset - startOffset;
  return {
    event_id: eventId,
    region_id: "region-a",
    source_sequence_key: "hydrophone:rpi_orcasound_lab",
    sequence_index: startOffset,
    start_timestamp: JOB_START + startOffset,
    end_timestamp: JOB_START + endOffset,
    token_id: tokenId,
    token_label: tokenLabel,
    token_confidence: 0.8,
    distance_to_centroid: 0.1,
    second_centroid_distance: 0.3,
    descriptor_values: {
      duration,
      ...descriptors,
    },
    descriptor_vector_values: {
      duration,
      ...descriptors,
    },
  };
}

async function setupMocks(page: Page): Promise<MockState> {
  const state: MockState = { timelineRequests: [], audioRequests: [] };

  await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) => {
    state.audioRequests.push(route.request().url());
    return route.fulfill({
      status: 200,
      contentType: "audio/wav",
      body: SILENT_WAV,
    });
  });

  await page.route("**/call-parsing/region-jobs/*/tile**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5g5r0ZQAAAABJRU5ErkJggg==",
        "base64",
      ),
    }),
  );

  await page.route("**/sequence-models/event-encoders**", (route) => {
    const url = route.request().url();
    const idMatch = url.match(/\/event-encoders\/([^/?#]+)/);

    if (!idMatch) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([COMPLETE_JOB, QUEUED_JOB]),
      });
    }

    const id = idMatch[1];
    if (url.includes("/timeline")) {
      state.timelineRequests.push(url);
      const selectedK = new URL(url).searchParams.get("k");
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(selectedK === "100" ? TIMELINE_100 : TIMELINE_50),
      });
    }

    if (url.includes("/projection")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: COMPLETE_JOB.id,
          selected_k: 50,
          valid_k_values: [50, 100],
          method: "umap",
          x_axis_label: "UMAP 1",
          y_axis_label: "UMAP 2",
          points: [],
        }),
      });
    }

    if (id === COMPLETE_JOB.id) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(COMPLETE_DETAIL),
      });
    }

    if (id === QUEUED_JOB.id) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(QUEUED_DETAIL),
      });
    }

    return route.fulfill({ status: 404 });
  });

  return state;
}

async function clickFirstEvent(page: Page) {
  const canvas = page.getByTestId("eej-piano-roll-canvas");
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  const width = box?.width ?? 1;
  const height = box?.height ?? 1;
  const x = 62 + (11 / 120) * (width - 72);
  const y = 8 + (1 - 480 / 2000) * (height - 32);
  await canvas.click({ position: { x, y } });
}

test.describe("Sequence Models - Event Encoder Piano Roll", () => {
  test("renders toolbar, canvas, minimap, legend, and k selector", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    await expect(page.getByTestId("eej-piano-roll-page")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-event-count")).toHaveText("5");
    await expect(page.getByTestId("eej-piano-roll-token-count")).toHaveText("2");
    await expect(page.getByTestId("eej-piano-roll-k-select")).toHaveValue("50");
    await expect(page.getByTestId("eej-piano-roll-play")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-minimap")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-legend-body")).toBeVisible();

    const canvasBox = await page.getByTestId("eej-piano-roll-canvas").boundingBox();
    expect(canvasBox?.width ?? 0).toBeGreaterThan(500);
    expect(canvasBox?.height ?? 0).toBeGreaterThan(300);
  });

  test("loading, not-found, and incomplete states render", async ({ page }) => {
    await setupMocks(page);

    await page.goto("/app/sequence-models/event-encoder/missing/piano-roll");
    await expect(page.getByTestId("eej-piano-roll-error")).toBeVisible();

    await page.goto(
      `/app/sequence-models/event-encoder/${QUEUED_JOB.id}/piano-roll`,
    );
    await expect(page.getByTestId("eej-piano-roll-unavailable")).toBeVisible();
  });

  test("k selector refetches timeline data", async ({ page }) => {
    const state = await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    await page.getByTestId("eej-piano-roll-k-select").selectOption("100");
    await expect.poll(() =>
      state.timelineRequests.some((url) => url.includes("k=100")),
    ).toBe(true);
  });

  test("selects events, filters legend tokens, and clears with Escape", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    await clickFirstEvent(page);
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-selected-event",
      "evt-a",
    );

    await page.getByTestId("eej-piano-roll-token-3").click();
    await expect(page.getByTestId("eej-piano-roll-token-3")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-token-filter",
      "3",
    );

    await page.keyboard.press("Escape");
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-selected-event",
      "",
    );
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-token-filter",
      "",
    );
  });

  test("legend toggles, minimap centers the viewport, and focused selects suppress shortcuts", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    await page.getByTestId("eej-piano-roll-legend-toggle").click();
    await expect(page.getByTestId("eej-piano-roll-legend-body")).toHaveCount(0);
    await page.getByTestId("eej-piano-roll-legend-toggle").click();
    await expect(page.getByTestId("eej-piano-roll-legend-body")).toBeVisible();

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    const beforeMinimap = await canvas.getAttribute("data-view-start");
    await page
      .getByTestId("eej-piano-roll-minimap")
      .click({ position: { x: 220, y: 20 } });
    await expect(canvas).not.toHaveAttribute("data-view-start", beforeMinimap ?? "");

    const beforeKeyboard = await canvas.getAttribute("data-view-start");
    await page.getByTestId("eej-piano-roll-k-select").focus();
    await page.keyboard.press("f");
    await expect(canvas).toHaveAttribute("data-view-start", beforeKeyboard ?? "");
  });

  test("playback uses selected event audio and back link returns to detail", async ({
    page,
  }) => {
    const state = await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    await clickFirstEvent(page);
    const audioRequest = page.waitForRequest((request) =>
      request.url().includes("/call-parsing/region-jobs/") &&
      request.url().includes("/audio-slice"),
    );
    await page.getByTestId("eej-piano-roll-play").click();
    await audioRequest;

    await expect(page.getByTestId("eej-piano-roll-audio")).toHaveAttribute(
      "src",
      /start_timestamp=1751644810.*duration_sec=2/,
    );
    expect(state.audioRequests.length).toBeGreaterThan(0);

    await page.getByTestId("eej-piano-roll-back").click();
    await expect(page).toHaveURL(
      new RegExp(`/app/sequence-models/event-encoder/${COMPLETE_JOB.id}$`),
    );
    await expect(page.getByTestId("eej-detail-page")).toBeVisible();
  });
});
