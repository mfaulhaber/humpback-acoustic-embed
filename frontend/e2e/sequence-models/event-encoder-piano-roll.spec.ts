import { expect, test, type Page } from "@playwright/test";

const SILENT_WAV = buildSilentWav(5);

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

const RIDGE_JOB = {
  ...COMPLETE_JOB,
  id: "eej-piano-roll-ridge",
  tokenizer_version: "crnn-event-encoder-v3",
  tokenization_signature: "piano-roll-ridge-token-sig",
  event_vector_dim: 150,
  event_vectors_path: "/tmp/event-encoders/eej-piano-roll-ridge/event_vectors.parquet",
  event_tokens_path: "/tmp/event-encoders/eej-piano-roll-ridge/event_tokens.parquet",
  token_sequences_path:
    "/tmp/event-encoders/eej-piano-roll-ridge/token_sequences.parquet",
  manifest_path: "/tmp/event-encoders/eej-piano-roll-ridge/manifest.json",
  report_path: "/tmp/event-encoders/eej-piano-roll-ridge/report.json",
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

const RIDGE_DETAIL = {
  job: RIDGE_JOB,
  manifest: {
    ...COMPLETE_DETAIL.manifest,
    job_id: RIDGE_JOB.id,
  },
  report: COMPLETE_DETAIL.report,
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

const TIMELINE_RIDGE = {
  ...TIMELINE_50,
  job_id: RIDGE_JOB.id,
  descriptor_feature_names: [
    ...TIMELINE_50.descriptor_feature_names,
    "ridge_median_frequency",
    "ridge_low_frequency",
    "ridge_high_frequency",
    "ridge_frequency_span",
    "ridge_coverage",
    "ridge_energy_ratio",
    "band_limited_peak_frequency",
    "high_band_energy_ratio",
  ],
  descriptor_feature_units: {
    ...TIMELINE_50.descriptor_feature_units,
    ridge_median_frequency: "Hz",
    ridge_low_frequency: "Hz",
    ridge_high_frequency: "Hz",
    ridge_frequency_span: "Hz",
    ridge_coverage: "ratio",
    ridge_energy_ratio: "ratio",
    band_limited_peak_frequency: "Hz",
    high_band_energy_ratio: "ratio",
  },
  events: [
    timelineEvent("ridge-low", 2, "T02", 10, 11.2, {
      median_f0: 430,
      f0_range: 40,
      peak_frequency: 430,
      voicing_fraction: 0.9,
      ridge_log_frequency_slope: 0.1,
      pulse_rate: 0,
      gap_to_previous: 0,
      ridge_median_frequency: 430,
      ridge_low_frequency: 400,
      ridge_high_frequency: 470,
      ridge_frequency_span: 70,
      ridge_coverage: 0.9,
      ridge_energy_ratio: 0.4,
      band_limited_peak_frequency: 430,
      high_band_energy_ratio: 0.05,
    }),
    timelineEvent("ridge-high", 12, "T12", 18, 19.2, {
      median_f0: 0,
      f0_range: 0,
      peak_frequency: 62.5,
      voicing_fraction: 0,
      ridge_log_frequency_slope: 1.4,
      pulse_rate: 0,
      gap_to_previous: 6.8,
      ridge_median_frequency: 2600,
      ridge_low_frequency: 2300,
      ridge_high_frequency: 2950,
      ridge_frequency_span: 650,
      ridge_coverage: 0.82,
      ridge_energy_ratio: 0.22,
      band_limited_peak_frequency: 2650,
      high_band_energy_ratio: 0.88,
    }),
  ],
};

interface MockState {
  timelineRequests: string[];
  audioRequests: string[];
  tileRequests: string[];
}

function buildSilentWav(seconds: number) {
  const sampleRate = 8000;
  const bytesPerSample = 2;
  const dataLength = sampleRate * seconds * bytesPerSample;
  const buffer = Buffer.alloc(44 + dataLength);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataLength, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * bytesPerSample, 28);
  buffer.writeUInt16LE(bytesPerSample, 32);
  buffer.writeUInt16LE(8 * bytesPerSample, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataLength, 40);
  return buffer;
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
    ridge_median_frequency?: number;
    ridge_low_frequency?: number;
    ridge_high_frequency?: number;
    ridge_frequency_span?: number;
    ridge_coverage?: number;
    ridge_energy_ratio?: number;
    band_limited_peak_frequency?: number;
    high_band_energy_ratio?: number;
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
  const state: MockState = {
    timelineRequests: [],
    audioRequests: [],
    tileRequests: [],
  };

  await page.addInitScript(() => {
    const nativeAudio = window.Audio;
    const captured: HTMLAudioElement[] = [];
    (
      window as Window & { __pianoRollAudioElements?: HTMLAudioElement[] }
    ).__pianoRollAudioElements = captured;
    function AudioShim(src?: string) {
      const audio = new nativeAudio(src);
      captured.push(audio);
      return audio;
    }
    AudioShim.prototype = nativeAudio.prototype;
    window.Audio = AudioShim as typeof Audio;
  });

  await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) => {
    state.audioRequests.push(route.request().url());
    return route.fulfill({
      status: 200,
      contentType: "audio/wav",
      body: SILENT_WAV,
    });
  });

  await page.route("**/call-parsing/region-jobs/*/tile**", (route) => {
    state.tileRequests.push(route.request().url());
    return route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5g5r0ZQAAAABJRU5ErkJggg==",
        "base64",
      ),
    });
  });

  await page.route("**/sequence-models/event-encoders**", (route) => {
    const url = route.request().url();
    const idMatch = url.match(/\/event-encoders\/([^/?#]+)/);

    if (!idMatch) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([COMPLETE_JOB, RIDGE_JOB, QUEUED_JOB]),
      });
    }

    const id = idMatch[1];
    if (url.includes("/timeline")) {
      state.timelineRequests.push(url);
      const selectedK = new URL(url).searchParams.get("k");
      if (id === RIDGE_JOB.id) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(TIMELINE_RIDGE),
        });
      }
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

    if (id === RIDGE_JOB.id) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(RIDGE_DETAIL),
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

async function eventPoint(
  page: Page,
  eventCenter: number,
  centerFrequency: number,
  frequencyMax = 2000,
) {
  const canvas = page.getByTestId("eej-piano-roll-canvas");
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  const viewStart = Number(await canvas.getAttribute("data-view-start"));
  const viewEnd = Number(await canvas.getAttribute("data-view-end"));
  const width = box?.width ?? 1;
  const height = box?.height ?? 1;
  const x =
    62 + ((eventCenter - viewStart) / (viewEnd - viewStart)) * (width - 72);
  const y = 8 + (1 - centerFrequency / frequencyMax) * (height - 16);
  return { box, x, y };
}

async function clickFirstEvent(page: Page) {
  const canvas = page.getByTestId("eej-piano-roll-canvas");
  const { x, y } = await eventPoint(page, JOB_START + 11, 480);
  await canvas.click({ position: { x, y } });
}

async function visibleCenterTime(page: Page) {
  const canvas = page.getByTestId("eej-piano-roll-canvas");
  const start = Number(await canvas.getAttribute("data-view-start"));
  const end = Number(await canvas.getAttribute("data-view-end"));
  return (start + end) / 2;
}

async function setCapturedAudioCurrentTime(
  page: Page,
  srcNeedle: string,
  currentTime: number,
) {
  await page.waitForFunction((needle) => {
    const audios = (
      window as Window & { __pianoRollAudioElements?: HTMLAudioElement[] }
    ).__pianoRollAudioElements ?? [];
    return audios.some((audio) => audio.src.includes(needle));
  }, srcNeedle);
  await page.evaluate(
    ({ needle, value }) => {
      const audios = (
        window as Window & { __pianoRollAudioElements?: HTMLAudioElement[] }
      ).__pianoRollAudioElements ?? [];
      const audio = audios.find((item) => item.src.includes(needle));
      if (!audio) throw new Error(`No captured audio for ${needle}`);
      Object.defineProperty(audio, "currentTime", {
        configurable: true,
        value,
      });
    },
    { needle: srcNeedle, value: currentTime },
  );
}

test.describe("Sequence Models - Event Encoder Piano Roll", () => {
  test("renders toolbar, canvas, bottom spectrogram, legend, and k selector", async ({
    page,
  }) => {
    const state = await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    await expect(page.getByTestId("eej-piano-roll-page")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-event-count")).toHaveText("5");
    await expect(page.getByTestId("eej-piano-roll-token-count")).toHaveText("2");
    await expect(page.getByTestId("eej-piano-roll-duration")).toHaveText("2:40");
    await expect(page.getByTestId("eej-piano-roll-k-select")).toHaveValue("50");
    await expect(page.getByTestId("eej-piano-roll-y-mode")).toHaveValue("f0");
    await expect(page.getByTestId("eej-piano-roll-play")).toBeVisible();
    await expect(
      page.getByTestId("eej-piano-roll-spectrogram-strip"),
    ).toBeVisible();
    await expect(
      page.getByTestId("eej-piano-roll-spectrogram-lod"),
    ).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-minimap")).toHaveCount(0);
    await expect(page.getByTestId("eej-piano-roll-legend-body")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-view-start",
      `${JOB_START - 20}.000`,
    );
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-view-end",
      `${JOB_START + 140}.000`,
    );
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-playhead-time",
      `${JOB_START + 60}.000`,
    );
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-cursor-state",
      "idle",
    );

    const canvasBox = await page.getByTestId("eej-piano-roll-canvas").boundingBox();
    const stripBox = await page
      .getByTestId("eej-piano-roll-spectrogram-strip")
      .boundingBox();
    expect(canvasBox?.width ?? 0).toBeGreaterThan(500);
    expect(canvasBox?.height ?? 0).toBeGreaterThan(300);
    expect(stripBox?.height ?? 0).toBeGreaterThanOrEqual(150);
    expect(stripBox?.y ?? 0).toBeGreaterThan(
      (canvasBox?.y ?? 0) + (canvasBox?.height ?? 0) - 1,
    );
    await expect
      .poll(() =>
        state.tileRequests.some(
          (url) => url.includes("freq_min=0") && url.includes("freq_max=2000"),
        ),
      )
      .toBe(true);
  });

  test("v3 ridge descriptors render high-frequency tokens in Ridge mode", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${RIDGE_JOB.id}/piano-roll`,
    );

    await expect(page.getByTestId("eej-piano-roll-y-mode")).toHaveValue("ridge");
    await page.getByTestId("eej-piano-roll-frequency-max").selectOption("6000");
    await expect(page.getByTestId("eej-piano-roll-frequency-max")).toHaveValue(
      "6000",
    );

    const high = await eventPoint(page, JOB_START + 18.6, 2600, 6000);
    const low = await eventPoint(page, JOB_START + 10.6, 430, 6000);
    expect(high.y).toBeLessThan(low.y);

    await page
      .getByTestId("eej-piano-roll-canvas")
      .click({ position: { x: high.x, y: high.y } });
    const tooltip = page.getByTestId("eej-piano-roll-tooltip");
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText("ridge_mid");
    await expect(tooltip).toContainText("ridge_band");
    await expect(tooltip).toContainText("band_peak");
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

    await page.keyboard.press("d");
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-selected-event",
      "evt-b",
    );
    await expect
      .poll(async () =>
        Math.abs((await visibleCenterTime(page)) - (JOB_START + 28.7)),
      )
      .toBeLessThan(0.02);

    await page.keyboard.press("a");
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-selected-event",
      "evt-a",
    );
    await expect
      .poll(async () =>
        Math.abs((await visibleCenterTime(page)) - (JOB_START + 11)),
      )
      .toBeLessThan(0.02);

    await page
      .getByTestId("eej-piano-roll-canvas")
      .click({ position: { x: 66, y: 18 } });
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-selected-event",
      "",
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

  test("legend toggles, spectrogram collapses, and focused selects suppress shortcuts", async ({
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
    await expect(canvas).toBeVisible();
    const beforeCollapseBox = await canvas.boundingBox();
    await page.getByTestId("eej-piano-roll-spectrogram-toggle").click();
    await expect(
      page.getByTestId("eej-piano-roll-spectrogram-strip"),
    ).toHaveCount(0);
    await expect
      .poll(async () => {
        const afterCollapseBox = await canvas.boundingBox();
        return afterCollapseBox?.height ?? 0;
      })
      .toBeGreaterThan(beforeCollapseBox?.height ?? 0);
    await page.getByTestId("eej-piano-roll-spectrogram-toggle").click();
    await expect(
      page.getByTestId("eej-piano-roll-spectrogram-strip"),
    ).toBeVisible();

    await page.keyboard.press("=");
    await expect
      .poll(async () => {
        const start = Number(await canvas.getAttribute("data-view-start"));
        const end = Number(await canvas.getAttribute("data-view-end"));
        return end - start;
      })
      .toBeLessThan(160);
    const beforeKeyboard = await canvas.getAttribute("data-view-start");
    await page.getByTestId("eej-piano-roll-k-select").focus();
    await page.keyboard.press("f");
    await expect(canvas).toHaveAttribute("data-view-start", beforeKeyboard ?? "");
  });

  test("spectrogram strip zoom and drag update the shared piano roll viewport", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    const strip = page.getByTestId("eej-piano-roll-spectrogram-strip");
    await expect(strip).toBeVisible();
    const stripBox = await strip.boundingBox();
    expect(stripBox).not.toBeNull();

    await page.mouse.move(
      (stripBox?.x ?? 0) + (stripBox?.width ?? 0) / 2,
      (stripBox?.y ?? 0) + (stripBox?.height ?? 0) / 2,
    );
    await page.mouse.wheel(0, -500);
    await expect
      .poll(async () => {
        const start = Number(await canvas.getAttribute("data-view-start"));
        const end = Number(await canvas.getAttribute("data-view-end"));
        return end - start;
      })
      .toBeLessThan(160);

    const beforeDrag = await canvas.getAttribute("data-view-start");
    await page.mouse.down();
    await page.mouse.move(
      (stripBox?.x ?? 0) + (stripBox?.width ?? 0) / 2 + 120,
      (stripBox?.y ?? 0) + (stripBox?.height ?? 0) / 2,
    );
    await page.mouse.up();
    await expect(canvas).not.toHaveAttribute("data-view-start", beforeDrag ?? "");
  });

  test("canvas cursor, drag panning, and tooltip placement stay usable", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    await expect(canvas).toBeVisible();
    const canvasBox = await canvas.boundingBox();
    expect(canvasBox).not.toBeNull();
    await page.mouse.move(
      (canvasBox?.x ?? 0) + (canvasBox?.width ?? 0) / 2,
      (canvasBox?.y ?? 0) + (canvasBox?.height ?? 0) / 2,
    );
    await page.mouse.wheel(0, -500);
    await expect
      .poll(async () => {
        const start = Number(await canvas.getAttribute("data-view-start"));
        const end = Number(await canvas.getAttribute("data-view-end"));
        return end - start;
      })
      .toBeLessThan(160);
    const beforeDrag = await canvas.getAttribute("data-view-start");
    const first = await eventPoint(page, JOB_START + 11, 480);

    await page.mouse.move(first.box.x + first.x, first.box.y + first.y);
    await expect(canvas).toHaveAttribute("data-cursor-state", "hover-token");
    await expect(canvas).toHaveClass(/cursor-pointer/);

    const tooltip = page.getByTestId("eej-piano-roll-tooltip");
    await expect(tooltip).toBeVisible();
    const tooltipBox = await tooltip.boundingBox();
    const viewport = page.viewportSize();
    expect(tooltipBox).not.toBeNull();
    expect(viewport).not.toBeNull();
    expect((tooltipBox?.x ?? 0) + (tooltipBox?.width ?? 0)).toBeLessThanOrEqual(
      viewport?.width ?? 0,
    );
    expect((tooltipBox?.y ?? 0) + (tooltipBox?.height ?? 0)).toBeLessThanOrEqual(
      viewport?.height ?? 0,
    );

    await page.mouse.down();
    await expect(canvas).toHaveAttribute("data-cursor-state", "dragging");
    await expect(canvas).toHaveClass(/cursor-grabbing/);
    await page.mouse.move(first.box.x + first.x + 140, first.box.y + first.y);
    await page.mouse.up();
    await expect(canvas).not.toHaveAttribute("data-cursor-state", "dragging");
    await expect(canvas).not.toHaveAttribute("data-view-start", beforeDrag ?? "");
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
    const request = await audioRequest;
    expect(request.url()).toMatch(/start_timestamp=1751644810/);
    expect(request.url()).toMatch(/duration_sec=2/);

    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-playback-mode",
      "selected",
    );
    expect(state.audioRequests.length).toBeGreaterThan(0);

    await setCapturedAudioCurrentTime(page, "start_timestamp=1751644810", 1.2);
    await expect(page.getByTestId("eej-piano-roll-canvas")).toHaveAttribute(
      "data-playhead-time",
      /1751644811\.200/,
    );
    await expect
      .poll(async () => {
        const canvas = page.getByTestId("eej-piano-roll-canvas");
        const start = Number(await canvas.getAttribute("data-view-start"));
        const end = Number(await canvas.getAttribute("data-view-end"));
        const playhead = Number(await canvas.getAttribute("data-playhead-time"));
        return Math.abs((start + end) / 2 - playhead);
      })
      .toBeLessThan(0.02);

    await page.getByTestId("eej-piano-roll-back").click();
    await expect(page).toHaveURL(
      new RegExp(`/app/sequence-models/event-encoder/${COMPLETE_JOB.id}$`),
    );
    await expect(page.getByTestId("eej-detail-page")).toBeVisible();
  });

  test("playback keeps the playhead centered while audio advances", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    await expect(canvas).toBeVisible();
    for (let i = 0; i < 10; i += 1) {
      await page.keyboard.press("=");
    }
    const beforeStart = Number(await canvas.getAttribute("data-view-start"));
    const playheadBefore = Number(await canvas.getAttribute("data-playhead-time"));

    const audioRequest = page.waitForRequest((request) =>
      request.url().includes("/call-parsing/region-jobs/") &&
      request.url().includes("/audio-slice"),
    );
    await page.getByTestId("eej-piano-roll-play").click();
    const request = await audioRequest;
    const requestParams = new URL(request.url()).searchParams;
    expect(Number(requestParams.get("start_timestamp"))).toBeCloseTo(
      playheadBefore,
      3,
    );
    expect(requestParams.get("duration_sec")).toBe("300");
    await expect(canvas).toHaveAttribute("data-playback-mode", "continuous");

    await setCapturedAudioCurrentTime(page, "duration_sec=300", 3);

    await expect
      .poll(async () => {
        const start = Number(await canvas.getAttribute("data-view-start"));
        const end = Number(await canvas.getAttribute("data-view-end"));
        const playhead = Number(await canvas.getAttribute("data-playhead-time"));
        return Math.abs((start + end) / 2 - playhead);
      })
      .toBeLessThan(0.02);
    await expect
      .poll(async () => Number(await canvas.getAttribute("data-view-start")))
      .toBeGreaterThan(beforeStart);
    await expect(canvas).toHaveAttribute("data-playhead-time", /1751644863/);
  });
});
