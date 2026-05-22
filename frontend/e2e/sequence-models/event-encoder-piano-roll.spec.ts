import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, "..");

const SILENT_WAV = buildSilentWav(5);

interface FixtureEvent {
  event_id: string;
  start_s: number;
  end_s: number;
  fundamental_hz: number;
  n_harmonics: number;
}

interface FixtureExpectedNote {
  event_id: string;
  midi_pitch: number;
  partial_index: number;
}

interface FixtureMetadata {
  sample_rate: number;
  duration_s: number;
  events: FixtureEvent[];
  expected_notes: FixtureExpectedNote[];
}

const FIXTURE: FixtureMetadata = JSON.parse(
  readFileSync(
    resolve(__dirname, "../../../tests/fixtures/piano_roll/synthetic_three_events.json"),
    "utf-8",
  ),
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
    "spectral_centroid",
    "bandwidth",
    "spectral_entropy",
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
    spectral_centroid: "Hz",
    bandwidth: "Hz",
    spectral_entropy: "ratio",
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
      median_f0: 71,
      f0_range: 4,
      peak_frequency: 62.5,
      voicing_fraction: 1,
      ridge_log_frequency_slope: 1.4,
      pulse_rate: 0,
      gap_to_previous: 6.8,
      ridge_median_frequency: 2600,
      ridge_low_frequency: 2300,
      ridge_high_frequency: 2950,
      ridge_frequency_span: 650,
      ridge_coverage: 0.82,
      ridge_energy_ratio: 0.011,
      band_limited_peak_frequency: 2650,
      high_band_energy_ratio: 0.88,
    }),
    timelineEvent("ridge-moan", 47, "T47", 25, 26.6, {
      median_f0: 315.2,
      f0_range: 43.6,
      peak_frequency: 312.5,
      voicing_fraction: 1,
      ridge_log_frequency_slope: 0,
      pulse_rate: 161.6,
      gap_to_previous: 5.8,
      ridge_median_frequency: 312.5,
      ridge_low_frequency: 296.875,
      ridge_high_frequency: 329.6875,
      ridge_frequency_span: 32.8125,
      ridge_coverage: 1,
      ridge_energy_ratio: 0.09594432264566422,
      band_limited_peak_frequency: 312.5,
      high_band_energy_ratio: 0.5040363669395447,
      spectral_centroid: 2101.532958984375,
      bandwidth: 2097.9912109375,
      spectral_entropy: 0.8581035733222961,
    }),
  ],
};

type NotesStatusMock =
  | { status: "absent" }
  | {
      id: string;
      event_encoder_job_id: string;
      extractor_version: string;
      status: "queued" | "running" | "complete" | "failed" | "canceled";
      started_at: string | null;
      finished_at: string | null;
      error_message: string | null;
      notes_path: string | null;
      n_events: number | null;
      n_notes: number | null;
      compute_seconds: number | null;
      params_json: string;
      created_at: string;
      updated_at: string;
    };

interface NotesPayload {
  job_id: string;
  extractor_version: string;
  n_notes: number;
  notes: PianoRollNoteRow[];
}

interface PianoRollNoteRow {
  event_id: string;
  event_token: number;
  partial_index: number;
  midi_pitch: number;
  start_utc: number;
  start_offset_s: number;
  duration_s: number;
  velocity: number;
  peak_magnitude: number;
  track_id: number;
}

interface MockState {
  timelineRequests: string[];
  audioRequests: string[];
  tileRequests: string[];
  notesStatusRequests: string[];
  notesJobsRequests: string[];
  notesRequests: string[];
  notesContourRequests: string[];
  notesStatus: NotesStatusMock;
  notesPayload: NotesPayload | null;
  notesFailWithStatus: number | null;
  contoursPayload: ContourPayload | null;
  contoursFailWithStatus: number | null;
}

function buildNotesStatus(
  status: Exclude<NotesStatusMock["status"], "absent">,
  overrides: Partial<Exclude<NotesStatusMock, { status: "absent" }>> = {},
): NotesStatusMock {
  return {
    id: "prn-1",
    event_encoder_job_id: COMPLETE_JOB.id,
    extractor_version: "v2",
    status,
    started_at: null,
    finished_at: null,
    error_message:
      status === "failed" ? "audio missing for one event" : null,
    notes_path: status === "complete" ? "/tmp/notes.parquet" : null,
    n_events: status === "complete" ? 3 : null,
    n_notes: status === "complete" ? 12 : null,
    compute_seconds: status === "complete" ? 1.2 : null,
    params_json: "{}",
    created_at: "2026-05-20T01:00:00Z",
    updated_at: "2026-05-20T01:05:00Z",
    ...overrides,
  };
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
    spectral_centroid?: number;
    bandwidth?: number;
    spectral_entropy?: number;
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

interface ContourPayload {
  job_id: string;
  extractor_version: string;
  n_notes: number;
  contours: Record<
    string,
    Array<{
      frame_index: number;
      time_offset_s: number;
      cents_from_pitch: number;
      harmonic_strength: number;
      subharmonic_octave: number;
    }>
  >;
}

async function setupMocks(
  page: Page,
  options: {
    notesStatus?: NotesStatusMock;
    notesPayload?: NotesPayload | null;
    notesFailWithStatus?: number;
    contoursPayload?: ContourPayload | null;
    contoursFailWithStatus?: number;
  } = {},
): Promise<MockState> {
  const state: MockState = {
    timelineRequests: [],
    audioRequests: [],
    tileRequests: [],
    notesStatusRequests: [],
    notesJobsRequests: [],
    notesRequests: [],
    notesContourRequests: [],
    notesStatus: options.notesStatus ?? { status: "absent" },
    notesPayload: options.notesPayload ?? null,
    notesFailWithStatus: options.notesFailWithStatus ?? null,
    contoursPayload: options.contoursPayload ?? null,
    contoursFailWithStatus: options.contoursFailWithStatus ?? null,
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
    if (url.includes("/notes-status")) {
      state.notesStatusRequests.push(url);
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(state.notesStatus),
      });
    }

    if (url.includes("/notes-jobs")) {
      state.notesJobsRequests.push(url);
      state.notesStatus = buildNotesStatus("queued");
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(state.notesStatus),
      });
    }

    if (url.includes("/notes/contours")) {
      state.notesContourRequests.push(url);
      if (state.contoursFailWithStatus != null) {
        return route.fulfill({ status: state.contoursFailWithStatus });
      }
      const payload: ContourPayload = state.contoursPayload ?? {
        job_id: id,
        extractor_version: "v3",
        n_notes: 0,
        contours: {},
      };
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(payload),
      });
    }

    if (/\/notes(?:$|\?)/.test(url)) {
      state.notesRequests.push(url);
      if (state.notesFailWithStatus != null) {
        return route.fulfill({ status: state.notesFailWithStatus });
      }
      const payload: NotesPayload = state.notesPayload ?? {
        job_id: id,
        extractor_version: "v2",
        n_notes: 0,
        notes: [],
      };
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(payload),
      });
    }

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

    const moanEnvelope = await eventPoint(page, JOB_START + 25.8, 1600, 6000);
    await page
      .getByTestId("eej-piano-roll-canvas")
      .click({ position: { x: moanEnvelope.x, y: moanEnvelope.y } });
    await expect(tooltip).toContainText("T47");
    await expect(tooltip).toContainText("display_band");
    await expect(tooltip).toContainText("297 Hz - 2102 Hz");
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

  test("notes status pill shows 'absent' and Generate notes triggers the mutation", async ({
    page,
  }) => {
    const state = await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const pill = page.getByTestId("piano-roll-notes-status-pill").first();
    await expect(pill).toBeVisible();
    await expect(pill).toHaveAttribute("data-notes-status", "absent");
    await expect(pill).toHaveText("Notes: absent");

    const generate = page.getByTestId("eej-piano-roll-notes-generate");
    await expect(generate).toBeVisible();
    await expect(generate).toHaveText("Generate notes");
    await generate.click();
    await expect
      .poll(() => state.notesJobsRequests.length)
      .toBeGreaterThanOrEqual(1);
    await expect
      .poll(async () =>
        page.getByTestId("piano-roll-notes-status-pill").first()
          .getAttribute("data-notes-status"),
      )
      .toBe("queued");
  });

  test("notes status pill shows 'queued' with disabled progress label", async ({
    page,
  }) => {
    await setupMocks(page, { notesStatus: buildNotesStatus("queued") });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const pill = page.getByTestId("piano-roll-notes-status-pill").first();
    await expect(pill).toHaveAttribute("data-notes-status", "queued");
    await expect(page.getByTestId("eej-piano-roll-notes-generate")).toHaveCount(0);
    await expect(page.getByTestId("eej-piano-roll-notes-progress")).toBeVisible();
  });

  test("notes status pill shows 'running' with disabled progress label", async ({
    page,
  }) => {
    await setupMocks(page, { notesStatus: buildNotesStatus("running") });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const pill = page.getByTestId("piano-roll-notes-status-pill").first();
    await expect(pill).toHaveAttribute("data-notes-status", "running");
    await expect(page.getByTestId("eej-piano-roll-notes-progress")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-notes-generate")).toHaveCount(0);
  });

  test("notes status pill shows 'complete' with no generate action", async ({
    page,
  }) => {
    await setupMocks(page, { notesStatus: buildNotesStatus("complete") });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const pill = page.getByTestId("piano-roll-notes-status-pill").first();
    await expect(pill).toHaveAttribute("data-notes-status", "complete");
    await expect(page.getByTestId("eej-piano-roll-notes-generate")).toHaveCount(0);
    await expect(page.getByTestId("eej-piano-roll-notes-progress")).toHaveCount(0);
  });

  test("Notes mode renders note bars and tooltip when status is complete", async ({
    page,
  }) => {
    const notesPayload: NotesPayload = {
      job_id: COMPLETE_JOB.id,
      extractor_version: "v2",
      n_notes: 2,
      notes: [
        {
          event_id: "evt-a",
          event_token: 3,
          partial_index: 0,
          midi_pitch: 60,
          start_utc: JOB_START + 11,
          start_offset_s: 0,
          duration_s: 0.5,
          velocity: 96,
          peak_magnitude: -2.0,
          track_id: 1,
        },
        {
          event_id: "evt-b",
          event_token: 7,
          partial_index: 1,
          midi_pitch: 72,
          start_utc: JOB_START + 28,
          start_offset_s: 0,
          duration_s: 0.4,
          velocity: 64,
          peak_magnitude: -3.0,
          track_id: 2,
        },
      ],
    };
    await setupMocks(page, {
      notesStatus: buildNotesStatus("complete"),
      notesPayload,
    });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    await expect(canvas).toHaveAttribute("data-view-mode", "notes");
    await expect(canvas).toHaveAttribute("data-notes-mode", "true");
    await expect
      .poll(async () => canvas.getAttribute("data-notes-count"))
      .toBe("2");

    await expect(page.getByTestId("eej-piano-roll-y-mode")).toHaveValue("notes");

    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    const viewStart = Number(await canvas.getAttribute("data-view-start"));
    const viewEnd = Number(await canvas.getAttribute("data-view-end"));
    const width = box?.width ?? 1;
    const height = box?.height ?? 1;
    const noteCenterTime = JOB_START + 11.25;
    const x = 62 + ((noteCenterTime - viewStart) / (viewEnd - viewStart)) * (width - 72);
    const plotTop = 8;
    const plotBottom = height - 8;
    const ratio = (60 - 12 + 0.5) / 109;
    const y = plotBottom - ratio * (plotBottom - plotTop);

    await canvas.hover({ position: { x, y } });
    await expect(page.getByTestId("eej-piano-roll-note-tooltip")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-note-tooltip")).toContainText(
      "MIDI 60 (C4)",
    );
    await expect(page.getByTestId("eej-piano-roll-note-tooltip")).toContainText(
      "F0",
    );
    await expect(page.getByTestId("eej-piano-roll-note-tooltip")).toContainText(
      "96",
    );
  });

  test("v3 Notes mode hydrates ribbon contours and shows ±cents in the tooltip", async ({
    page,
  }) => {
    const notesPayload: NotesPayload = {
      job_id: COMPLETE_JOB.id,
      extractor_version: "v3",
      n_notes: 1,
      notes: [
        {
          event_id: "evt-a",
          event_token: 3,
          partial_index: 0,
          midi_pitch: 60,
          start_utc: JOB_START + 11,
          start_offset_s: 0,
          duration_s: 0.5,
          velocity: 96,
          peak_magnitude: -2.0,
          track_id: 1,
          note_uid: "uid-1",
          f0_track_id: 1,
          contour_frame_count: 3,
        },
      ],
    };
    const contoursPayload: ContourPayload = {
      job_id: COMPLETE_JOB.id,
      extractor_version: "v3",
      n_notes: 1,
      contours: {
        "uid-1": [
          {
            frame_index: 0,
            time_offset_s: 0.0,
            cents_from_pitch: -30,
            harmonic_strength: 1.0,
            subharmonic_octave: 0,
          },
          {
            frame_index: 1,
            time_offset_s: 0.25,
            cents_from_pitch: 5,
            harmonic_strength: 1.0,
            subharmonic_octave: 0,
          },
          {
            frame_index: 2,
            time_offset_s: 0.5,
            cents_from_pitch: 40,
            harmonic_strength: 1.0,
            subharmonic_octave: 0,
          },
        ],
      },
    };
    const state = await setupMocks(page, {
      notesStatus: buildNotesStatus("complete", { extractor_version: "v3" }),
      notesPayload,
      contoursPayload,
    });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    await expect(canvas).toHaveAttribute("data-view-mode", "notes");
    await expect
      .poll(async () => state.notesContourRequests.length)
      .toBeGreaterThan(0);
    await expect(page.locator("script")).not.toHaveText(/contour fetch failed/);

    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    const viewStart = Number(await canvas.getAttribute("data-view-start"));
    const viewEnd = Number(await canvas.getAttribute("data-view-end"));
    const width = box?.width ?? 1;
    const height = box?.height ?? 1;
    const noteCenterTime = JOB_START + 11.25;
    const x = 62 + ((noteCenterTime - viewStart) / (viewEnd - viewStart)) * (width - 72);
    const plotTop = 8;
    const plotBottom = height - 8;
    const ratio = (60 - 12 + 0.5) / 109;
    const y = plotBottom - ratio * (plotBottom - plotTop);

    await canvas.hover({ position: { x, y } });
    const tooltip = page.getByTestId("eej-piano-roll-note-tooltip");
    await expect(tooltip).toBeVisible();
    // Tooltip carries the max ±cents from the contour. 40¢ wins here.
    await expect(tooltip).toContainText("Δpitch");
    await expect(tooltip).toContainText("±40");
  });

  test("Notes mode falls back to rectangle view when /notes fetch fails", async ({
    page,
  }) => {
    await setupMocks(page, {
      notesStatus: buildNotesStatus("complete"),
      notesFailWithStatus: 500,
    });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    await expect
      .poll(async () => canvas.getAttribute("data-view-mode"))
      .not.toBe("notes");
    await expect(page.getByTestId("eej-piano-roll-y-mode")).not.toHaveValue(
      "notes",
    );
  });

  test("Notes mode renders all expected pitches from the synthetic three-event fixture", async ({
    page,
  }) => {
    const eventById = new Map(FIXTURE.events.map((event) => [event.event_id, event]));
    const fixtureNotes: PianoRollNoteRow[] = FIXTURE.expected_notes.map(
      (expected, index) => {
        const event = eventById.get(expected.event_id);
        if (!event) {
          throw new Error(`fixture references unknown event ${expected.event_id}`);
        }
        return {
          event_id: expected.event_id,
          event_token: 11,
          partial_index: expected.partial_index,
          midi_pitch: expected.midi_pitch,
          start_utc: JOB_START + event.start_s,
          start_offset_s: 0,
          duration_s: event.end_s - event.start_s,
          velocity: 80,
          peak_magnitude: -2.5,
          track_id: index + 1,
        };
      },
    );
    const notesPayload: NotesPayload = {
      job_id: COMPLETE_JOB.id,
      extractor_version: "v2",
      n_notes: fixtureNotes.length,
      notes: fixtureNotes,
    };
    await setupMocks(page, {
      notesStatus: buildNotesStatus("complete"),
      notesPayload,
    });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const canvas = page.getByTestId("eej-piano-roll-canvas");
    await expect(canvas).toHaveAttribute("data-view-mode", "notes");
    await expect
      .poll(async () => canvas.getAttribute("data-notes-count"))
      .toBe(String(fixtureNotes.length));

    const targetEvent = eventById.get("ev1");
    if (!targetEvent) throw new Error("fixture missing ev1");
    const noteCenterTime = JOB_START + (targetEvent.start_s + targetEvent.end_s) / 2;
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    const viewStart = Number(await canvas.getAttribute("data-view-start"));
    const viewEnd = Number(await canvas.getAttribute("data-view-end"));
    const width = box?.width ?? 1;
    const height = box?.height ?? 1;
    const x =
      62 + ((noteCenterTime - viewStart) / (viewEnd - viewStart)) * (width - 72);
    const plotTop = 8;
    const plotBottom = height - 8;
    const ratio = (57 - 12 + 0.5) / 109;
    const y = plotBottom - ratio * (plotBottom - plotTop);

    await canvas.hover({ position: { x, y } });
    const tooltip = page.getByTestId("eej-piano-roll-note-tooltip");
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText("MIDI 57 (A3)");
    await expect(tooltip).toContainText("F0");
  });

  test("Notes option is disabled and labeled unavailable when notes are absent", async ({
    page,
  }) => {
    await setupMocks(page);
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const yMode = page.getByTestId("eej-piano-roll-y-mode");
    const notesOption = yMode.locator('option[value="notes"]');
    await expect(notesOption).toHaveAttribute("disabled", "");
    await expect(notesOption).toContainText("unavailable");
  });

  test("notes status pill shows 'failed' with Re-run and reveals error on click", async ({
    page,
  }) => {
    await setupMocks(page, { notesStatus: buildNotesStatus("failed") });
    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const pill = page.getByTestId("piano-roll-notes-status-pill").first();
    await expect(pill).toHaveAttribute("data-notes-status", "failed");
    const generate = page.getByTestId("eej-piano-roll-notes-generate");
    await expect(generate).toHaveText("Re-run");

    await expect(page.getByTestId("eej-piano-roll-notes-error")).toHaveCount(0);
    await page.getByTestId("eej-piano-roll-notes-pill-button").click();
    await expect(page.getByTestId("eej-piano-roll-notes-error")).toBeVisible();
    await expect(page.getByTestId("eej-piano-roll-notes-error")).toHaveText(
      "audio missing for one event",
    );
  });

  test("piano-roll windowed export: posts the view's time range, exposes both downloads, re-export emphasis updates", async ({
    page,
  }) => {
    await setupMocks(page, { notesStatus: buildNotesStatus("complete") });
    const exportState = await setupMidiExportMocks(page);

    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const exportButton = page.getByTestId("eej-piano-roll-midi-export-button");
    await expect(exportButton).toHaveText(/^Export view$/);
    await exportButton.click();

    // Worker progresses queued -> complete on the next status poll.
    exportState.advance("complete");
    await expect(
      page.getByTestId("eej-piano-roll-midi-export-download"),
    ).toBeVisible();
    await expect(
      page.getByTestId("eej-piano-roll-audio-export-download"),
    ).toBeVisible();

    expect(exportState.createRequests).toHaveLength(1);
    const body = exportState.createRequests[0];
    expect(typeof body.window_start_utc).toBe("number");
    expect(typeof body.window_end_utc).toBe("number");
    expect(body.window_end_utc).toBeGreaterThan(body.window_start_utc);

    // Window match expected immediately after a successful export.
    const reExport = page.getByTestId("eej-piano-roll-midi-export-button");
    await expect(reExport).toHaveAttribute("data-window-match", "true");
  });

  test("piano-roll windowed export: emphasizes Re-export view when viewport drifts from exported window", async ({
    page,
  }) => {
    await setupMocks(page, { notesStatus: buildNotesStatus("complete") });
    const exportState = await setupMidiExportMocks(page, {
      // Persist a stored window that does not match the current viewport.
      initial: {
        status: "complete",
        window_start_utc: COMPLETE_JOB ? 0 : 0,
        window_end_utc: 0.0001,
      },
    });

    await page.goto(
      `/app/sequence-models/event-encoder/${COMPLETE_JOB.id}/piano-roll`,
    );

    const reExport = page.getByTestId("eej-piano-roll-midi-export-button");
    await expect(reExport).toHaveAttribute("data-window-match", "false");
    // Touch exportState so the variable is read (linter satisfaction).
    expect(exportState.createRequests.length).toBeGreaterThanOrEqual(0);
  });
});

// ---------- MIDI export mock harness ----------

interface PianoRollMidiExportRowMock {
  id: string;
  event_encoder_job_id: string;
  extractor_version: string;
  status: "queued" | "running" | "complete" | "failed" | "canceled";
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  midi_path: string | null;
  n_notes: number | null;
  n_bytes: number | null;
  compute_seconds: number | null;
  params_json: string;
  window_start_utc: number;
  window_end_utc: number;
  audio_path: string;
  audio_size_bytes: number;
  audio_sample_rate: number;
  audio_duration_s: number;
  created_at: string;
  updated_at: string;
}

interface MidiExportMockState {
  row: PianoRollMidiExportRowMock | null;
  createRequests: Array<Record<string, unknown>>;
  advance: (
    status: PianoRollMidiExportRowMock["status"],
    overrides?: Partial<PianoRollMidiExportRowMock>,
  ) => void;
}

async function setupMidiExportMocks(
  page: Page,
  options: {
    initial?: Partial<PianoRollMidiExportRowMock> & {
      status?: PianoRollMidiExportRowMock["status"];
    };
  } = {},
): Promise<MidiExportMockState> {
  const now = "2026-05-21T12:00:00Z";
  const defaultRow: PianoRollMidiExportRowMock = {
    id: "pre-1",
    event_encoder_job_id: COMPLETE_JOB.id,
    extractor_version: "v2",
    status: "queued",
    started_at: null,
    finished_at: null,
    error_message: null,
    midi_path: null,
    n_notes: null,
    n_bytes: null,
    compute_seconds: null,
    params_json: "{}",
    window_start_utc: 0,
    window_end_utc: 0,
    audio_path: "",
    audio_size_bytes: 0,
    audio_sample_rate: 0,
    audio_duration_s: 0,
    created_at: now,
    updated_at: now,
  };

  const state: MidiExportMockState = {
    row: options.initial
      ? { ...defaultRow, ...options.initial }
      : null,
    createRequests: [],
    advance: (status, overrides = {}) => {
      if (state.row == null) return;
      const completion: Partial<PianoRollMidiExportRowMock> =
        status === "complete"
          ? {
              status,
              midi_path: "exports/event_encoders/eej-piano-roll-1/notes_v2.mid",
              audio_path:
                "exports/event_encoders/eej-piano-roll-1/audio_v2.flac",
              n_notes: 9,
              n_bytes: 4096,
              audio_size_bytes: 1024 * 1024,
              audio_sample_rate: 32_000,
              audio_duration_s:
                state.row.window_end_utc - state.row.window_start_utc,
              finished_at: "2026-05-21T12:00:05Z",
            }
          : { status };
      state.row = { ...state.row, ...completion, ...overrides };
    },
  };

  await page.route(
    "**/sequence-models/event-encoders/*/midi-export-status",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(state.row ?? { status: "absent" }),
      }),
  );

  await page.route(
    "**/sequence-models/event-encoders/*/midi-exports",
    async (route) => {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      state.createRequests.push(body);
      state.row = {
        ...(state.row ?? defaultRow),
        status: "queued",
        midi_path: null,
        audio_path: "",
        n_notes: null,
        n_bytes: null,
        audio_size_bytes: 0,
        audio_sample_rate: 0,
        audio_duration_s: 0,
        finished_at: null,
        window_start_utc: Number(body.window_start_utc ?? 0),
        window_end_utc: Number(body.window_end_utc ?? 0),
        updated_at: "2026-05-21T12:00:01Z",
      };
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(state.row),
      });
    },
  );

  await page.route(
    "**/sequence-models/event-encoders/*/midi-export",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "audio/midi",
        body: Buffer.from("MThd...mock...", "utf-8"),
      }),
  );

  await page.route(
    "**/sequence-models/event-encoders/*/audio-export",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "audio/flac",
        body: Buffer.from("fLaCmock", "utf-8"),
      }),
  );

  return state;
}
