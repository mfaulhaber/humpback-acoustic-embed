import { expect, test } from "@playwright/test";

/**
 * Build a minimal SMF Type-1 file with `numTracks` empty tracks. Each track
 * contains only an end_of_track meta event. Used as the mocked
 * /midi-export response body so the test exercises a structurally valid
 * MIDI payload (matching the slim 7-channel layout produced by the
 * backend: tempo + 7 channel tracks = 8 total).
 */
function buildMinimalSMF(numTracks: number, ticksPerBeat = 480): Buffer {
  const header = Buffer.from([
    0x4d, 0x54, 0x68, 0x64, // "MThd"
    0x00, 0x00, 0x00, 0x06, // header length = 6
    0x00, 0x01, // format = 1
    (numTracks >> 8) & 0xff, numTracks & 0xff,
    (ticksPerBeat >> 8) & 0xff, ticksPerBeat & 0xff,
  ]);
  // Each MTrk chunk: 4-byte tag + 4-byte length + delta=0 + 0xFF 0x2F 0x00 (EOT)
  const trackChunk = Buffer.from([
    0x4d, 0x54, 0x72, 0x6b, // "MTrk"
    0x00, 0x00, 0x00, 0x04, // length = 4
    0x00, 0xff, 0x2f, 0x00, // delta=0; end_of_track meta
  ]);
  const tracks = Buffer.concat(Array.from({ length: numTracks }, () => trackChunk));
  return Buffer.concat([header, tracks]);
}

const MOCK_MIDI_BYTES = buildMinimalSMF(8); // tempo + 7 channel tracks

const JOB_ID = "eej-midi-export-1";
const REGION_JOB_ID = "rj-midi-export-1";
const JOB_START = 1_751_644_800;
const JOB_END = JOB_START + 60;

const COMPLETE_JOB = {
  id: JOB_ID,
  status: "complete",
  event_segmentation_job_id: "seg-midi-export-1",
  event_source_mode: "raw",
  continuous_embedding_job_id: "cej-midi-export-1",
  continuous_embedding_signature: "midi-export-sig",
  tokenizer_version: "crnn-event-encoder-v2",
  pooling_config_json: "{}",
  descriptor_config_json: "{}",
  preprocessing_config_json: "{}",
  k_values_json: "[50]",
  random_seed: 0,
  tokenization_signature: "midi-export-token-sig",
  event_vector_dim: 136,
  total_events: 1,
  encoded_events: 1,
  skipped_events: 0,
  event_vectors_path: `/tmp/event-encoders/${JOB_ID}/event_vectors.parquet`,
  event_tokens_path: `/tmp/event-encoders/${JOB_ID}/event_tokens.parquet`,
  token_sequences_path: `/tmp/event-encoders/${JOB_ID}/token_sequences.parquet`,
  manifest_path: `/tmp/event-encoders/${JOB_ID}/manifest.json`,
  report_path: `/tmp/event-encoders/${JOB_ID}/report.json`,
  error_message: null,
  created_at: "2026-05-20T01:00:00Z",
  updated_at: "2026-05-20T01:10:00Z",
};

const COMPLETE_DETAIL = {
  job: COMPLETE_JOB,
  manifest: {
    job_id: JOB_ID,
    valid_k_values: [50],
    total_events: 1,
    encoded_events: 1,
    skipped_events: 0,
  },
  report: {
    summary: {
      total_events: 1,
      encoded_events: 1,
      skipped_events: 0,
      valid_k_values: [50],
    },
    sequence_preview: { "50": ["T01"] },
  },
};

const TIMELINE = {
  job_id: JOB_ID,
  event_segmentation_job_id: COMPLETE_JOB.event_segmentation_job_id,
  event_source_mode: "raw",
  continuous_embedding_job_id: COMPLETE_JOB.continuous_embedding_job_id,
  region_detection_job_id: REGION_JOB_ID,
  selected_k: 50,
  valid_k_values: [50],
  descriptor_feature_names: ["duration", "peak_frequency"],
  descriptor_feature_units: { duration: "seconds", peak_frequency: "Hz" },
  job_start_timestamp: JOB_START,
  job_end_timestamp: JOB_END,
  events: [
    {
      event_id: "evt-a",
      event_index: 0,
      token_id: 1,
      token_label: "T01",
      start_offset_seconds: 1.0,
      end_offset_seconds: 2.0,
      gap_to_previous_seconds: 0,
      descriptor_values: { duration: 1.0, peak_frequency: 440 },
      descriptor_vector: [1.0, 440],
      event_vector_index: 0,
    },
  ],
  notes_status: { status: "absent" },
};

type NotesStatus =
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

type MidiExportStatus =
  | { status: "absent" }
  | {
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
      created_at: string;
      updated_at: string;
    };

function completeNotesStatus(): NotesStatus {
  return {
    id: "notes-job-1",
    event_encoder_job_id: JOB_ID,
    extractor_version: "v2",
    status: "complete",
    started_at: "2026-05-20T01:00:00Z",
    finished_at: "2026-05-20T01:01:00Z",
    error_message: null,
    notes_path: `event_encoders/${JOB_ID}/event_notes_v2.parquet`,
    n_events: 1,
    n_notes: 1,
    compute_seconds: 1.0,
    params_json: "{}",
    created_at: "2026-05-20T01:00:00Z",
    updated_at: "2026-05-20T01:01:00Z",
  };
}

function midiExportRow(
  state: "queued" | "running" | "complete" | "failed",
): MidiExportStatus {
  return {
    id: "midi-export-1",
    event_encoder_job_id: JOB_ID,
    extractor_version: "v2",
    status: state,
    started_at: state === "queued" ? null : "2026-05-20T01:02:00Z",
    finished_at: state === "complete" ? "2026-05-20T01:02:05Z" : null,
    error_message: state === "failed" ? "synth failed" : null,
    midi_path:
      state === "complete"
        ? `exports/event_encoders/${JOB_ID}/notes_v2.mid`
        : null,
    n_notes: state === "complete" ? 1 : null,
    n_bytes: state === "complete" ? 256 : null,
    compute_seconds: state === "complete" ? 0.1 : null,
    params_json: "{}",
    created_at: "2026-05-20T01:02:00Z",
    updated_at: "2026-05-20T01:02:05Z",
  };
}

interface MockState {
  notesStatus: NotesStatus;
  midiExportStatus: MidiExportStatus;
  midiExportPosts: Array<{ url: string; body: string }>;
}

async function mockRoutes(page: import("@playwright/test").Page, state: MockState) {
  await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) =>
    route.fulfill({ status: 200, contentType: "audio/wav", body: Buffer.alloc(44) }),
  );
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

  await page.route("**/sequence-models/event-encoders**", async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const idMatch = url.match(/\/event-encoders\/([^/?#]+)/);
    if (!idMatch) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([COMPLETE_JOB]),
      });
    }

    if (url.includes("/midi-export-status")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(state.midiExportStatus),
      });
    }
    if (url.includes("/midi-exports") && method === "POST") {
      const postData = route.request().postData() || "{}";
      state.midiExportPosts.push({ url, body: postData });
      state.midiExportStatus = midiExportRow("queued");
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(state.midiExportStatus),
      });
    }
    if (url.includes("/midi-export") && method === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "audio/midi",
        headers: {
          "content-disposition": `attachment; filename="event_encoder_${JOB_ID}_notes_v2.mid"`,
        },
        body: MOCK_MIDI_BYTES,
      });
    }
    if (url.includes("/notes-status")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(state.notesStatus),
      });
    }
    if (url.includes("/timeline")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ...TIMELINE,
          notes_status: state.notesStatus,
        }),
      });
    }
    if (url.endsWith(`/event-encoders/${idMatch[1]}`)) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(COMPLETE_DETAIL),
      });
    }
    return route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "unhandled" }),
    });
  });
}

test.describe("Event Encoder Piano Roll — MIDI export", () => {
  test("button disabled when notes are not complete", async ({ page }) => {
    const state: MockState = {
      notesStatus: { status: "absent" },
      midiExportStatus: { status: "absent" },
      midiExportPosts: [],
    };
    await mockRoutes(page, state);
    await page.goto(`/app/sequence-models/event-encoder/${JOB_ID}/piano-roll`);

    const button = page.getByTestId("eej-piano-roll-midi-export-button");
    await expect(button).toBeVisible();
    await expect(button).toBeDisabled();
    await expect(button).toHaveText("Export MIDI");
  });

  test("export click transitions to Download MIDI", async ({ page }) => {
    const state: MockState = {
      notesStatus: completeNotesStatus(),
      midiExportStatus: { status: "absent" },
      midiExportPosts: [],
    };
    await mockRoutes(page, state);
    await page.goto(`/app/sequence-models/event-encoder/${JOB_ID}/piano-roll`);

    const button = page.getByTestId("eej-piano-roll-midi-export-button");
    await expect(button).toBeEnabled();
    await expect(button).toHaveText("Export MIDI");

    await button.click();
    expect(state.midiExportPosts).toHaveLength(1);
    expect(JSON.parse(state.midiExportPosts[0].body)).toEqual({});

    await expect(button).toHaveText("Exporting…");

    state.midiExportStatus = midiExportRow("complete");
    await expect(button).toHaveText("Download MIDI", { timeout: 5_000 });
  });

  test("download endpoint returns the v2 channelized SMF via page fetch", async ({
    page,
  }) => {
    const state: MockState = {
      notesStatus: completeNotesStatus(),
      midiExportStatus: midiExportRow("complete"),
      midiExportPosts: [],
    };
    await mockRoutes(page, state);
    await page.goto(`/app/sequence-models/event-encoder/${JOB_ID}/piano-roll`);

    // Use page-context fetch so the mocked routes intercept the request.
    // (Playwright's request fixture is a separate APIRequestContext and
    // does not honor page.route handlers.)
    const result = await page.evaluate(async (jobId: string) => {
      const resp = await fetch(
        `/api/sequence-models/event-encoders/${jobId}/midi-export`,
      );
      const buf = await resp.arrayBuffer();
      return {
        ok: resp.ok,
        contentDisposition: resp.headers.get("content-disposition"),
        bytes: Array.from(new Uint8Array(buf)),
      };
    }, JOB_ID);

    expect(result.ok).toBe(true);
    expect(result.contentDisposition ?? "").toContain(
      `event_encoder_${JOB_ID}_notes_v2.mid`,
    );

    const bytes = new Uint8Array(result.bytes);
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    expect(magic).toBe("MThd");
    // SMF format (bytes 8-9, big-endian)
    expect((bytes[8] << 8) | bytes[9]).toBe(1);
    // Track count >= 3 (tempo + at least 2 channel tracks); the v2
    // channelized layout actually emits 8.
    expect((bytes[10] << 8) | bytes[11]).toBeGreaterThanOrEqual(3);
  });

  test("re-export submits force=true", async ({ page }) => {
    const state: MockState = {
      notesStatus: completeNotesStatus(),
      midiExportStatus: midiExportRow("complete"),
      midiExportPosts: [],
    };
    await mockRoutes(page, state);
    await page.goto(`/app/sequence-models/event-encoder/${JOB_ID}/piano-roll`);

    const primary = page.getByTestId("eej-piano-roll-midi-export-button");
    await expect(primary).toHaveText("Download MIDI");

    await page.getByTestId("eej-piano-roll-midi-export-menu-button").click();
    await page.getByTestId("eej-piano-roll-midi-export-rerun").click();

    expect(state.midiExportPosts).toHaveLength(1);
    expect(JSON.parse(state.midiExportPosts[0].body)).toEqual({ force: true });
  });
});
