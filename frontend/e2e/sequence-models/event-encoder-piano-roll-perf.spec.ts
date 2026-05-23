// Perf regression for the Piano Roll Notes ribbon renderer (ADR-069
// §9.6 / Plan Task 7). The synthetic backend serves 10k notes with 10
// contour frames each; during a 1 s pan gesture the rAF median frame
// interval must stay ≤ 33 ms (≈ 30 fps). The test is intentionally
// permissive because Playwright runs headless on CI hardware — local
// dev typically hits 60 fps on the same fixture.

import { expect, test, type Page } from "@playwright/test";

const JOB_ID = "eej-piano-roll-perf";
const REGION_JOB_ID = "rj-piano-roll-perf";
const JOB_START = 1_751_644_800;
const JOB_END = JOB_START + 300;
const N_NOTES = 10_000;
const N_CONTOUR_FRAMES_PER_NOTE = 10;

const COMPLETE_JOB = {
  id: JOB_ID,
  status: "complete",
  event_segmentation_job_id: "seg-perf",
  event_source_mode: "raw",
  continuous_embedding_job_id: "cej-perf",
  continuous_embedding_signature: "perf-sig",
  tokenizer_version: "crnn-event-encoder-v3",
  pooling_config_json: "{}",
  descriptor_config_json: "{}",
  preprocessing_config_json: "{}",
  k_values_json: "[50]",
  random_seed: 0,
  tokenization_signature: "perf-token-sig",
  event_vector_dim: 150,
  total_events: 200,
  encoded_events: 200,
  skipped_events: 0,
  event_vectors_path: `/tmp/event-encoders/${JOB_ID}/event_vectors.parquet`,
  event_tokens_path: `/tmp/event-encoders/${JOB_ID}/event_tokens.parquet`,
  token_sequences_path: `/tmp/event-encoders/${JOB_ID}/token_sequences.parquet`,
  manifest_path: `/tmp/event-encoders/${JOB_ID}/manifest.json`,
  report_path: `/tmp/event-encoders/${JOB_ID}/report.json`,
  error_message: null,
  created_at: "2026-05-22T01:00:00Z",
  updated_at: "2026-05-22T01:10:00Z",
};

interface SyntheticNote {
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
  note_uid: string;
  f0_track_id: number;
  contour_frame_count: number;
}

function buildSyntheticNotes(): SyntheticNote[] {
  // Spread 10k notes across the 300 s window, ~3 per midi-pitch row, ~0.3 s each.
  const notes: SyntheticNote[] = [];
  for (let i = 0; i < N_NOTES; i += 1) {
    const start_offset_s = (i / N_NOTES) * 280;
    const midi_pitch = 36 + (i % 60);
    notes.push({
      event_id: `ev-${Math.floor(i / 50)}`,
      event_token: i % 50,
      partial_index: 0,
      midi_pitch,
      start_utc: JOB_START + start_offset_s,
      start_offset_s,
      duration_s: 0.3,
      velocity: 64 + (i % 32),
      peak_magnitude: -3.0 + (i % 100) / 100.0,
      track_id: i,
      note_uid: `perf-uid-${i.toString(16).padStart(8, "0")}`,
      f0_track_id: i,
      contour_frame_count: N_CONTOUR_FRAMES_PER_NOTE,
    });
  }
  return notes;
}

const SYNTHETIC_NOTES = buildSyntheticNotes();

interface ContourFrame {
  frame_index: number;
  time_offset_s: number;
  cents_from_pitch: number;
  harmonic_strength: number;
  subharmonic_octave: number;
}

function buildContoursFor(noteUids: string[]): Record<string, ContourFrame[]> {
  const out: Record<string, ContourFrame[]> = {};
  for (const uid of noteUids) {
    const frames: ContourFrame[] = [];
    for (let i = 0; i < N_CONTOUR_FRAMES_PER_NOTE; i += 1) {
      frames.push({
        frame_index: i,
        time_offset_s: (i / (N_CONTOUR_FRAMES_PER_NOTE - 1)) * 0.3,
        // Small sinusoidal cents wobble so polylines aren't trivially flat.
        cents_from_pitch: 20 * Math.sin((i / N_CONTOUR_FRAMES_PER_NOTE) * Math.PI * 2),
        harmonic_strength: -2.5,
        subharmonic_octave: 0,
      });
    }
    out[uid] = frames;
  }
  return out;
}

async function installRoutes(page: Page) {
  await page.route("**/health", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: '{"ok":true}' }),
  );
  await page.route(
    `**/sequence-models/event-encoders/${JOB_ID}`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ job: COMPLETE_JOB, manifest: null, report: null }),
      }),
  );
  await page.route(
    `**/sequence-models/event-encoders/${JOB_ID}/timeline*`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: JOB_ID,
          source_kind: "region_crnn",
          region_detection_job_id: REGION_JOB_ID,
          job_start_timestamp: JOB_START,
          job_end_timestamp: JOB_END,
          selected_k: 50,
          valid_k_values: [50],
          descriptor_feature_names: [],
          descriptor_units: {},
          events: [],
          notes_status: { status: "complete", extractor_version: "v3" },
        }),
      }),
  );
  await page.route(
    `**/sequence-models/event-encoders/${JOB_ID}/notes-status*`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "complete",
          extractor_version: "v3",
        }),
      }),
  );
  await page.route(
    `**/sequence-models/event-encoders/${JOB_ID}/notes*`,
    (route) => {
      const url = route.request().url();
      // /notes (GET) and /notes/contours (POST) share a prefix; route by method+path.
      if (url.includes("/contours")) {
        return; // handled below
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: JOB_ID,
          extractor_version: "v3",
          n_notes: SYNTHETIC_NOTES.length,
          notes: SYNTHETIC_NOTES,
        }),
      });
    },
  );
  await page.route(
    `**/sequence-models/event-encoders/${JOB_ID}/notes/contours`,
    async (route) => {
      const body = route.request().postDataJSON() as { note_uids: string[] };
      const contours = buildContoursFor(body.note_uids ?? []);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: JOB_ID,
          extractor_version: "v3",
          n_notes: Object.keys(contours).length,
          contours,
        }),
      });
    },
  );
  await page.route("**/call-parsing/region-jobs/*/tile**", (route) =>
    route.fulfill({ status: 204, body: "" }),
  );
  await page.route("**/call-parsing/region-jobs/*/audio-slice**", (route) =>
    route.fulfill({ status: 204, body: "" }),
  );
}

test.describe("Piano Roll perf", () => {
  test("pan gesture maintains ≥ 30 fps median on 10k notes × 10 frames", async ({
    page,
  }) => {
    test.setTimeout(60_000);
    await installRoutes(page);
    await page.goto(`/app/sequence-models/event-encoder/${JOB_ID}/piano-roll`);
    // Wait for at least one contour batch to land before sampling FPS.
    await page.waitForResponse(
      (r) => r.url().includes("/notes/contours") && r.request().method() === "POST",
      { timeout: 30_000 },
    );
    // Give React Query a beat to flush the cache update.
    await page.waitForTimeout(500);

    const canvas = await page.locator("canvas").first().boundingBox();
    if (!canvas) throw new Error("piano roll canvas not visible");

    // Sample requestAnimationFrame intervals for ~1 s while a pan gesture
    // runs in parallel.
    const samplerHandle = await page.evaluateHandle(async () => {
      const intervals: number[] = [];
      const start = performance.now();
      let last = start;
      let raf = 0;
      const tick = () => {
        const now = performance.now();
        intervals.push(now - last);
        last = now;
        if (now - start < 1000) {
          raf = requestAnimationFrame(tick);
        }
      };
      raf = requestAnimationFrame(tick);
      // Resolve when sampling ends.
      return new Promise<number[]>((resolve) => {
        const wait = () => {
          if (performance.now() - start >= 1000) {
            cancelAnimationFrame(raf);
            resolve(intervals);
          } else {
            setTimeout(wait, 50);
          }
        };
        wait();
      });
    });

    const cx = canvas.x + canvas.width / 2;
    const cy = canvas.y + canvas.height / 2;
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    // Drag left to pan the time axis.
    for (let dx = 0; dx <= 300; dx += 20) {
      await page.mouse.move(cx - dx, cy);
      await page.waitForTimeout(30);
    }
    await page.mouse.up();

    const intervals = (await samplerHandle.jsonValue()) as number[];
    expect(intervals.length).toBeGreaterThanOrEqual(5);
    const sorted = [...intervals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    // 30 fps median → ≤ 33.3 ms median interval. Allow 50 ms headroom for CI noise.
    expect(median, `median rAF interval ${median} ms (intervals: ${intervals.length})`).toBeLessThanOrEqual(50);
  });
});
