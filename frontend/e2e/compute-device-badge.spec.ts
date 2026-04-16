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

const SEG_MODEL = {
  id: "sm-1",
  name: "crnn-bootstrap-v1",
  model_family: "pytorch_crnn",
  model_path: "/tmp/crnn.pt",
  config_json: JSON.stringify({ framewise_f1: 0.81, event_f1_iou_0_3: 0.73 }),
  training_job_id: null,
  created_at: "2026-04-11T00:00:00Z",
};

function makeSegJob(
  id: string,
  compute_device: string | null,
  gpu_fallback_reason: string | null,
) {
  return {
    id,
    status: "complete",
    region_detection_job_id: REGION_JOB.id,
    segmentation_model_id: SEG_MODEL.id,
    config_json: JSON.stringify({ high_threshold: 0.5, low_threshold: 0.3 }),
    parent_run_id: null,
    event_count: 3,
    compute_device,
    gpu_fallback_reason,
    error_message: null,
    created_at: "2026-04-16T04:00:00Z",
    updated_at: "2026-04-16T04:05:00Z",
    started_at: "2026-04-16T04:00:01Z",
    completed_at: "2026-04-16T04:05:00Z",
  };
}

function makeClassifyJob(
  id: string,
  compute_device: string | null,
  gpu_fallback_reason: string | null,
) {
  return {
    id,
    status: "complete",
    event_segmentation_job_id: "sj-cpu",
    vocalization_model_id: "vm-1",
    typed_event_count: 3,
    compute_device,
    gpu_fallback_reason,
    parent_run_id: null,
    error_message: null,
    created_at: "2026-04-16T05:00:00Z",
    updated_at: "2026-04-16T05:05:00Z",
    started_at: "2026-04-16T05:00:01Z",
    completed_at: "2026-04-16T05:05:00Z",
  };
}

const SEG_JOBS = [
  makeSegJob("sj-cpu", "cpu", null),
  makeSegJob("sj-mps", "mps", null),
  makeSegJob("sj-fallb", "cpu", "mps_output_mismatch"),
];

const CLASSIFY_JOBS = [
  makeClassifyJob("cj-cpu", "cpu", null),
  makeClassifyJob("cj-mps", "mps", null),
  makeClassifyJob("cj-fallb", "cpu", "cuda_load_error"),
];

const VOC_MODEL = {
  id: "vm-1",
  name: "event-cnn-v1",
  vocabulary_snapshot: "[]",
  per_class_thresholds: "{}",
  model_dir_path: "/tmp/vm",
  model_family: "pytorch_event_cnn",
  input_mode: "segmented_event",
  feature_config_json: "{}",
  created_at: "2026-04-11T00:00:00Z",
};

async function setupCommonMocks(page: Page) {
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
  await page.route("**/call-parsing/segmentation-models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([SEG_MODEL]),
    }),
  );
  await page.route("**/call-parsing/classifier-models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([VOC_MODEL]),
    }),
  );
  await page.route("**/classifier/models", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/admin/models", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/vocalization/types", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
}

test.describe("ComputeDeviceBadge", () => {
  test("renders cpu, mps, and fallback shapes on segment jobs page", async ({
    page,
  }) => {
    await setupCommonMocks(page);
    await page.route("**/call-parsing/segmentation-jobs", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(SEG_JOBS),
      }),
    );

    await page.goto("/app/call-parsing/segment");
    await expect(page.locator("text=Previous Jobs")).toBeVisible();

    const cpuRow = page.locator("tr").filter({ hasText: "sj-cpu" }).first();
    await expect(cpuRow).toBeVisible();
    await expect(cpuRow.getByText("CPU", { exact: true })).toBeVisible();

    const mpsRow = page.locator("tr").filter({ hasText: "sj-mps" }).first();
    await expect(mpsRow.getByText("MPS", { exact: true })).toBeVisible();

    const fbRow = page.locator("tr").filter({ hasText: "sj-fallb" }).first();
    await expect(
      fbRow.getByText("CPU (fallback: mps_output_mismatch)"),
    ).toBeVisible();
  });

  test("renders cpu, mps, and fallback shapes on classify jobs page", async ({
    page,
  }) => {
    await setupCommonMocks(page);
    await page.route("**/call-parsing/segmentation-jobs*", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(SEG_JOBS),
      }),
    );
    await page.route("**/call-parsing/classification-jobs", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(CLASSIFY_JOBS),
      }),
    );

    await page.goto("/app/call-parsing/classify");
    await expect(page.locator("text=Previous Jobs")).toBeVisible();

    const cpuRow = page.locator("tr").filter({ hasText: "cj-cpu" }).first();
    await expect(cpuRow).toBeVisible();
    await expect(cpuRow.getByText("CPU", { exact: true })).toBeVisible();

    const mpsRow = page.locator("tr").filter({ hasText: "cj-mps" }).first();
    await expect(mpsRow.getByText("MPS", { exact: true })).toBeVisible();

    const fbRow = page.locator("tr").filter({ hasText: "cj-fallb" }).first();
    await expect(fbRow.getByText("CPU (fallback: cuda_load_error)")).toBeVisible();
  });
});
