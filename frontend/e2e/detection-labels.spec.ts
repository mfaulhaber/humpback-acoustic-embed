import { test, expect } from "@playwright/test";

/**
 * Tests for detection label annotation (humpback, orca, ship, background checkboxes).
 * Requires: backend running on :8000 with at least one completed detection job.
 */

// Helper to find a completed detection job
async function findCompletedJob(request: any) {
  const jobsRes = await request.get(
    "http://localhost:8000/classifier/detection-jobs",
  );
  expect(jobsRes.ok()).toBeTruthy();
  const jobs = await jobsRes.json();
  return jobs.find(
    (j: { status: string; output_tsv_path: string | null }) =>
      j.status === "complete" && j.output_tsv_path,
  );
}

test.describe("Detection labels API", () => {
  test("GET content returns label columns", async ({ request }) => {
    const job = await findCompletedJob(request);
    if (!job) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    const row = rows[0];
    // Label fields should be present (initially null if TSV has no label columns)
    expect("humpback" in row).toBe(true);
    expect("orca" in row).toBe(true);
    expect("ship" in row).toBe(true);
    expect("background" in row).toBe(true);
    // avg_confidence should still be present
    expect("avg_confidence" in row).toBe(true);
  });

  test("PUT labels round-trip persists and resets", async ({ request }) => {
    const job = await findCompletedJob(request);
    if (!job) {
      test.skip(true, "No completed detection jobs available");
      return;
    }

    // Get initial content
    const contentRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    expect(contentRes.ok()).toBeTruthy();
    const rows = await contentRes.json();
    expect(rows.length).toBeGreaterThan(0);

    const firstRow = rows[0];

    // Save labels on the first row
    const putRes = await request.put(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/labels`,
      {
        data: [
          {
            filename: firstRow.filename,
            start_sec: firstRow.start_sec,
            end_sec: firstRow.end_sec,
            humpback: 1,
            orca: 0,
            ship: 0,
            background: null,
          },
        ],
      },
    );
    expect(putRes.ok()).toBeTruthy();
    const putBody = await putRes.json();
    expect(putBody.status).toBe("ok");
    expect(putBody.updated).toBe(1);

    // Verify labels persisted
    const verifyRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    expect(verifyRes.ok()).toBeTruthy();
    const verifyRows = await verifyRes.json();
    const updatedRow = verifyRows.find(
      (r: any) =>
        r.filename === firstRow.filename &&
        r.start_sec === firstRow.start_sec &&
        r.end_sec === firstRow.end_sec,
    );
    expect(updatedRow).toBeTruthy();
    expect(updatedRow.humpback).toBe(1);
    expect(updatedRow.orca).toBe(0);
    expect(updatedRow.ship).toBe(0);
    expect(updatedRow.background).toBeNull();

    // Reset labels back to null (cleanup)
    const resetRes = await request.put(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/labels`,
      {
        data: [
          {
            filename: firstRow.filename,
            start_sec: firstRow.start_sec,
            end_sec: firstRow.end_sec,
            humpback: null,
            orca: null,
            ship: null,
            background: null,
          },
        ],
      },
    );
    expect(resetRes.ok()).toBeTruthy();

    // Verify reset
    const resetVerifyRes = await request.get(
      `http://localhost:8000/classifier/detection-jobs/${job.id}/content`,
    );
    const resetRows = await resetVerifyRes.json();
    const resetRow = resetRows.find(
      (r: any) =>
        r.filename === firstRow.filename &&
        r.start_sec === firstRow.start_sec &&
        r.end_sec === firstRow.end_sec,
    );
    expect(resetRow.humpback).toBeNull();
    expect(resetRow.orca).toBeNull();
    expect(resetRow.ship).toBeNull();
    expect(resetRow.background).toBeNull();
  });
});

