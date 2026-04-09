# Tuning Page Improvements — Design Spec

**Date**: 2026-04-09

## Overview

Four UI and API improvements to the hyperparameter Tuning page: manifest table
enrichment, manifest detail detection job display, candidate deletion with
promotion warning, and disk artifact cleanup on all deletes.

---

## 1. Manifest Table: Positives & Negatives Columns

**Goal**: Show positive and negative example counts directly in the manifest list table.

**Backend**: Add `positive_count` and `negative_count` fields to `ManifestSummary`
response schema. Compute by summing across all splits in the stored `split_summary`
JSON. Returns `null` for manifests that haven't completed (queued/running/failed) since
`split_summary` is only populated by the worker on completion.

No new database columns — `split_summary` already persists the per-split breakdown.

**Frontend**: Add two columns after "Examples" in the manifest table: **Positives** and
**Negatives**. Display integer counts, or "—" when null.

---

## 2. Manifest Detail: Detection Job List

**Goal**: Show included detection jobs in the manifest detail expand section with
human-readable format: `"Orcasound Lab 2021-10-29 00:00 UTC — 2021-10-30 00:00 UTC"`.

**No API changes**. The frontend already fetches detection jobs for the manifest create
dialog (`useDetectionJobs`). The manifest detail response includes `detection_job_ids`.
Resolve formatting client-side by looking up each ID in the already-loaded detection
jobs list and formatting `hydrophone_name` + `start_timestamp`/`end_timestamp` as UTC
date strings.

Display as a simple string list in the expanded manifest detail section, replacing the
current generic source summary.

---

## 3. Candidate Delete with Promotion Warning

**Goal**: Allow deleting candidates from the candidate list, with a warning when the
candidate has been promoted to a classifier model.

**Backend**: Add `DELETE /classifier/hyperparameter/candidates/{candidate_id}` endpoint.
Always allows deletion (no 409 blocking). Returns `{"status": "deleted"}`. Deletes the
database record and any candidate-owned disk artifacts (see section 4).

**Frontend**: Add a trash icon button to each candidate row in the candidates section.
On click:
- Check if the candidate has a linked classifier model (candidate detail already
  exposes model linkage via `replay_verification`).
- If linked model exists: show confirmation dialog with warning text like
  "This candidate has been promoted to a classifier model. Delete anyway?"
- If no linked model: show standard delete confirmation.
- On confirm: call DELETE endpoint, invalidate queries, show success toast.

---

## 4. Disk Artifact Cleanup on All Deletes

**Manifest delete**: Already removes `hyperparameter/manifests/{id}/` directory.
No change needed.

**Search delete**: Already removes `hyperparameter/searches/{id}/` directory.
No change needed.

**Candidate delete** (new): Investigate candidate artifact storage during
implementation. Candidates may reference paths from the search they were imported from
(shared `manifest_path`, etc.) or have their own copied artifacts. Delete only artifacts
exclusively owned by the candidate — do not remove shared artifacts belonging to an
existing search. If candidates store artifacts under a dedicated directory, remove that
directory. If they only hold references to search artifacts, no additional disk cleanup
is needed beyond the DB record deletion.
