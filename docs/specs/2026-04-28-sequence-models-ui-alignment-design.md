# Sequence Models UI Alignment Design

**Date**: 2026-04-28
**Status**: Approved

## Goal

Bring the Sequence Models pages (Continuous Embedding, HMM Sequence) in line
with the Call Parsing Segment page UI patterns. Specifically: add job deletion
(DB + disk artifacts), add Review buttons, convert card layouts to table layouts
with Active/Previous panels, and fix the broken breadcrumb navigation.

## Reference Pattern

The Call Parsing Segment page (`SegmentPage.tsx` + `SegmentJobTable.tsx`)
provides the reference implementation:

- **Active Jobs panel**: Bordered panel with title + badge count. Table with
  no checkboxes, no sort/filter/pagination. Cancel button (X icon) per row.
  Panel hides when no active jobs exist.
- **Previous Jobs panel**: Bordered panel with toolbar (filter input, bulk
  Delete button, pagination). Table with checkboxes, sortable column headers.
  Per-row actions: Review link, Delete button (red text). Bulk delete via
  `BulkDeleteDialog`.
- **Delete**: Backend removes disk artifacts directory then deletes DB row.
  Frontend calls DELETE endpoint, invalidates query cache.

## 1. Backend — Delete Endpoints

### 1.1 DELETE /sequence-models/continuous-embeddings/{job_id}

New service function `delete_continuous_embedding_job` in
`continuous_embedding_service.py`:

1. Fetch DB row by ID; return `False` if not found.
2. Remove disk artifacts via `shutil.rmtree(continuous_embedding_dir(...))`,
   ignoring missing directories.
3. Delete DB row.
4. Commit transaction.
5. Return `True`.

API route in `sequence_models.py` router: returns 204 on success, 404 if job
not found.

### 1.2 DELETE /sequence-models/hmm-sequences/{job_id}

New service function `delete_hmm_sequence_job` in `hmm_sequence_service.py`,
same pattern using `hmm_sequence_dir(...)`.

API route: 204 on success, 404 if not found.

Both follow the established pattern from
`call_parsing.py:delete_event_segmentation_job` — no soft delete, permanent
removal of DB record and all disk artifacts.

## 2. Frontend — Table Components

### 2.1 ContinuousEmbeddingJobTable

New component replacing card-based job list. Follows `SegmentJobTable` pattern.

**Active mode columns**: Status | Created | Region Job (short ID) |
Model Version | Spans | Windows | Actions (Cancel button with X icon)

**Previous mode columns**: Checkbox | Status (sortable) | Created (sortable) |
Region Job (sortable) | Model Version | Spans (sortable) | Windows (sortable) |
Actions (Review link + Delete button)

Previous mode features: filter input, sortable headers, pagination (20/page),
bulk delete with `BulkDeleteDialog`, per-row checkbox selection.

Review link navigates to the existing detail page:
`/app/sequence-models/continuous-embedding/{jobId}`

### 2.2 HMMSequenceJobTable

Same structural pattern.

**Active mode columns**: Status | Created | Source CE Job (short ID) |
States | PCA Dims | Actions (Cancel)

**Previous mode columns**: Checkbox | Status (sortable) | Created (sortable) |
Source CE Job (sortable) | States (sortable) | PCA Dims | Train Seqs (sortable) |
Log Likelihood (sortable) | Actions (Review link + Delete)

Review link navigates to:
`/app/sequence-models/hmm-sequence/{jobId}`

### 2.3 Page-Level Changes

Both `ContinuousEmbeddingJobsPage` and `HMMSequenceJobsPage` switch from card
grids to `*JobTablePanel` wrappers (bordered panels with title + badge count).
Active panel hides when empty. Create form stays at top.

### 2.4 API Client + Query Hooks

Add to `sequenceModels.ts`:

- `deleteContinuousEmbeddingJob(jobId)` — DELETE request
- `deleteHMMSequenceJob(jobId)` — DELETE request
- `useDeleteContinuousEmbeddingJob()` — mutation invalidating
  `continuous-embedding-jobs` cache
- `useDeleteHMMSequenceJob()` — mutation invalidating
  `hmm-sequence-jobs` cache

### 2.5 Removed Components

Card components become unused and are deleted:
- `ContinuousEmbeddingJobCard.tsx`
- `HMMSequenceJobCard.tsx`

## 3. Breadcrumb Fix

### 3.1 Static Routes

Add to `staticRoutes` in `Breadcrumbs.tsx`:

```
"/app/sequence-models/continuous-embedding": [
  { label: "Sequence Models", to: "/app/sequence-models" },
  { label: "Continuous Embedding" },
]
"/app/sequence-models/hmm-sequence": [
  { label: "Sequence Models", to: "/app/sequence-models" },
  { label: "HMM Sequence" },
]
```

### 3.2 Dynamic Detail Page Breadcrumbs

Add a path-prefix match for `/app/sequence-models/` in the breadcrumb logic
(alongside the existing `audioId` and clustering `jobId` patterns). Extract
the job ID from the URL path to render:

- **Sequence Models > Continuous Embedding > Job {short_id}**
- **Sequence Models > HMM Sequence > Job {short_id}**

This fixes the current fallback to "Audio" for all Sequence Models routes.

## Scope Boundaries

- No new review workspaces — Review links go to existing detail pages.
- No changes to the create forms.
- No changes to the detail pages themselves.
- No changes to cancel behavior (already implemented).
- Reuses existing shared components: `StatusBadge`, `BulkDeleteDialog`,
  `Badge`, `Checkbox`.
