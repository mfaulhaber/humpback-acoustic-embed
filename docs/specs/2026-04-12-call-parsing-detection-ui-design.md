# Call Parsing Detection UI — Design Spec

**Date:** 2026-04-12
**Scope:** Frontend UI for Call Parsing Pass 1 (Region Detection)

---

## 1. Overview

Add a frontend page for creating and managing Call Parsing Pass 1 region detection jobs. The page follows the same dual-panel (Active/Previous) pattern as the existing Hydrophone detection page but is simpler: no expanded row detail, no labeling, no sample extraction, no timeline (deferred).

The page lives under a new top-level "Call Parsing" nav group, designed to accommodate future Pass 2–4 pages.

---

## 2. Navigation & Routing

**Route:** `/app/call-parsing/detection`

**Left-hand nav:** New collapsible "Call Parsing" group (positioned after Classifier group) with sub-items:
- **Detection** — this page (active link)
- Segmentation, Classification, Sequence — future, not rendered yet

---

## 3. Job Creation Form

Card titled "Region Detection" at the top of the page.

### 3.1 Main Fields (always visible)

| Field | Control | Default | Notes |
|-------|---------|---------|-------|
| Hydrophone | `<select>` dropdown | — | Orcasound + NOAA sources, reuses `useHydrophones()` |
| Date Range | `DateRangePickerUtc` | — | Reused shared component, UTC epoch seconds |
| Classifier Model | `<select>` dropdown | — | From `useClassifierModels()`, shows name + version |
| High Threshold | Slider 0–1 | 0.90 | Hysteresis onset threshold |
| Low Threshold | Slider 0–1 | 0.80 | Hysteresis continuation threshold |

### 3.2 Advanced Settings (collapsible, collapsed by default)

| Field | Control | Default | Notes |
|-------|---------|---------|-------|
| Hop Size | Number input | 1.0s | Window stride for Perch scoring |
| Padding | Number input | 1.0s | Symmetric padding around detected regions |
| Min Region Duration | Number input | 0.0s | Filter regions shorter than this |
| Stream Chunk Size | Number input | 1800s | Hydrophone streaming chunk size |

### 3.3 Submit

"Start Detection" button. Disabled when:
- Hydrophone not selected
- Date range incomplete
- Classifier model not selected
- Mutation in progress

### 3.4 Model Config Resolution

The Pass 1 API requires both `classifier_model_id` and `model_config_id`. The UI shows a single classifier model dropdown. On submit:
1. Look up the selected classifier model's `model_version` field
2. Match against the model configs list to find the corresponding `model_config_id`
3. Include both IDs in the `POST /call-parsing/region-jobs` request

---

## 4. Active Jobs Panel

Panel titled "Active Jobs" with a badge showing count of queued/running jobs.

### 4.1 Table Columns

| Column | Content |
|--------|---------|
| Status | Badge (queued / running) |
| Created | Timestamp |
| Hydrophone | Name |
| Date Range | UTC start–end |
| Thresholds | High / Low values |
| Actions | Cancel button |

### 4.2 Behavior

- Auto-refresh at 3-second interval
- Jobs move to Previous panel on completion/failure
- Cancel button calls `DELETE /call-parsing/region-jobs/{id}` (removes the job and artifacts entirely). The backend has no separate cancel-and-keep-row endpoint, so canceling is equivalent to deleting an in-flight job.

---

## 5. Previous Jobs Panel

Panel titled "Previous Jobs" with badge showing filtered count.

### 5.1 Header Bar

- Search input: "Filter by hydrophone..."
- Pagination controls (Previous / Next, page size: 10/20/50/100)
- "Delete (N)" button — enabled when rows selected

### 5.2 Table Columns

| Column | Content |
|--------|---------|
| Checkbox | Row selection for bulk delete |
| Status | Badge (complete / failed / canceled), sortable |
| Created | Sortable |
| Hydrophone | Sortable |
| Date Range | Sortable, UTC |
| Thresholds | High / Low |
| Regions | Count from `region_count`, sortable |
| Timeline | Disabled button (placeholder for future session) |
| Error | Red text, failed jobs only |

### 5.3 Behavior

- Default sort: Created descending
- Column sorting via header click
- Pagination with configurable page size
- Bulk delete with `BulkDeleteDialog` confirmation
- No expanded row view

---

## 6. Component Structure

```
frontend/src/
├── components/call-parsing/
│   ├── RegionJobForm.tsx        — creation form with all fields + advanced section
│   ├── RegionJobTable.tsx       — table layout for both active + previous panels
│   └── RegionJobSummary.tsx     — summary cell rendering (region count, thresholds)
├── pages/call-parsing/
│   └── DetectionPage.tsx        — page shell composing sub-components
├── hooks/
│   └── useCallParsing.ts        — TanStack Query hooks for region job API
└── types/
    └── callParsing.ts           — TypeScript interfaces for jobs, configs, responses
```

**RegionJobTable.tsx** handles both Active and Previous panels, parameterized by:
- Status filter (active vs. completed)
- Column set (actions vs. checkboxes)
- Whether pagination/search is shown

**RegionJobForm.tsx** owns form state and validation, emits a submit callback with the fully-built request payload.

---

## 7. API Integration

### 7.1 Queries (TanStack Query)

| Hook | Endpoint | Notes |
|------|----------|-------|
| `useRegionDetectionJobs()` | `GET /call-parsing/region-jobs` | Polled at 3s when active jobs exist |
| `useClassifierModels()` | existing | Reused from classifier hooks |
| `useHydrophones()` | existing | Reused from classifier hooks |
| `useModelConfigs()` | `GET /processing/model-configs` or equivalent | For resolving `model_config_id` |

### 7.2 Mutations

| Hook | Endpoint | Notes |
|------|----------|-------|
| `useCreateRegionJob()` | `POST /call-parsing/region-jobs` | Includes resolved `model_config_id` |
| `useCancelRegionJob()` | `DELETE /call-parsing/region-jobs/{id}` | Removes job entirely (no cancel-in-place) |
| `useBulkDeleteRegionJobs()` | Sequential `DELETE /call-parsing/region-jobs/{id}` | Invalidates query cache after |

---

## 8. Excluded from This Page

These features from the Hydrophone detection page are intentionally omitted:

- Local Cache source option
- Audio file source option
- Confidence Threshold slider
- Window Selection dropdown + prominence/tiling parameters
- Expanded row detail view (spectrograms, audio, labels)
- Save Labels / Extract Samples / Download TSV
- Timeline navigation (button shown disabled as placeholder)

---

## 9. Future Considerations

- **Timeline support**: The disabled Timeline button will be enabled in a future session when the timeline viewer supports region detection jobs.
- **Pass 2–4 pages**: The "Call Parsing" nav group and component structure are designed to accommodate Segmentation, Classification, and Sequence pages. Sub-components like the hydrophone selector may be shared.
- **Audio file source**: Could be added later if region detection on uploaded files needs a UI path.
