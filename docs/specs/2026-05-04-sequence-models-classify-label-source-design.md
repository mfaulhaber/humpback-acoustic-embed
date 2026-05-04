# Sequence Models — Replace Vocalization Labels with Call Parsing Classify Labels

**Status:** design
**Date:** 2026-05-04
**Supersedes:** ADR-060 (tier-aware HMM label distribution storage)
**Related:** ADR-054 (read-time correction overlay), ADR-056 (Sequence Models track), ADR-057 (CRNN region-based source), ADR-059 (source-agnostic loader Protocol), ADR-061 (masked-transformer), ADR-062 (segmentation-scoped event identity)

---

## 1. Goal

Replace `vocalization_labels` with the Call Parsing Pass 3 output (`EventClassificationJob` plus `EventTypeCorrection` and `EventBoundaryCorrection` overlays via `load_effective_events()`) as the **only** label source feeding HMM Sequence and Masked Transformer label-distribution artifacts and exemplar annotations.

Vocalization Labeling workspace and the `vocalization_labels` table are not touched; they continue to serve their own workflow.

---

## 2. Motivation

The current label coupling pulls per-detection-window labels manually annotated in the Vocalization Labeling workspace. To do so, HMM/MT loaders walk:

```
ContinuousEmbeddingJob → EventSegmentationJob → RegionDetectionJob → hydrophone_id
                       → all DetectionJobs on that hydrophone → vocalization_labels
```

…then center-time-match every sequence window against every detection-window's `[start_utc, end_utc)` to find any labels whose detection-window contains the sequence-window's center.

This has three problems:
1. **Stale source.** `vocalization_labels` reflects manual annotation in the older 5-second-window workspace. Call Parsing Classify (Pass 3) is the active labeling pipeline for the same audio, with a managed multi-label classifier and human-feedback loops.
2. **Indirect join.** Sequence Models already pivot on `EventSegmentationJob`. Going out to detection windows on the same hydrophone is a longer chain than necessary and pulls in label rows from unrelated detection runs.
3. **Coarse granularity.** Detection windows are fixed 5-second buckets. Pass 2 events are precise variable-length intervals; using them yields tighter state-to-label mapping with fewer false-overlap labels.

By switching to the Classify pipeline's events, the label source becomes (a) up to date with the active classification workflow, (b) tightly bound to the same `EventSegmentationJob` the embeddings already pivot on, and (c) precise about onset/offset.

---

## 3. Decisions (locked in via brainstorming)

| # | Decision | Notes |
|---|---|---|
| Q1 | **Pure replacement.** `vocalization_labels` no longer feeds HMM/MT label distribution. | Vocalization Labeling workspace untouched. |
| Q2 | **Event-scoped inversion** for window→label join. Each effective event distributes its corrected, above-threshold types to every sequence window whose center falls inside its `[start, end)`. Windows outside every event map to a reserved `(background)` bucket. | Two-pointer O(n_windows + n_events). Events are non-overlapping by construction (Pass 2 segmentation, ADR-062). |
| Q3 | **Chart + exemplars** are the scope. `overlay.parquet` (PCA/UMAP scatter) is unchanged. | Exemplar cards gain type chips and link back to Classify Review for the underlying event. |
| Q4 | **Strict above-threshold + corrections.** Effective types per event = (model types where `above_threshold == True`) ∪ (user-added/confirmed) − (user-deleted). No configurable threshold knob. | Matches what users see in Classify Review as "real" labels. |
| Q5 | **Explicit `event_classification_job_id` selection at submit time.** New nullable FK on `hmm_sequence_jobs` and `masked_transformer_jobs`. Default to most recent completed Classify job for the upstream segmentation; submit rejected if none exist. | Reproducibility: the bound Classify job determines labels deterministically; later Classify runs do not silently change a finished HMM/MT job's labels. |
| Q6 | **Drop tier dimension** from `label_distribution.json`. SurfPerch and CRNN sources both produce `{n_states, total_windows, states: {state: {label: count}}}`. CRNN's `extras.tier` on `decoded.parquet` and exemplars is unchanged (it powers the per-state tier-composition strip, which is independent of label distribution). | All Sequence Models jobs already deleted via UI; no on-disk migration, no dual-format loader, no version field. ADR-060 superseded. |
| Q7 | **Manual "Regenerate label distribution" button** on HMM and MT detail pages. Re-runs the label-distribution + exemplars pass against the bound Classify job (with current corrections); optionally re-binds to a different Classify job in the same atomic write. | No auto-recompute on correction writes; no live-query rendering; artifacts remain the source of truth. |

---

## 4. Architecture

### 4.1 Pipeline shape (one substitution)

```
ContinuousEmbeddingJob
   └─ EventSegmentationJob ── (NEW) EventClassificationJob[bound by FK]
        ├─ HMMSequenceJob ──► decoded.parquet, exemplars.json,
        │                     label_distribution.json, overlay.parquet
        └─ MaskedTransformerJob (per-k) ──► same artifact set under k<N>/
```

The bound Classify job determines the label source for label-distribution and exemplar annotation. The upstream `EventSegmentationJob` continues to drive the embeddings.

### 4.2 Component changes

| Component | Change |
|---|---|
| `hmm_sequence_jobs` table | Add nullable FK `event_classification_job_id` |
| `masked_transformer_jobs` table | Add nullable FK `event_classification_job_id` |
| `hmm_sequence_service.generate_label_distribution()` | Rewritten to use event-scoped inversion against bound Classify job |
| `masked_transformer_service.generate_interpretations()` | Same rewrite (per-k loop unchanged) |
| `sequence_models/label_distribution.py` | New event-scoped join helper; old detection-window helper removed |
| `sequence_models/surfperch.py`, `crnn_region.py` (loaders) | Drop `hydrophone_id → DetectionJob` fan-out path used only for label coverage; embedding loading unchanged; tier dimension removed from label-distribution output |
| Loader Protocol (ADR-059) | `LabelDistribution` shape no longer carries a tier dimension; SurfPerch synthetic `"all"` tier removed |
| `exemplars.json` schema | Add `event_id`, `event_types`, `event_confidence` to each exemplar's `extras` |
| HMM/MT submit endpoints | Require completed `EventClassificationJob` for upstream segmentation; accept optional `event_classification_job_id` (default latest completed) |
| HMM/MT regenerate endpoints | New: `POST .../regenerate-label-distribution` (optional re-bind) |
| Frontend `LabelDistributionChart` | Remove tier-collapse `useMemo`; read `states[i]` as `Record<string, number>` directly |
| Frontend HMM/MT submit form | New "Event Classification Job" `<Select>` |
| Frontend HMM/MT detail page | "Regenerate label distribution" button + re-bind dialog; bound-Classify badge in header; exemplar cards gain types chips with click-through to Classify Review; background exemplars show `(background)` chip |
| ADR-060 | Superseded — note the replacement in DECISIONS.md |
| Disk artifacts (existing) | Wiped — no migration loader; SQL rows already deleted via UI |

### 4.3 Non-goals

- Coloring `overlay.parquet` scatter by label (Q3:B, not C).
- Auto-recompute on correction writes (Q7:A, not C).
- Configurable confidence threshold parameter (Q4:A — strict `above_threshold`).
- Touching Vocalization Labeling workspace, `vocalization_labels` table, or its UI.
- Changing CRNN's `extras.tier` metadata on `decoded.parquet` / exemplars, or the per-state tier-composition strip on the detail page.
- Changing exemplar selection logic (high-confidence, nearest-centroid, boundary picks). Annotation only.

---

## 5. Data model & migration

### 5.1 Alembic migration `065_sequence_models_classify_binding.py`

```python
revision = "065"
down_revision = "064"

def upgrade() -> None:
    with op.batch_alter_table("hmm_sequence_jobs") as batch:
        batch.add_column(sa.Column(
            "event_classification_job_id",
            sa.Integer(),
            sa.ForeignKey("event_classification_jobs.id", ondelete="RESTRICT"),
            nullable=True,
        ))
        batch.create_index(
            "ix_hmm_sequence_jobs_event_classification_job_id",
            ["event_classification_job_id"],
        )

    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(sa.Column(
            "event_classification_job_id",
            sa.Integer(),
            sa.ForeignKey("event_classification_jobs.id", ondelete="RESTRICT"),
            nullable=True,
        ))
        batch.create_index(
            "ix_masked_transformer_jobs_event_classification_job_id",
            ["event_classification_job_id"],
        )

def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_index("ix_masked_transformer_jobs_event_classification_job_id")
        batch.drop_column("event_classification_job_id")
    with op.batch_alter_table("hmm_sequence_jobs") as batch:
        batch.drop_index("ix_hmm_sequence_jobs_event_classification_job_id")
        batch.drop_column("event_classification_job_id")
```

### 5.2 Backup gate (mandatory, CLAUDE.md §3.5)

The migration is preceded by:

1. Read `HUMPBACK_DATABASE_URL` from `.env`.
2. `cp "$DB_PATH" "${DB_PATH}.YYYY-MM-DD-HH:MM.bak"` (UTC timestamp).
3. Confirm the `.bak` exists and is non-zero in size.
4. Only then `uv run alembic upgrade head`.

This is the first acceptance criterion of the implementation plan, not a parenthetical.

### 5.3 Field semantics

- **Nullable** because of the brief in-transaction window between row insert and FK resolution by the validation step. In committed state the column is always non-NULL after job-create succeeds. No CHECK constraint.
- **`ondelete="RESTRICT"`** — deleting a Classify job that is bound to any HMM/MT job is rejected at the DB layer. Users wanting to clean up should delete the dependent HMM/MT jobs first (or, future work, re-bind them).
- **No backfill.** All existing Sequence Models rows have been deleted via the UI.

---

## 6. Loader & service rewrite

### 6.1 New event-scoped join helper

`src/humpback/sequence_models/label_distribution.py`:

```python
BACKGROUND_LABEL = "(background)"

@dataclass(frozen=True)
class EffectiveEventLabels:
    event_id: int
    start_utc: float
    end_utc: float
    types: frozenset[str]               # corrected, above-threshold; may be empty
    confidences: dict[str, float]       # type -> classifier confidence; empty for user-added types

def load_effective_event_labels(
    session: Session,
    event_classification_job_id: int,
) -> list[EffectiveEventLabels]:
    """
    Reads typed_events.parquet for the given Classify job, applies
    EventTypeCorrection and EventBoundaryCorrection overlays via
    load_effective_events() (ADR-054), and returns one record per
    effective event in start_utc order.

    Type set per event = (model types where above_threshold == True)
                       ∪ (user-added/confirmed types)
                       − (user-deleted types)

    An event with empty `types` after correction is returned with
    `types = frozenset()`. Such events still represent a real interval
    but `assign_labels_to_windows()` treats their windows the same as
    windows outside any event: `event_id = None`, `event_types = []`,
    bucketed into BACKGROUND_LABEL. Rationale: keeps the chart and
    exemplar invariants consistent — `event_id` is set iff at least
    one surviving label exists.
    """

def assign_labels_to_windows(
    decoded_df: pd.DataFrame,           # columns: start_timestamp, end_timestamp, label, ...
    events: list[EffectiveEventLabels],
) -> pd.DataFrame:
    """
    Adds `event_id: int | None` and `event_types: list[str]` columns
    to decoded_df via event-scoped inversion:

      1. Sort decoded rows by center time `(start + end) / 2`.
      2. Two-pointer cursor through events in start order; for each
         event, advance the row cursor while center < event.start_utc,
         then tag rows with center in [event.start_utc, event.end_utc).
      3. Untouched rows are background: event_id=None, event_types=[].

    O(n_windows + n_events). Events are non-overlapping by construction
    (Pass 2 segmentation output, ADR-062), so each window matches at
    most one event.
    """
```

### 6.2 `hmm_sequence_service.generate_label_distribution()` (rewrite)

```
1. Load decoded.parquet from the job's artifact dir.
2. Load effective event labels for self.event_classification_job_id.
3. annotated_df = assign_labels_to_windows(decoded, events).
4. For each HMM state s in 0..n_states-1:
     count = annotated_df.loc[state == s, "event_types"]
              .pipe(explode_or_background)         # list -> rows; [] -> BACKGROUND_LABEL
              .value_counts()
     buckets[s] = dict(count)                       # {label: int}, no tier dimension
5. Atomic write label_distribution.json:
     {"n_states": N, "total_windows": len(decoded),
      "states": {"0": {...}, "1": {...}, ...}}
```

`explode_or_background` is the small bit of pandas glue that turns each row's `event_types` list into one row per type (counted in each label's bucket — multi-label union semantics) and replaces empty lists with a single `BACKGROUND_LABEL` entry so background windows show up exactly once per state.

The per-state total may exceed `total_windows` because of multi-label events — same union semantics today's loader already uses; the chart treats each label's bar independently.

### 6.3 `masked_transformer_service.generate_interpretations()` (rewrite)

Identical to §6.2 but loops over `k ∈ k_values` and writes each result to `k<N>/label_distribution.json`. The `assign_labels_to_windows()` call result is cached across the k loop since the decoded windows' `(start_timestamp, end_timestamp)` and the events don't change with `k` — only the per-row `label` (k-means token) does. Compute the annotation once, group differently per k.

### 6.4 Exemplar annotation

Selection logic unchanged (high-confidence, nearest-centroid, boundary picks per state). After selection, each exemplar's `extras` dict is annotated:

```python
exemplar.extras["event_id"] = int | None
exemplar.extras["event_types"] = list[str]              # [] for background
exemplar.extras["event_confidence"] = dict[str, float]  # subset of types with classifier confidence
```

Lookup uses the same annotated decoded frame, indexed by `(sequence_id, position_in_sequence)`. Background exemplars get `event_id=None`, `event_types=[]`, `event_confidence={}`.

CRNN source's existing `extras.tier` is untouched.

### 6.5 Removed code

- `surfperch.py` and `crnn_region.py`: the `hydrophone_id → DetectionJob[]` fan-out path used only for label coverage. Hydrophone resolution stays only where embedding loading actually requires it.
- `label_distribution.py`: the old detection-window center-time match against `vocalization_labels`. Deleted, not deprecated.
- The synthetic `"all"` tier write in SurfPerch's label-distribution path. Gone.

### 6.6 Loader Protocol (ADR-059) update

```python
class LabelDistribution(TypedDict):
    n_states: int
    total_windows: int
    states: dict[str, dict[str, int]]   # state_idx -> { label: count }
```

No tier dimension. Both SurfPerch and CRNN sources produce this shape.

ADR-060 is superseded by this design — DECISIONS.md gets an entry: *"Sequence Models label source switched to Call Parsing Classify; tier dimension removed from `label_distribution.json` (CRNN tier metadata persists on `decoded.parquet` and exemplars unchanged)."*

### 6.7 New endpoints

```
POST /api/sequence-models/hmm/{id}/regenerate-label-distribution
  body: {event_classification_job_id?: int}
  returns: {label_distribution: ..., exemplars: ...}

POST /api/sequence-models/masked-transformer/{id}/regenerate-label-distribution?k={k}
  body: {event_classification_job_id?: int}
  returns: {label_distribution: ..., exemplars: ...}
```

Synchronous; reads cached embeddings, runs §6.1–6.4, **per-file** atomic temp-then-rename for `label_distribution.json` and `exemplars.json` (and per-k for MT). If `event_classification_job_id` is provided and differs from the bound value, the order is: (1) validate, (2) write all artifact files via temp-then-rename, (3) commit the FK update in a single SQL transaction. Failure during step 2 leaves the existing files and the existing FK untouched; failure during step 3 may leave updated files paired with the old FK — in that recovery case, calling regenerate again with no body re-derives consistent artifacts from the still-current FK. Validation: the new Classify job's `event_segmentation_job_id` must equal the HMM/MT job's upstream segmentation; otherwise 400 with no FK change and no artifact change. The MT regenerate endpoint rebuilds **all** `k<N>/label_distribution.json` files in one shot (effective events loaded once); the response payload returns the active `k`'s payload. Directory-level atomicity is not provided — partial-failure recovery is via re-running regenerate.

### 6.8 Submit-endpoint validation

HMM and MT job-create endpoints accept optional `event_classification_job_id`. Server side:

1. If omitted, query the most recent `EventClassificationJob` whose `event_segmentation_job_id` matches the upstream segmentation and `status == COMPLETED`. If zero rows → 400 with message naming the segmentation.
2. If provided, verify it's `COMPLETED` and its `event_segmentation_job_id` matches the upstream segmentation; otherwise 400.
3. Store on the new column.

---

## 7. Frontend

### 7.1 HMM / MT submit forms

**Files:** `frontend/src/components/sequence-models/HMMSequenceCreatePage.tsx`, `frontend/src/components/sequence-models/MaskedTransformerCreatePage.tsx`.

New "Event Classification Job" `<Select>` directly under the existing "Event Segmentation Job" select, populated by:

```
GET /api/call-parsing/event-classification-jobs?event_segmentation_job_id={id}&status=completed
  returns: [{id, created_at, model_name, n_events_classified}, ...]   # newest first
```

- Default selection: first option (most recent).
- Disabled with helper text *"Run Pass 3 Classify on this segmentation first"* when the list is empty; the form's submit button stays disabled in that state.
- Label format per option: `#{id} • {model_name} • {n_events_classified} events • {created_at:relative}`.
- Changing the segmentation select clears and refetches the Classify select.
- TanStack Query key: `["event-classification-jobs", segmentationJobId]`.

### 7.2 HMM detail page

**File:** `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`.

- **Bound-Classify badge** in the header strip: *"Labels from Classify job #{id} ({model_name})"*; click opens Classify Review for that job.
- **`LabelDistributionChart`** simplified — `useMemo` tier-collapse step removed. Reads `states[i]` directly as `Record<string, number>`. Stacked-bar render unchanged; `(background)` is a label with a reserved neutral-gray color slot.
- **"Regenerate label distribution" button** in the chart card header. Click opens a dialog with a `<Select>` defaulting to the currently bound Classify job (other completed jobs for the same segmentation listed below). Confirm POSTs to the regenerate endpoint with optional `event_classification_job_id`. While in flight: spinner + disabled. On success: invalidate `["hmm-job", id]`, `["hmm-label-distribution", id]`, `["hmm-exemplars", id]`; toast *"Label distribution regenerated."* On error: toast with the server message; nothing changes on disk. If the user picks a different Classify job, the bound-job badge updates after refetch.
- **Exemplar gallery — type chips.** `ExemplarCard` renders `extras.event_types` as small chips below the spectrogram, in the chart legend's color palette. Chips wrap a click-target opening Classify Review filtered to `extras.event_id`. Background exemplars (`event_types: []`) show a single neutral *"(background)"* chip with no link. CRNN's existing `extras.tier` chip stays where it is; the new types row sits below it.

The existing per-state tier-composition strip is unchanged.

### 7.3 Masked Transformer detail page

**File:** `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`.

Same changes as §7.2, scoped to the active `?k=` view:

- Bound-Classify badge in header (per-MT-job, not per-k).
- `LabelDistributionChart` simplification — same.
- Regenerate button — body sends the active `k` for response routing; backend rebuilds **all** `k<N>/label_distribution.json` files in one shot. Dialog wording: *"Regenerate label distribution for all k values"*.
- Exemplar type chips — same, attached to the per-k exemplars.

The URL-synced `?k=` picker and `RegionNavBar` are untouched.

### 7.4 No changes to

- Vocalization Labeling workspace (`LabelingWorkspace.tsx`).
- `overlay.parquet` consumers (PCA/UMAP scatter).
- HMM motif extraction page.
- `DiscreteSequenceBar`, `RegionNavBar`, exemplar selection logic, audio playback paths.

### 7.5 New / changed API client functions

```ts
// frontend/src/api/sequence-models.ts (additions)
listEventClassificationJobsForSegmentation(segmentationJobId: number)
regenerateHMMLabelDistribution(jobId: number, body?: {event_classification_job_id?: number})
regenerateMTLabelDistribution(jobId: number, k: number, body?: {event_classification_job_id?: number})
```

---

## 8. Testing strategy

### 8.1 Backend unit tests

`tests/sequence_models/test_label_distribution_event_scoped.py` (new):

- `test_assign_labels_inverts_events_to_windows` — three disjoint events + decoded windows straddling them; assert center-time placement, multi-label union, background bucket for outside windows.
- `test_assign_labels_handles_empty_event_types` — event with all types user-deleted via `EventTypeCorrection` → empty type set → its windows go to background.
- `test_assign_labels_two_pointer_o_n` — large synthetic input (10k windows, 1k events); single-pass execution.
- `test_load_effective_event_labels_applies_corrections` — fixtures cover above-threshold model type kept, below-threshold model type filtered, user-added included, user-deleted excluded, user-confirmed below-threshold included.
- `test_load_effective_event_labels_applies_boundary_corrections` — `EventBoundaryCorrection` shifts an event start; window membership changes accordingly.

`tests/sequence_models/test_hmm_service_label_distribution.py` (rewrite/replace):

- End-to-end test against an in-memory fixture with one `EventSegmentationJob`, one `EventClassificationJob`, sample `typed_events.parquet`, synthetic `decoded.parquet` and HMM job row.
- Assert produced `label_distribution.json` shape (`{n_states, total_windows, states: {state: {label: count}}}`); no tier dimension; same shape for SurfPerch and CRNN sources.
- Assert `(background)` bucket appears with expected count.
- Assert exemplar JSON has `event_id`, `event_types`, `event_confidence` populated.
- Multi-label union: an event with two types contributes `+1` to each label's bucket per overlapping window.

`tests/sequence_models/test_masked_transformer_service_label_distribution.py` (rewrite/replace):

- Mirror of HMM test for per-k `k<N>/label_distribution.json`.
- Assert effective-events load happens once per service call, not once per k.

### 8.2 Submit-time validation tests

`tests/api/test_sequence_models_submit.py` (additions):

- `test_hmm_submit_rejects_when_no_classification` — 400, message names segmentation, no row created.
- `test_hmm_submit_defaults_to_latest_classification` — multiple completed Classify jobs; submit without explicit FK picks newest.
- `test_hmm_submit_rejects_mismatched_classification` — explicit `event_classification_job_id` whose segmentation differs → 400.
- `test_hmm_submit_rejects_non_completed_classification` — Classify in `RUNNING` → 400.
- Mirrored set for Masked Transformer.

### 8.3 Regenerate endpoint tests

`tests/api/test_sequence_models_regenerate.py` (new):

- `test_regenerate_hmm_rebuilds_artifacts` — write known stale `label_distribution.json`; POST regenerate; assert disk content matches expected; returned payload matches.
- `test_regenerate_hmm_rebinds_classification_job` — POST with new FK; assert FK updated and artifacts rebuilt from the new source.
- `test_regenerate_hmm_rejects_mismatched_rebind` — new Classify job belongs to a different segmentation → 400, no FK change, no artifact change.
- `test_regenerate_hmm_atomic_on_failure` — induce a write failure during the artifact-write step; assert FK is not changed and existing artifact remains the previous version (temp file cleaned). Per §6.7 partial-failure recovery is by re-running regenerate; this test only covers the step-2 failure case.
- `test_regenerate_mt_rebuilds_all_k` — multi-k MT job; regenerate; assert every `k<N>/label_distribution.json` updated; effective events loaded once.

### 8.4 Loader Protocol tests

`tests/sequence_models/test_loader_protocol.py` (additions):

- `LabelDistribution` TypedDict shape no longer carries a tier dimension; both source loaders return the simplified shape.
- SurfPerch source no longer emits the synthetic `"all"` tier.

### 8.5 Frontend Playwright tests

`frontend/tests/sequence-models-classify-binding.spec.ts` (new):

- HMM create page: empty Classify dropdown → submit disabled + helper text; populated → defaults to most recent and submit enabled.
- HMM detail page: Regenerate runs to completion; chart re-renders with new bucket counts; exemplar chips appear with type names; `(background)` chip on background exemplars.
- HMM detail page: Re-bind via dialog updates header badge text and chart contents; second regenerate without arg keeps the binding.
- MT detail page: Regenerate works irrespective of active `?k=`; switching `k` after regenerate shows updated chart for the new k.
- Vocalization Labeling workspace remains functional and visually unchanged (smoke).

### 8.6 Type/lint gates

- `cd frontend && npx tsc --noEmit` — clean.
- `uv run ruff format --check` and `uv run ruff check` — clean on modified files.
- `uv run pyright` — full run (loader Protocol changed).

### 8.7 Manual smoke before PR

1. **Backup prod DB** per CLAUDE.md §3.5 (Acceptance Criterion #1 of the migration step).
2. `uv run alembic upgrade head` to land migration 065.
3. `rm -rf data/sequence_models/hmm_sequence_jobs/* data/sequence_models/masked_transformer_jobs/*` (SQL rows already deleted via UI).
4. Submit one HMM job and one Masked Transformer job against an existing `EventSegmentationJob` with completed Classify; open both detail pages; confirm chart renders, exemplar chips render, regenerate works, re-binding to a different Classify job (if available) works.
5. Open Vocalization Labeling workspace — confirm unchanged.
6. `cd frontend && npx playwright test` — full pass.
7. `uv run pytest tests/` — full pass.

### 8.8 Project verification gates (CLAUDE.md §10.2)

In order:

1. `uv run ruff format --check` on modified Python files.
2. `uv run ruff check` on modified Python files.
3. `uv run pyright` (full).
4. `uv run pytest tests/`.
5. `cd frontend && npx tsc --noEmit`.
6. `cd frontend && npx playwright test`.

---

## 9. Documentation updates

Per CLAUDE.md §10.2 doc-update matrix:

- **DECISIONS.md** — append new ADR: *"Sequence Models label source switched to Call Parsing Classify"* (supersedes ADR-060 explicitly; references ADR-054 for correction overlay, ADR-059 for loader Protocol).
- **CLAUDE.md §9.1** — update Sequence Models bullet: state-to-label distribution now sourced from `EventClassificationJob` + `EventTypeCorrection` overlay; tier dimension removed from label-distribution artifacts (CRNN tier metadata persists on `decoded.parquet`/exemplars).
- **CLAUDE.md §9.2** — bump latest migration to `065_sequence_models_classify_binding.py`.
- **`docs/reference/sequence-models-api.md`** — add the two `regenerate-label-distribution` endpoints; document the `event_classification_job_id` field on submit.
- **`docs/reference/data-model.md`** — note the new FK columns on `hmm_sequence_jobs` and `masked_transformer_jobs`.
- **`docs/reference/behavioral-constraints.md`** — note the submit-time precondition (Classify required) and the manual-regenerate semantics.

---

## 10. Out-of-scope follow-ups

- Auto-recompute on correction writes (Q7:C). Possible later if pain emerges.
- Configurable confidence-threshold knob on submit (Q4:C). Add when a concrete need surfaces.
- Color `overlay.parquet` scatter by label (Q3:C). Bigger UI lift than its incremental value today.
- Bias exemplar selection toward labeled (non-background) windows. Out of scope; current selection logic preserved.
- Re-bind Classify job from a "regenerate" affordance after initial submission is supported; a standalone "edit binding" UI without regenerate is not added.
