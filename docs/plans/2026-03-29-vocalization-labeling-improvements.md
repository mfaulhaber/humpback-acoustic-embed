# Vocalization Labeling Page Improvements — Implementation Plan

**Goal:** Improve the vocalization labeling page with source abstraction, local-state label management, spectrogram enhancements, and retrain visibility fixes.
**Spec:** `docs/specs/2026-03-29-vocalization-labeling-improvements-design.md`

---

### Task 1: Source Abstraction Type and Selector Component

**Files:**
- Create: `frontend/src/components/vocalization/SourceSelector.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationLabelingTab.tsx`
- Modify: `frontend/src/api/types.ts`

**Acceptance criteria:**
- [ ] `LabelingSource` discriminated union type defined in `types.ts`
- [ ] `SourceSelector` component renders segmented toggle with three options: Detection Jobs, Embedding Set, Local
- [ ] Detection Jobs mode renders existing hydrophone + local grouped dropdown
- [ ] Embedding Set mode renders dropdown populated from `fetchEmbeddingSets()`, display value = top-level folder name from `parquet_path`
- [ ] Local mode renders a text input for folder path
- [ ] `VocalizationLabelingTab` state changes from `selectedJobId: string` to `source: LabelingSource | null`
- [ ] Selecting a new source resets downstream state (inference, labels)

**Tests needed:**
- TypeScript compile check confirms `LabelingSource` type is correctly discriminated
- Manual verification: switching source types shows correct inputs, resets state

---

### Task 2: Adapt Pipeline Components to Source Abstraction

**Files:**
- Modify: `frontend/src/components/vocalization/VocalizationLabelingTab.tsx`
- Modify: `frontend/src/components/vocalization/EmbeddingStatusPanel.tsx`
- Modify: `frontend/src/components/vocalization/InferencePanel.tsx`

**Acceptance criteria:**
- [ ] `EmbeddingStatusPanel` accepts `LabelingSource` and adapts behavior: skipped for `embedding_set`, shown for `detection_job` and `local`
- [ ] `InferencePanel` accepts `LabelingSource` and uses correct `source_type`/`source_id` when creating inference jobs (detection_job ID, embedding_set ID, or folder-derived embedding set ID)
- [ ] For `embedding_set` source, inference creation uses `source_type: "embedding_set"` and `source_id: embeddingSetId`
- [ ] For `local` source, a backend call finds or creates an embedding set for the folder, then proceeds through embedding generation and inference
- [ ] Pipeline conditional rendering: `embedding_set` skips EmbeddingStatusPanel, `detection_job` and `local` show it

**Tests needed:**
- Manual verification: each source type triggers the correct pipeline steps
- Verify embedding_set source goes directly to inference without embedding generation

---

### Task 3: Backend Endpoint for Local Folder Embedding Set

**Files:**
- Modify: `src/humpback/api/routers/processing.py`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useVocalization.ts`

**Acceptance criteria:**
- [ ] New API endpoint `POST /processing/folder-embedding-set` that accepts `{ folder_path: string }` and returns the embedding set (finding existing or triggering processing)
- [ ] If a completed embedding set already exists for audio files in that folder, return it directly
- [ ] If not, import the audio folder (if not already imported) and create a processing job, returning status indicating processing is needed
- [ ] Frontend `client.ts` has a typed fetch wrapper for the new endpoint
- [ ] Frontend hook available for the local source pipeline

**Tests needed:**
- Unit test: endpoint returns existing embedding set when one exists for the folder
- Unit test: endpoint triggers processing when no embedding set exists
- Integration test: folder path -> embedding set creation flow

---

### Task 4: Label Local State Management

**Files:**
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`
- Modify: `frontend/src/hooks/queries/useLabeling.ts`

**Acceptance criteria:**
- [ ] `pendingAdds` state: Map keyed by `"${start_utc}_${end_utc}"`, value is `Set<string>` of label names
- [ ] `pendingRemovals` state: Map keyed by `"${start_utc}_${end_utc}"`, value is `Set<string>` of label IDs
- [ ] `isDirty` derived from non-empty pendingAdds or pendingRemovals
- [ ] Clicking an inference-suggested badge adds to `pendingAdds` (no API call)
- [ ] Clicking X on a saved label adds to `pendingRemovals` (no API call)
- [ ] Clicking + popover adds to `pendingAdds`
- [ ] Removing a pending add (before save) removes from `pendingAdds`
- [ ] Per-row label rendering merges saved labels, pending adds, and pending removals into a unified view
- [ ] All label mutations removed from click handlers (no `useAddVocalizationLabel` / `useDeleteVocalizationLabel` on click)

**Tests needed:**
- Manual verification: clicking labels updates visual state without network requests
- Verify pendingAdds and pendingRemovals accumulate correctly across pages

---

### Task 5: Save / Cancel Bar and Batch Persistence

**Files:**
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useLabeling.ts`

**Acceptance criteria:**
- [ ] Sticky bar appears when `isDirty` is true, showing unsaved change count, Cancel button, Save button
- [ ] Cancel discards all `pendingAdds` and `pendingRemovals`, reverts to last-saved state
- [ ] Save iterates pending changes: POST for each add, DELETE for each removal, then clears pending state
- [ ] On successful Save, `labelCount` in parent is incremented by the net new label count
- [ ] Save/Cancel bar hidden when source is `embedding_set` (readonly)
- [ ] Query invalidation happens once after the full batch, not per-operation

**Tests needed:**
- Manual verification: Save persists all pending changes, Cancel reverts
- Verify label count propagates to retrain footer after Save

---

### Task 6: Label Visual Feedback (Three States)

**Files:**
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] Inference-suggested labels (above threshold, not yet added): outline/muted badges, clickable to add
- [ ] Saved labels (persisted from previous sessions): solid badges, X button stages removal
- [ ] Pending new labels (in pendingAdds, unsaved): solid badges with a dot indicator, distinguishable from saved
- [ ] Pending removals: saved badges show dimmed/strikethrough state
- [ ] Readonly mode (`embedding_set` source): all badges non-interactive, no add/remove buttons, "View only" text in workspace header

**Tests needed:**
- Manual verification: three label states are visually distinct
- Verify readonly mode disables all label interaction

---

### Task 7: Default Sort Change

**Files:**
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] Initial `sortMode` state is `"score_desc"` instead of `"uncertainty"`

**Tests needed:**
- Manual verification: labeling workspace opens sorted by score descending

---

### Task 8: Retrain Footer Fix

**Files:**
- Modify: `frontend/src/components/vocalization/RetrainFooter.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationLabelingTab.tsx`

**Acceptance criteria:**
- [ ] `labelCount` incremented on successful batch Save (not on individual clicks)
- [ ] Retrain footer appears after first Save with new labels
- [ ] Retrain footer hidden when source is `embedding_set`
- [ ] Retrain button disabled when `labelCount === 0`
- [ ] Existing retrain logic (training job creation, polling, model activation) unchanged

**Tests needed:**
- Manual verification: retrain footer appears after saving labels
- Verify retrain footer hidden for embedding_set source

---

### Task 9: Double-Height Detection Items and Spectrogram Popup

**Files:**
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] Detection item row height approximately doubled
- [ ] Spectrogram image ~120x80px, filling the left portion of the card
- [ ] Audio controls, score, labels, and time info stack vertically to the right of the spectrogram
- [ ] Clicking the spectrogram opens a Dialog (shadcn) with a larger spectrogram image
- [ ] Dialog includes: large spectrogram, time range display, audio playback controls, close button
- [ ] No label editing in the dialog
- [ ] Dialog uses the same spectrogram endpoint (larger dimensions or natural size)

**Tests needed:**
- Manual verification: row height increased, spectrogram larger
- Verify click opens popup with larger spectrogram and audio playback
- Verify dialog closes cleanly

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `uv run ruff format --check src/humpback/api/routers/processing.py`
3. `uv run ruff check src/humpback/api/routers/processing.py`
4. `uv run pyright src/humpback/api/routers/processing.py`
5. `uv run pytest tests/`
6. Manual UI testing: exercise all three source types, verify label interaction, save/cancel, retrain, spectrogram popup
