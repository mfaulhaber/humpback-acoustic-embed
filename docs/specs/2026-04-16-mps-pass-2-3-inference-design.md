# MPS Inference for Pass 2 & 3 (Call Parsing)

**Status:** Approved
**Date:** 2026-04-16

## Goal

Run Pass 2 (event segmentation) and Pass 3 (event classification) inference on Apple Silicon GPU (MPS) when available, with a per-job validated fallback to CPU. Surface the chosen device to the UI.

## Background

On Apple Silicon, training for Pass 2 (CRNN) and Pass 3 (CNN) already uses MPS via `select_device()` in `src/humpback/ml/device.py`, which prefers MPS → CUDA → CPU and honors `HUMPBACK_FORCE_CPU=1`. Inference, however, leaves both models on CPU:

- `src/humpback/call_parsing/segmentation/inference.py` — `_infer_single` builds the feature tensor with `.float()` and calls `model(...)` without moving either to a device; `event_segmentation_worker.py:109` loads the checkpoint and calls `model.eval()` with no device move.
- `src/humpback/call_parsing/event_classifier/inference.py:105` — `classify_events` already accepts `device: torch.device | None = None` but defaults to CPU, and the worker (`event_classification_worker.py:73`) never passes one.

Pass 1 region detection uses a TF2 SavedModel (SurfPerch/Perch) that already attempts Metal GPU with validation and falls back to CPU internally — not in scope for this change.

## Design decisions

### 1. Fallback strategy — per-job load-time validation

Mirror the `TF2SavedModel` pattern in `src/humpback/processing/inference.py:318`:

- On job start, after loading the checkpoint, run one forward pass on CPU and one on the target device (MPS or CUDA) with a deterministic dummy input.
- Compare outputs with `torch.allclose(rtol=1e-4, atol=1e-5)`.
- If outputs match within tolerance, commit to the target device for the rest of the job.
- If the target-device forward raises **or** outputs diverge, log a warning and use CPU for the whole job.
- Record the chosen device and (if applicable) the fallback reason on the job row.

Mid-job failures after validation are **not** recovered — the job fails loudly. Post-validation divergence is almost always a hard error (OOM, etc.), not something CPU would recover from silently.

### 2. Shared helper in `ml/device.py`

Validation logic is nearly identical for the CRNN and the CNN; only the dummy input shape differs. A single helper keeps device policy centralized:

```
select_and_validate_device(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> tuple[torch.device, str | None]
```

Returns `(device, fallback_reason)`. `fallback_reason` is `None` when CPU was picked cleanly (no GPU available, or `HUMPBACK_FORCE_CPU=1`) **and** when the target device succeeded. It is non-`None` only when a non-CPU device was attempted and rejected.

Fallback reason values:
- `"mps_load_error"` / `"cuda_load_error"` — forward on target device raised
- `"mps_output_mismatch"` / `"cuda_output_mismatch"` — outputs diverged beyond tolerance

The helper mutates the model in place (`model.to(chosen_device)`). If CPU is chosen because `select_device()` returned CPU, the helper short-circuits without running validation.

Tolerances live as module-level constants in `ml/device.py` and are tunable if false fallbacks appear in practice.

### 3. Device plumbing in inference modules

**Pass 2** (`call_parsing/segmentation/inference.py`):
- `_infer_single` gains a `device: torch.device` parameter; moves the feature tensor to the device before forward.
- `run_inference` gains `device` and forwards it through `_infer_windowed` → `_infer_single`.
- `event_segmentation_worker.py` builds a deterministic sample input with shape `(1, 1, n_mels=64, ~500 frames)` (matches a ~5s region at the default hop), calls the helper, records the device on the job row, passes the device into `run_inference`.

**Pass 3** (`call_parsing/event_classifier/inference.py`):
- `classify_events` already accepts `device=`; no API change needed.
- `event_classification_worker.py` builds a deterministic sample input with shape `(1, 1, n_mels=64, ~150 frames)` (matches a ~1.5s event crop), calls the helper, records the device on the job row, passes the device into `classify_events`.

### 4. Schema: two columns per job table

Alembic migration `048_job_compute_device.py` adds to both `event_segmentation_jobs` and `event_classification_jobs`:

- `compute_device TEXT NULL` — `"mps"` | `"cuda"` | `"cpu"`
- `gpu_fallback_reason TEXT NULL` — null on successful GPU use, null when GPU was not attempted, non-null only when GPU was attempted and rejected

Both nullable so existing rows keep working. Written by the worker at job start, in the same transaction that moves the row to `running`.

### 5. UI — shared badge component

A new `<ComputeDeviceBadge device fallbackReason />` component in `frontend/`, rendered on the Pass 2 and Pass 3 job detail pages adjacent to the existing status badge.

- `mps` → green "MPS"
- `cuda` → green "CUDA"
- `cpu`, null reason → neutral "CPU"
- `cpu`, non-null reason → yellow "CPU (fallback: <reason>)" with the reason visible
- `null` device (pre-migration row, or job that hasn't started) → renders nothing

## Testing

**Unit tests** for `select_and_validate_device` (`tests/ml/test_device.py`):
- `HUMPBACK_FORCE_CPU=1` → `(cpu, None)`, model stays on CPU, no validation forward calls
- No GPU available (monkeypatched `select_device`) → `(cpu, None)`, no validation forward calls
- Monkeypatched `select_device` returns non-CPU + injected model whose forward raises on the target device → `(cpu, "mps_load_error"`-family reason)
- Same, but outputs diverge beyond tolerance → `(cpu, "mps_output_mismatch"`-family reason)
- Happy path, gated on `@pytest.mark.skipif(not torch.backends.mps.is_available())`: real MPS forward matches CPU → `(mps, None)`

**Worker integration tests** — extend Pass 2 and Pass 3 worker tests. Run jobs under `HUMPBACK_FORCE_CPU=1`, assert the DB row has `compute_device == "cpu"` and `gpu_fallback_reason is None`.

**Frontend** — Playwright test on both job detail pages verifies the `ComputeDeviceBadge` renders the three shapes (cpu, mps, cpu-with-fallback) from mocked API responses.

## Edge cases

- **BiGRU on MPS** — historically patchy; the validation step is exactly what catches divergence. Users see a yellow badge with the reason.
- **Hydrophone Pass 2** — processes multiple chunks, but device choice is fixed once per job at validation and reused across chunks.
- **Mid-job MPS failure after validation** — not handled; job fails loudly. Acceptable under option B's per-job granularity.
- **Back-compat** — old job rows have `NULL` in both columns; the UI renders nothing, which is correct.

## Non-goals

- Pass 1 region detection (already has its own GPU path with fallback inside `TF2SavedModel`).
- Training workers (already use MPS successfully; device reporting for training is a separate observability concern).
- Vocalization pipelines, binary classifier, or any other ML pipeline.
- Per-inference fallback granularity (rejected in favor of per-job).
- Device reporting for any table other than `event_segmentation_jobs` and `event_classification_jobs`.

## File-level change summary

| File | Change |
|---|---|
| `src/humpback/ml/device.py` | New `select_and_validate_device()` helper + tolerance constants |
| `src/humpback/call_parsing/segmentation/inference.py` | `_infer_single` and `run_inference` accept `device`; tensors moved with `.to(device)` |
| `src/humpback/workers/event_segmentation_worker.py` | Build sample input, call helper, record device on job row, pass device into `run_inference` |
| `src/humpback/workers/event_classification_worker.py` | Same pattern; pass `device` into existing `classify_events` |
| `alembic/versions/048_job_compute_device.py` | New migration, two columns on each of two tables |
| `src/humpback/database.py` (or the call_parsing model module) | New columns on both SQLAlchemy models |
| `src/humpback/schemas/call_parsing.py` | New optional response fields on segmentation-job and classification-job schemas |
| `frontend/src/components/ComputeDeviceBadge.tsx` (new) | Shared badge component |
| Pass 2 and Pass 3 job detail pages (frontend) | Render `<ComputeDeviceBadge>` next to status badge |
| `docs/reference/call-parsing-api.md` | Note new response fields |
| `docs/reference/data-model.md` | Note new columns |
| `tests/ml/test_device.py` | Unit tests for the helper |
| Existing Pass 2 and Pass 3 worker tests | Assert compute_device persisted under force-CPU |
| Frontend Playwright test | Assert badge renders the three shapes |
