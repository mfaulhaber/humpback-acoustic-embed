# Audio Loader Consolidation — Design Spec

**Date:** 2026-04-15
**Motivation:** Five independent `_build_audio_loader` implementations across four workers re-derive the same pattern (resolve audio for a time range, return an array) with subtle differences in coordinate handling, return types, and caching. The event classifier training collapse (#117) was caused by one copy getting the relative-vs-absolute coordinate conversion wrong.

**Report:** `docs/reports/2026-04-15-audio-loader-consolidation.md`

---

## Problem

Six private audio loader implementations exist across five files. Each independently converts between relative offsets and absolute timestamps, wraps `resolve_timeline_audio`, and decides its own caching strategy. This duplication caused a training collapse bug and creates ongoing risk of coordinate conversion errors.

Two distinct consumer protocols exist:
- **Event classification** consumers expect `Callable[[sample], tuple[np.ndarray, float]]` — a buffer plus the offset of that buffer's start, so the dataset can compute crop positions.
- **Segmentation** consumers expect `Callable[[sample], np.ndarray]` — a pre-sliced waveform for a region.

These are not the same contract and should not be forced into one.

---

## Design

### Module: `src/humpback/call_parsing/audio_loader.py`

A shared module with two layers: an internal `CachedAudioSource` that owns coordinate conversion and caching, and two thin factory functions that produce the right protocol for each consumer family.

### Layer 1: CachedAudioSource

A class wrapping either a file-based or hydrophone-based audio source. Constructed via two classmethods:

- **`CachedAudioSource.from_file(audio_file, target_sr, storage_root)`** — Decodes the audio file once at construction. Holds the full waveform in memory. Returns `(audio, 0.0)` from `get_audio()`.

- **`CachedAudioSource.from_hydrophone(hydrophone_id, job_start_ts, job_end_ts, target_sr, settings, preload_span=None)`** — If `preload_span=(rel_start, rel_end)` is given (seconds relative to job start), calls `resolve_timeline_audio` once at construction with `start_sec=job_start_ts + rel_start`. Otherwise loads per-request.

Exposes one method:

- **`get_audio(rel_start_sec, duration_sec) -> tuple[np.ndarray, float]`** — Arguments are relative to job start. Returns `(audio_buffer, buffer_start_relative)`. For file-based sources, the arguments are ignored and the full cached waveform is always returned with offset `0.0` (the factories handle slicing). For pre-loaded hydrophone sources, returns the cached buffer and its relative start offset. For per-request hydrophone sources, calls `resolve_timeline_audio` with `start_sec=job_start_ts + rel_start_sec`.

**Coordinate conversion happens exactly here.** The `job_start_ts + relative_offset` → absolute timestamp calculation is done inside `CachedAudioSource` only. No caller performs this conversion.

### Layer 2: Factory Functions

#### `build_event_audio_loader`

```python
def build_event_audio_loader(
    *,
    target_sr: int,
    settings: Settings,
    # File-based source (provide one group or the other):
    audio_file: AudioFile | None = None,
    storage_root: Path | None = None,
    # Hydrophone source:
    hydrophone_id: str | None = None,
    job_start_ts: float | None = None,
    job_end_ts: float | None = None,
    # Optional pre-loading (hydrophone only):
    preload_events: Sequence[Any] | None = None,
) -> Callable[[Any], tuple[np.ndarray, float]]:
```

If `preload_events` is provided, computes the bounding span across all events (using `start_sec` / `end_sec` attributes) with context padding, and passes it as `preload_span` to `CachedAudioSource`.

**Context padding strategy** (adopted from the feedback worker's approach):
- `context_sec = max(10.0, event_duration)` for each event
- Symmetric padding: `pad = (context_sec - duration) / 2.0`
- Clamped to `[0.0, job_end_ts - job_start_ts]`
- Pre-load span = `(min_padded_start, max_padded_end)` across all events

Returns a closure: `(event) -> (audio_buffer, buffer_start_relative)`.

#### `build_region_audio_loader`

```python
def build_region_audio_loader(
    *,
    target_sr: int,
    settings: Settings,
    # File-based source:
    audio_file: AudioFile | None = None,
    storage_root: Path | None = None,
    # Hydrophone source:
    hydrophone_id: str | None = None,
    job_start_ts: float | None = None,
    job_end_ts: float | None = None,
    # Optional pre-loading (hydrophone only):
    preload_span: tuple[float, float] | None = None,
) -> Callable[[Any], np.ndarray]:
```

Returns a closure: `(region) -> np.ndarray`. The closure reads `padded_start_sec` / `padded_end_sec` from the region, calls `source.get_audio()`, and slices the returned buffer to the exact region span. For file-based sources, slicing uses sample indices directly from the cached waveform.

---

## Worker Migration

| Worker | Current function | Replaced by |
|--------|-----------------|-------------|
| `event_classification_worker.py` | `_build_audio_loader` (file) | `build_event_audio_loader(audio_file=...)` |
| `event_classification_worker.py` | `_build_hydrophone_audio_loader` | `build_event_audio_loader(hydrophone_id=..., preload_events=events)` |
| `event_classifier_feedback_worker.py` | `_build_audio_loader` | `build_event_audio_loader(hydrophone_id=..., preload_events=samples)` |
| `segmentation_training_worker.py` | `_build_audio_loader` | `build_region_audio_loader(hydrophone_id=..., preload_span=...)` computed from sample bounds |
| `event_segmentation_worker.py` | `_build_file_audio_loader` | `build_region_audio_loader(audio_file=...)` |
| `event_segmentation_worker.py` | `_build_hydrophone_audio_loader` | `build_region_audio_loader(hydrophone_id=...)` |
| `scripts/bootstrap_classifier.py` | `_build_audio_loader` | `build_event_audio_loader(hydrophone_id=..., preload_events=samples)` |

All six private functions are deleted. Existing protocol type aliases (`AudioLoader`, `EventAudioLoader`, `RegionAudioLoader`) in `dataset.py` and `inference.py` remain unchanged — the factories produce callables that match them.

---

## Test Plan

Test module: `tests/test_audio_loader.py`

All hydrophone tests mock `resolve_timeline_audio` to verify arguments and call counts without network access.

### Coordinate Conversion
1. Relative offset + job_start = correct absolute timestamp passed to `resolve_timeline_audio`
2. Event at job start (start_sec = 0.0) — no negative absolute timestamps
3. Event at job end — context padding does not exceed job_end_ts
4. Multiple events spanning nearly the full job range — preload span covers all with context
5. Event shorter than context minimum (< 10s) — padding expands symmetrically
6. Event longer than context minimum (> 10s) — context scales with duration

### Pre-load Span Caching
7. Pre-loaded source returns same buffer reference for different events (confirms caching)
8. Pre-load span with events that have identical start/end (degenerate span)
9. Empty event list passed to preload — graceful fallback to per-sample loading
10. Events from different hydrophone regions within one job

### Boundary / Degenerate Inputs
11. Zero-duration event (start_sec == end_sec) — no crash, returns valid audio
12. Very short event (< 1 sample at target_sr) — returns non-empty array
13. Context padding clamped to job bounds on both sides simultaneously (short job)
14. File-based source with very short audio (shorter than event's expected span)

### Protocol Contracts
15. Event factory always returns `tuple[np.ndarray, float]`, never bare ndarray
16. Region factory always returns `np.ndarray`, never tuple
17. Region factory slices correctly — returned array length matches `(padded_end - padded_start) * sr` within rounding tolerance
18. File-based event loader always returns offset `0.0`

### Integration (mock resolve_timeline_audio)
19. `resolve_timeline_audio` called exactly once when `preload_span` is set, regardless of event count
20. `resolve_timeline_audio` called N times when no `preload_span` is set and N samples are requested

---

## Scope Boundaries

**In scope:**
- New shared module with `CachedAudioSource` + two factory functions
- Replace all six private implementations
- Comprehensive test suite for the new module
- Delete dead private functions after migration

**Out of scope:**
- Changing the consumer-side protocol types (`AudioLoader`, `EventAudioLoader`, `RegionAudioLoader`)
- Changing how datasets use the loaded audio (crop logic stays in dataset classes)
- Performance optimization beyond the pre-load caching already described
- Any UI or API changes
