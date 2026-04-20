# Autoresearch Control Plane Implementation Plan

**Goal:** Enable a Claude-based autoresearch agent to optimize the full call parsing pipeline by exposing a tiered control plane with numeric dials, strategy registries, agent-authored algorithms, gold standard evaluation, and cross-run memory.
**Spec:** [docs/specs/2026-04-20-autoresearch-control-plane-design.md](../specs/2026-04-20-autoresearch-control-plane-design.md)

---

### Task 1: Pass 1 Region Correction — Data Model + API

**Files:**
- Create: `alembic/versions/052_region_boundary_corrections.py`
- Modify: `src/humpback/database.py`
- Modify: `src/humpback/models/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `region_boundary_corrections` table created with columns: `id`, `region_detection_job_id` (FK), `region_id` (UUID string), `correction_type` (adjust/add/delete), `start_sec`, `end_sec`, `created_at`
- [ ] Unique constraint on `(region_detection_job_id, region_id)`
- [ ] Pydantic schemas for `RegionCorrectionCreate`, `RegionCorrectionResponse`
- [ ] `POST /region-detection-jobs/{id}/corrections` endpoint upserts corrections
- [ ] `GET /region-detection-jobs/{id}/corrections` endpoint returns corrections for a job
- [ ] Migration runs cleanly with `uv run alembic upgrade head`

**Tests needed:**
- Migration up/down
- Upsert semantics (create, then update same region_id)
- API endpoint round-trip (POST then GET)
- Validation: correction_type must be one of adjust/add/delete
- Validation: add corrections require start_sec and end_sec; delete corrections do not

---

### Task 2: Pass 1 Region Correction — Read-Time Overlay

**Files:**
- Create: `src/humpback/call_parsing/regions_overlay.py`
- Modify: `src/humpback/workers/event_segmentation_worker.py`

**Acceptance criteria:**
- [ ] `load_corrected_regions()` function reads `regions.parquet` and applies SQL corrections (adjust boundaries, insert added regions, remove deleted regions)
- [ ] Function returns a list of `Region` objects with corrections merged
- [ ] Pass 2 event segmentation worker calls `load_corrected_regions()` instead of reading raw `regions.parquet` directly
- [ ] Original `regions.parquet` is never modified

**Tests needed:**
- Overlay with no corrections returns original regions unchanged
- Adjust correction modifies region boundaries
- Add correction inserts a new region with generated UUID
- Delete correction removes the region from the list
- Multiple corrections on the same job compose correctly

---

### Task 3: Pass 1 Region Correction — Frontend UI

**Files:**
- Modify: `frontend/src/pages/` (region detection timeline viewer component)
- Modify: `frontend/src/api/` (API client additions for correction endpoints)

**Acceptance criteria:**
- [ ] Region detection timeline viewer has an editing mode toggle
- [ ] In editing mode, users can adjust region boundaries (drag start/end)
- [ ] In editing mode, users can add new regions (click + drag to define bounds)
- [ ] In editing mode, users can delete regions (select + delete action)
- [ ] Corrections are submitted via POST to the corrections endpoint
- [ ] Existing corrections are loaded and displayed on the timeline

**Tests needed:**
- Playwright: toggle editing mode on/off
- Playwright: submit a boundary adjustment correction and verify it persists on reload
- TypeScript type-check passes

---

### Task 4: Gold Standard Data Model + Promotion API

**Files:**
- Create: `alembic/versions/053_gold_standard.py`
- Modify: `src/humpback/database.py`
- Create: `src/humpback/models/gold_standard.py`
- Create: `src/humpback/schemas/gold_standard.py`
- Create: `src/humpback/services/gold_standard.py`
- Modify: `src/humpback/api/routers/call_parsing.py` (or create new router)

**Acceptance criteria:**
- [ ] `gold_standard_segments` table with columns: `id`, `audio_file_id` (nullable FK), `hydrophone_id`/`start_timestamp`/`end_timestamp` (nullable), `region_detection_job_id`, `event_segmentation_job_id`, `event_classification_job_id`, `promoted_at`, `promoted_by` (string), `notes`
- [ ] Promotion endpoint: `POST /gold-standard/promote` accepts job triple + optional notes
- [ ] Validation: all three correction types (region, boundary, type) must exist for the referenced jobs before promotion is allowed
- [ ] `GET /gold-standard/summary` returns count of segments, events, type coverage, audio source diversity
- [ ] `GET /gold-standard/segments` returns paginated list of gold segments with their ground truth
- [ ] Migration runs cleanly

**Tests needed:**
- Promotion succeeds when corrections exist at all three levels
- Promotion fails when corrections are missing at any level
- Summary endpoint returns correct counts
- Duplicate promotion of the same segment is rejected or idempotent

---

### Task 5: Gold Standard Held-Out Enforcement + Evaluation Endpoint

**Files:**
- Modify: `src/humpback/call_parsing/segmentation/trainer.py` (gold exclusion in data loading)
- Modify: `src/humpback/call_parsing/event_classifier/trainer.py` (gold exclusion in data loading)
- Create: `src/humpback/services/gold_evaluation.py`
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `split_by_audio_source()` in both Pass 2 and Pass 3 trainers excludes samples whose audio source overlaps a gold-promoted segment
- [ ] `POST /gold-standard/evaluate` endpoint accepts a pipeline configuration (Pass 1 config + Pass 2 model + decoder config + Pass 3 model), runs full pipeline on gold audio, compares output to corrected ground truth
- [ ] Returns per-pass metrics: Pass 1 recall + false positive rate, Pass 2 boundary IoU + onset/offset MAE, Pass 3 event F1 + per-type F1 + correction rate
- [ ] Returns composite score: `event_f1 / correction_rate`
- [ ] Evaluation runs synchronously (or queued as a job if long-running)

**Tests needed:**
- Gold exclusion: training data assembly with a gold segment present excludes it
- Gold exclusion: training data assembly without gold segments is unchanged
- Evaluation endpoint returns correct metrics for a known gold set with known pipeline output
- Composite score computation is correct

---

### Task 6: Algorithm Protocol Definitions + Existing Implementation Wrappers

**Files:**
- Create: `src/humpback/autoresearch/protocols.py`
- Create: `src/humpback/autoresearch/__init__.py`
- Create: `src/humpback/autoresearch/builtin_adapters.py`

**Acceptance criteria:**
- [ ] Python Protocol classes defined for all 7 algorithm slots: `RegionDetector`, `RegionShaper`, `EventDecoder`, `FeatureNormalizer`, `InferenceWindower`, `ThresholdOptimizer`, `EventMatcher`
- [ ] Each Protocol has a clear docstring specifying the contract
- [ ] Wrapper classes for existing implementations that conform to each Protocol (e.g., `HysteresisRegionDetector` wraps `merge_detection_events`)
- [ ] All wrapper classes pass `isinstance` checks against their Protocol via `runtime_checkable`

**Tests needed:**
- Each wrapper class conforms to its Protocol
- Each wrapper produces identical output to the unwrapped function for the same inputs
- Protocol definitions are runtime-checkable

---

### Task 7: Tier 2a Strategy Registries

**Files:**
- Create: `src/humpback/autoresearch/registries.py`
- Create: `src/humpback/autoresearch/normalizers.py`
- Create: `src/humpback/autoresearch/decoders.py`
- Create: `src/humpback/autoresearch/losses.py`
- Modify: `src/humpback/call_parsing/segmentation/trainer.py` (accept registry-resolved normalizer/loss)
- Modify: `src/humpback/call_parsing/segmentation/inference.py` (accept registry-resolved normalizer/decoder)
- Modify: `src/humpback/call_parsing/event_classifier/trainer.py` (accept registry-resolved normalizer/loss)
- Modify: `src/humpback/call_parsing/event_classifier/inference.py` (accept registry-resolved normalizer)

**Acceptance criteria:**
- [ ] Registry class that maps string keys to Protocol-conforming implementations with config dicts
- [ ] Normalization registry with built-in entries: `zscore` (existing), `pcen` (new implementation)
- [ ] Decoder registry with built-in entries: `hysteresis` (existing), `peak_picking` (new implementation)
- [ ] Loss function registry with built-in entries: `bce` (existing), `focal` (new implementation)
- [ ] Pass 2 and Pass 3 training and inference pipelines accept registry-resolved implementations via their config
- [ ] Existing behavior unchanged when default registry entries are used

**Tests needed:**
- Registry lookup returns correct implementation for each key
- Registry raises clear error for unknown key
- PCEN normalizer produces valid output (correct shape, finite values)
- Peak-picking decoder produces valid events from synthetic frame probabilities
- Focal loss produces valid gradients
- End-to-end: Pass 2 training with `normalization: pcen` completes without error
- End-to-end: Pass 2 inference with `decoder: peak_picking` produces events

---

### Task 8: Tier 2b Dynamic Module Loader + Test Harness

**Files:**
- Create: `src/humpback/autoresearch/module_loader.py`
- Create: `src/humpback/autoresearch/test_harness.py`

**Acceptance criteria:**
- [ ] `load_authored_algorithm(path, protocol_class)` dynamically imports a Python file and returns an instance conforming to the Protocol
- [ ] Import allowlist enforcement: authored modules may only import from numpy, scipy, torch, librosa, and standard library; loader raises on disallowed imports
- [ ] `validate_authored_algorithm(path, protocol_class, test_module)` loads the implementation, runs the existing unit tests for that algorithm slot, and returns a pass/fail result with details
- [ ] Artifact directory convention: `{experiment_artifacts_dir}/authored_algorithms/{experiment_name}/{slot_name}.py`

**Tests needed:**
- Load a valid authored module that conforms to a Protocol
- Load an authored module with a disallowed import raises error
- Load an authored module that does not conform to Protocol raises error
- Test harness runs existing decoder tests against a mock authored decoder and reports pass/fail
- Invalid Python file (syntax error) produces a clear error message

---

### Task 9: Experiment Spec Schema + Result Logging

**Files:**
- Create: `src/humpback/autoresearch/experiment_spec.py`
- Create: `src/humpback/autoresearch/experiment_log.py`

**Acceptance criteria:**
- [ ] Pydantic models for `ExperimentSpec` (full pipeline config with per-pass dials, strategy switches, authored algorithm references, evaluation config)
- [ ] Pydantic models for `ExperimentResult` (per-pass metrics, composite score, vs-parent deltas, verdict, reasoning)
- [ ] Pydantic model for `AgentConfig` (model, effort, budget, allowed_tiers, guardrails, memory settings)
- [ ] Tier enforcement: `ExperimentSpec.validate_tiers(allowed_tiers)` raises if spec uses dials/switches/authoring from disallowed tiers
- [ ] `ExperimentLog` class that persists specs and results as YAML files in `{experiment_artifacts_dir}/experiments/`
- [ ] Log supports reading full history, reading by experiment name, and appending new entries

**Tests needed:**
- Spec validation accepts a valid full-tier spec
- Spec validation rejects Tier 2b usage when `allowed_tiers=[1, "2a"]`
- Spec validation rejects Tier 1 dial changes when `allowed_tiers=["2b"]`
- Result persistence round-trip (write then read)
- vs-parent delta computation is correct
- Agent config validation accepts valid config and rejects invalid

---

### Task 10: Experiment Runner

**Files:**
- Create: `src/humpback/autoresearch/runner.py`

**Acceptance criteria:**
- [ ] `ExperimentRunner` class takes an `ExperimentSpec`, executes the pipeline on gold audio, and returns an `ExperimentResult`
- [ ] Cascade invalidation: if Pass 1 config differs from parent, Passes 2+3 are re-run; if Pass 2 config differs, Pass 3 is re-run
- [ ] When `retrain: true`, triggers model training via existing training APIs and uses the new model for inference
- [ ] When authored algorithms are specified, loads them via the module loader and injects into the pipeline
- [ ] Multi-seed execution: runs evaluation `seed_count` times and averages metrics
- [ ] Training/inference consistency enforcement: same feature config and normalizer used in both paths
- [ ] Acceptance criteria evaluation: compares result to parent, checks floors, checks degradation thresholds
- [ ] Logs experiment spec before execution and result after

**Tests needed:**
- Runner executes a threshold-only experiment (no retraining) and returns metrics
- Runner detects cascade invalidation when Pass 1 config changes
- Runner rejects an experiment that uses a tier not in `allowed_tiers`
- Runner loads and uses an authored algorithm
- Acceptance criteria correctly accept an improving experiment
- Acceptance criteria correctly reject a degrading experiment

---

### Task 11: Run Memory System

**Files:**
- Create: `src/humpback/autoresearch/run_memory.py`

**Acceptance criteria:**
- [ ] Pydantic model for `RunMemory` (run_id, timestamp, agent_config_summary, champion, findings, dead_ends, recommendations)
- [ ] `RunMemoryStore` class that reads/writes YAML memory files in `{experiment_artifacts_dir}/autoresearch_memory/`
- [ ] `load_all_memories()` reads all memory files chronologically
- [ ] `write_memory(memory)` persists a new memory file with timestamped filename
- [ ] Memory files are append-only — `write_memory` never modifies existing files

**Tests needed:**
- Write a memory file and read it back
- Load multiple memory files returns them in chronological order
- Memory file schema validates correctly
- Dead ends with conditional revisit context are preserved

---

### Task 12: Agent Tool Definitions

**Files:**
- Create: `src/humpback/autoresearch/tools.py`
- Create: `src/humpback/autoresearch/tool_schemas.py`

**Acceptance criteria:**
- [ ] Claude-compatible tool definitions (JSON schema) for all 14 tools: `read_run_memories`, `read_correction_history`, `read_experiment_log`, `read_gold_set_summary`, `run_pass_1`, `run_pass_2_inference`, `run_pass_2_training`, `run_pass_3_inference`, `run_pass_3_training`, `evaluate_against_gold`, `write_algorithm`, `list_algorithm_slots`, `log_experiment`, `write_run_memory`
- [ ] Each tool function is a thin wrapper around existing services/APIs
- [ ] `write_algorithm` tool: accepts slot name + Python source code, writes to artifact directory, runs test harness, returns pass/fail
- [ ] `list_algorithm_slots` tool: returns Protocol names, docstrings, existing implementations, and test module paths
- [ ] Tool functions can be invoked programmatically for testing

**Tests needed:**
- Each tool function returns the expected schema
- `write_algorithm` with valid code passes and returns success
- `write_algorithm` with invalid code returns failure with clear error
- `list_algorithm_slots` returns all 7 Protocol slots
- `read_correction_history` returns corrections grouped by pass

---

### Task 13: Orchestrator Agent Prompt + Configuration

**Files:**
- Create: `src/humpback/autoresearch/orchestrator_prompt.py`
- Create: `src/humpback/autoresearch/orchestrator.py`

**Acceptance criteria:**
- [ ] System prompt for the autoresearch orchestrator Claude agent that includes: optimization objective, pipeline map summary, tool descriptions, budget management guidelines, dead end detection heuristics, memory usage instructions
- [ ] `AgentConfig` loading from YAML file
- [ ] Orchestrator entry point that loads config, reads memories, initializes tools, and starts the agent loop
- [ ] The orchestrator prompt instructs the agent to write a run memory at session end

**Tests needed:**
- Agent config loads from a valid YAML file
- Agent config rejects invalid configurations
- Orchestrator entry point initializes without error with a valid config
- System prompt includes all required sections

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/autoresearch/ src/humpback/services/gold_standard.py src/humpback/services/gold_evaluation.py src/humpback/call_parsing/regions_overlay.py`
2. `uv run ruff check src/humpback/autoresearch/ src/humpback/services/gold_standard.py src/humpback/services/gold_evaluation.py src/humpback/call_parsing/regions_overlay.py`
3. `uv run pyright src/humpback/autoresearch/ src/humpback/services/gold_standard.py src/humpback/services/gold_evaluation.py src/humpback/call_parsing/regions_overlay.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
