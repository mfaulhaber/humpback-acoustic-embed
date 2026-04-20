# Autoresearch Control Plane for Call Parsing Pipeline

**Date:** 2026-04-20
**Status:** Draft
**Approach:** B — Experiment loop with strategy exploration

## 1. Optimization Objective

### Primary Objective

Maximize correctly classified vocalization events per unit of human review effort across the call parsing pipeline.

Operationalized as a composite metric:

- **End-to-end event accuracy**: Event-level F1 of vocalization type labels on the gold standard set. Measured by running the full pipeline (Pass 1 -> 2 -> 3) on gold audio and comparing typed events against gold truth.
- **Correction efficiency**: The fraction of events in the gold set that would require human correction (boundary adjustment OR type relabeling). Lower is better.

Primary score: `event_F1 / correction_rate`.

### Secondary Objectives

- Pass 1 detection recall must not degrade below a configurable floor
- Improvements must hold across at least 2 distinct audio sources in the gold set
- Changes must be expressible as configuration or bounded strategy switches
- Experiment results must reproduce across 2 repeated runs with different random seeds

### Acceptance Criteria

A candidate change is an improvement only if:

1. The composite score improves
2. Pass 1 recall stays above floor
3. Result reproduces across seeds
4. No single-pass metric degrades catastrophically (>15% relative drop)

---

## 2. Pipeline Map

### Pass 1 — Region Detection

No trainable model in this pass. Dense Perch inference produces per-window scores; hysteresis thresholding produces padded regions.

- **Agent dials**: `high_threshold`, `low_threshold`, `padding_sec`, `min_region_duration_sec`, classifier model selection
- **Error propagation**: Missed regions are invisible to all downstream passes. False regions waste human review time and can pollute training data. Highest-leverage pass for the agent to tune.
- **Human effort**: Boundary review and correction (adjust/add/delete regions). Currently unimplemented — prerequisite for this project.

### Pass 2 — Event Segmentation (SegmentationCRNN, ~300k params)

Log-mel spectrogram -> 4x Conv2d + BiGRU -> framewise boundary predictions -> hysteresis decoder -> events.

- **Agent dials**: Training hyperparameters (lr, epochs, architecture), feature config (n_mels, fmin/fmax, normalization), decoder config (thresholds, min_event_sec, merge_gap_sec), strategy switches (normalization method, decoder algorithm, loss function)
- **Error propagation**: Bad boundaries mean Pass 3 classifies incorrectly-cropped audio. Too many merged events lose fine structure. Too many split events create noise.
- **Human effort**: Boundary review and correction (adjust/add/delete events)

### Pass 3 — Event Classification (EventClassifierCNN)

Per-event log-mel spectrogram -> 4x Conv2d -> AdaptiveAvgPool -> multi-label classification.

- **Agent dials**: Training hyperparameters, feature config, per-type thresholds, strategy switches (normalization, loss weighting), training data composition (corrections_only flag, min_examples_per_type)
- **Error propagation**: Terminal pass — errors here are the final output quality
- **Human effort**: Type label review and correction

### Cross-Pass Interactions

- Lowering Pass 1 `high_threshold` -> more regions -> more events -> better recall but more noise for Pass 3
- Changing Pass 2 `merge_gap_sec` -> different event boundaries -> Pass 3 sees different spectrograms -> different classification accuracy
- Retraining Pass 2 with new corrections -> different event boundaries on all audio -> Pass 3 model trained on old boundaries may degrade (requires Pass 3 retrain too)

---

## 3. Gold Standard Evaluation Set

### Purpose

A curated set of audio segments with human-verified ground truth at every pipeline level. The autoresearch agent evaluates against this set but never trains on it.

### Structure

| Pass | Gold truth | What it captures |
|------|-----------|-----------------|
| Pass 1 | Region boundaries (start/end UTC) + confirmed negatives | Detection recall and false positive rate |
| Pass 2 | Exact event boundaries within each region | Boundary accuracy (onset/offset MAE, IoU) |
| Pass 3 | Vocalization type label(s) per event | Classification F1, per-type precision/recall |

Each gold item is a segment of audio with all three levels of annotation.

### Promotion Workflow

Gold items are promoted from the existing correction workflow:

1. A human reviews a segment in the Pass 1 region timeline viewer, corrects region boundaries
2. The same segment flows through Pass 2, human reviews and corrects event boundaries
3. The segment flows through Pass 3, human reviews and corrects type labels
4. Once corrections exist at all three levels, the segment becomes a gold candidate
5. A human explicitly promotes the segment to gold status, marking it as held-out from training
6. Promotion is recorded with a timestamp so the agent knows when the gold set changed

### Held-Out Enforcement

When the agent triggers retraining, the training data assembly pipeline must exclude all gold-promoted segments. The gold exclusion filter is applied during `split_by_audio_source()` and any other data loading paths.

### Minimum Viable Gold Set

- At least 3 distinct audio sources (mix of hydrophone and uploaded files)
- At least 50 events with type labels
- Coverage of the most common vocalization types (at least 5 examples per type)
- Both true positive regions and confirmed negative regions

---

## 4. Control Plane

### Tier 1 — Numeric Dials

Scalar parameters the agent can sweep. Bounded ranges enforced by the experiment spec schema.

| Pass | Parameter | Default | Description |
|------|-----------|---------|-------------|
| 1 | `high_threshold` | 0.70 | Hysteresis upper bound |
| 1 | `low_threshold` | 0.45 | Hysteresis lower bound |
| 1 | `padding_sec` | 1.0 | Symmetric padding around regions |
| 1 | `min_region_duration_sec` | 0.0 | Minimum region length filter |
| 2 | `learning_rate` | 1e-3 | Training step size |
| 2 | `weight_decay` | 1e-4 | Regularization strength |
| 2 | `epochs` | 30 | Max training epochs |
| 2 | `early_stopping_patience` | 5 | Epochs without improvement before stop |
| 2 | decoder `high_threshold` | 0.5 | Event boundary sensitivity |
| 2 | decoder `low_threshold` | 0.3 | Event termination sensitivity |
| 2 | decoder `min_event_sec` | 0.2 | Minimum event duration |
| 2 | decoder `merge_gap_sec` | 0.1 | Merge gap for adjacent events |
| 3 | `learning_rate` | 1e-3 | Training step size |
| 3 | `weight_decay` | 1e-4 | Regularization strength |
| 3 | `epochs` | 30 | Max training epochs |
| 3 | `min_examples_per_type` | varies | Drop rare types from training |
| 3 | per-type thresholds | F1-optimized | Precision/recall tradeoff per type |

### Tier 2a — Built-In Strategy Switches

Enumerated choices among known algorithms that ship with the codebase. Implemented via a registry pattern: each switch has a string key, a registry of implementations, and a config dict.

| Pass | Switch | Current | Candidates |
|------|--------|---------|-----------|
| 2 | Normalization | `per_region_zscore` | zscore, PCEN, per-channel energy norm |
| 2 | Decoder algorithm | Hysteresis | Hysteresis, peak-picking, learned threshold |
| 2 | Architecture variant | CRNN (Conv+BiGRU) | CRNN, Conv-only, UNet-style |
| 2 | Loss function | BCE w/ auto pos_weight | BCE, focal loss, dice loss |
| 2 | Train/val split | `split_by_audio_source` | by-source, temporal, stratified random |
| 3 | Normalization | `per_region_zscore` | Same as Pass 2 |
| 3 | Architecture variant | Conv4+AvgPool | Conv4, ResNet-small, attention pooling |
| 3 | Loss function | BCE w/ per-type pos_weight | BCE, focal loss, asymmetric loss |
| 3 | `corrections_only` | false | true/false (exclude bootstrap data) |
| 1 | Classifier model | user-selected | Agent selects from available models |

Tier 2a switches are the starter set. They cover the most common alternatives and can be selected by name in the experiment spec.

### Tier 2b — Agent-Authored Algorithms

The agent can write new algorithm implementations at runtime, going beyond the pre-built registry. Each pluggable algorithm slot is defined by a Python Protocol specifying its input/output contract. The agent writes a new implementation conforming to the Protocol, the experiment runner validates it against existing unit tests, and if tests pass, the implementation is used in the experiment.

#### Protocol Definitions

```python
class RegionDetector(Protocol):
    """Pass 1: Convert per-window scores into merged regions."""
    def detect(self, scores: np.ndarray, config: dict) -> list[Region]: ...

class RegionShaper(Protocol):
    """Pass 1: Pad, merge, and filter raw detection events into regions."""
    def shape(self, events: list[dict], audio_duration_sec: float,
              config: dict) -> list[Region]: ...

class EventDecoder(Protocol):
    """Pass 2: Convert framewise probabilities into discrete events."""
    def decode(self, frame_probs: np.ndarray, region_id: str,
               region_start_sec: float, hop_sec: float,
               config: dict) -> list[Event]: ...

class FeatureNormalizer(Protocol):
    """Passes 2+3: Normalize a spectrogram before model input."""
    def normalize(self, spectrogram: np.ndarray) -> np.ndarray: ...

class InferenceWindower(Protocol):
    """Pass 2: Strategy for running inference on long audio."""
    def infer(self, model: nn.Module, audio: np.ndarray,
              feature_config: dict, device: str) -> np.ndarray: ...

class ThresholdOptimizer(Protocol):
    """Pass 3: Find optimal per-type classification thresholds."""
    def optimize(self, scores: np.ndarray,
                 labels: np.ndarray) -> tuple[float, float]: ...

class EventMatcher(Protocol):
    """Evaluation: Match predicted events to ground truth."""
    def match(self, pred_events: list[Event],
              gt_events: list[Event],
              config: dict) -> EventMatchResult: ...
```

#### Algorithm Inventory

Algorithms that currently exist as hard-wired implementations and are candidates for agent rewriting:

| Algorithm | Pass | File | Lines | Contract | Complexity |
|-----------|------|------|-------|----------|------------|
| Hysteresis merging | 1 | `classifier/detector_utils.py` | ~55 | `scores + thresholds → regions` | Pure stateless, O(n) |
| Region decode & pad | 1 | `call_parsing/regions.py` | ~82 | `events + config → padded regions` | Pure stateless, O(n log n) |
| Framewise hysteresis | 2 | `segmentation/decoder.py` | ~79 | `frame_probs + config → events` | Pure stateless, O(n) |
| Log-mel + normalize | 2+3 | `segmentation/features.py` | ~48 | `audio + config → spectrogram` | Pure, librosa dependency |
| Windowed inference | 2 | `segmentation/inference.py` | ~51 | `model + long_audio → frame_probs` | 50% overlap averaging |
| Masked BCE loss | 2 | `segmentation/trainer.py` | ~33 | `logits + targets + mask → loss` | PyTorch nn.Module |
| Train/val split | 2+3 | `segmentation/trainer.py` | ~44 | `samples + config → train, val` | Pure stateless |
| IoU event matching | eval | `segmentation/trainer.py` | ~67 | `pred + gt → matches` | Greedy O(n*m) |
| Per-type threshold opt | 3 | `event_classifier/trainer.py` | ~22 | `scores + labels → threshold` | Grid search |
| Per-type pos-weight | 3 | `event_classifier/trainer.py` | ~14 | `samples → weight vector` | Pure O(n) |
| Multi-label post-proc | 3 | `event_classifier/inference.py` | ~78 | `logits + thresholds → typed events` | Includes fallback rule |

All algorithms are pure functions or lightweight nn.Modules with existing unit tests. Each is under 100 lines. The agent can write a replacement, run the existing test suite, and know immediately if it's valid.

#### Agent Authoring Workflow

1. Agent decides to try a new approach (e.g., "adaptive threshold decoder that uses local signal energy instead of fixed thresholds")
2. Agent writes a Python file implementing the relevant Protocol
3. File is saved to `{experiment_artifacts_dir}/authored_algorithms/{experiment_name}/`
4. Experiment runner dynamically loads the implementation and runs existing unit tests
5. If tests pass, the implementation is used in the experiment pipeline
6. If the experiment is accepted, the implementation is persisted alongside the experiment log
7. Implementations are never auto-merged into the main codebase — promotion to codebase is a human decision (Tier 3)

#### Validation Requirements

- Authored code must pass all existing unit tests for the algorithm slot it replaces
- Authored code must conform to the Protocol interface (enforced by the runner)
- Authored code may only import from an allowlist: numpy, scipy, torch, librosa, standard library
- Each authored algorithm is tested in isolation before being used in a pipeline run

### Tier 3 — Restricted Operations

Structural changes the agent can propose but not execute:

- Promoting an agent-authored algorithm into the main codebase
- Changing feature extraction parameters that affect stored model compatibility (sample_rate, n_fft)
- Modifying cross-pass data flow (e.g., passing confidence scores from Pass 2 to Pass 3)
- Changing gold set composition or promotion criteria
- Adding entirely new Protocol slots

Tier 3 proposals are logged as recommendations in the experiment log for human review.

### Cross-Pass Interaction Rules

1. Pass 1 config change -> must re-run Passes 2+3 and re-evaluate
2. Pass 2 retrained -> must re-run Pass 3 inference before evaluating Pass 3
3. Pass 2 feature config change -> Pass 3 feature config likely should match (agent must justify mismatch)
4. Pass 3 per-type thresholds can be tuned independently without re-running upstream

---

## 5. Experiment Specification & Logging

### Experiment Spec Schema

```yaml
experiment:
  name: "exp-027-lower-p1-threshold-pcen-norm"
  parent_experiment: "exp-026"
  hypothesis: "Lowering Pass 1 threshold to 0.60 recovers missed regions; switching Pass 2 to PCEN normalization should improve boundary accuracy on low-SNR calls"

  pass_1:
    classifier_model_id: 14
    high_threshold: 0.60
    low_threshold: 0.35
    padding_sec: 1.5
    min_region_duration_sec: 0.0
    authored_algorithms:
      region_detector: null          # null = use built-in; or path to authored implementation
      region_shaper: null

  pass_2:
    retrain: true
    training:
      learning_rate: 5e-4
      weight_decay: 1e-4
      epochs: 40
      early_stopping_patience: 7
      conv_channels: [32, 64, 96, 128]
      gru_hidden: 64
    features:
      normalization: pcen
      n_mels: 64
      fmin: 20.0
      fmax: 4000.0
    decoder:
      algorithm: hysteresis
      high_threshold: 0.5
      low_threshold: 0.3
      min_event_sec: 0.2
      merge_gap_sec: 0.1
    authored_algorithms:
      event_decoder: null            # null = use built-in decoder config above
      feature_normalizer: null       # null = use features.normalization above
      inference_windower: null

  pass_3:
    retrain: true
    training:
      learning_rate: 5e-4
      weight_decay: 1e-4
      epochs: 40
      corrections_only: false
      min_examples_per_type: 5
    features:
      normalization: pcen
    loss:
      function: focal
      params:
        gamma: 2.0
    authored_algorithms:
      feature_normalizer: null
      threshold_optimizer: null

  evaluation:
    gold_set_id: "gold-v3"
    seeds: [42, 137]
    pass_1_recall_floor: 0.85
    max_degradation_pct: 15
```

Key properties:

- **`hypothesis`**: Forces the agent to articulate why it is trying this configuration. Makes the experiment log interpretable and improves Claude's reasoning.
- **`parent_experiment`**: Links experiments into a chain for provenance tracking.
- **`retrain` flags**: Explicitly declares whether the agent is retraining a model or tuning thresholds/decoder on an existing model. Threshold-only experiments are much faster.
- **Strategy switches by name**: Resolved via Tier 2a registries.
- **`authored_algorithms`**: Per-pass references to agent-written implementations. `null` means use the built-in (Tier 2a) selection. A path value points to an authored Python file in the experiment artifacts directory. When present, the authored implementation overrides the corresponding built-in strategy switch.

### Experiment Result Log

```yaml
result:
  experiment: "exp-027"
  timestamp: "20260420T143022Z"
  duration_sec: 2847
  seeds_run: [42, 137]

  per_pass_metrics:
    pass_1:
      recall: 0.91
      false_positive_rate: 0.12
      regions_detected: 284
    pass_2:
      boundary_iou: 0.78
      onset_mae_sec: 0.14
      offset_mae_sec: 0.19
      events_detected: 612
    pass_3:
      event_f1: 0.73
      per_type_f1:
        moan: 0.81
        whup: 0.69
        cry: 0.64
      correction_rate: 0.31

  composite_score: 2.35
  
  vs_parent:
    composite_delta: +0.18
    pass_1_recall_delta: -0.02
    pass_2_boundary_iou_delta: +0.06
    event_f1_delta: +0.04
    correction_rate_delta: -0.05

  verdict: "accept"
  reasoning: "Composite score improved. Pass 1 recall dipped slightly but stays above floor. PCEN normalization improved boundary accuracy. Focal loss gave marginal classification gain."
```

### Run Memory

Each autoresearch run produces a memory file that persists institutional knowledge for subsequent runs. Memory files are stored in `{experiment_artifacts_dir}/autoresearch_memory/` and loaded into the agent's context at session start.

#### Memory File Structure

```yaml
run_memory:
  run_id: "run-2026-04-20-001"
  timestamp: "20260420T180000Z"
  agent_config_summary:
    allowed_tiers: [1, "2a", "2b"]
    pass_focus: null
    time_budget_hours: 4
  gold_set_version: "gold-v3"
  gold_set_size: 67 events

  champion:
    experiment: "exp-029"
    composite_score: 2.52
    key_config:
      pass_1_padding_sec: 2.0
      pass_2_normalization: pcen
      pass_2_decoder: authored/energy_adaptive_decoder
      pass_3_loss: bce

  findings:
    - "PCEN normalization outperforms z-score on boundary accuracy (+8% IoU), especially on low-SNR hydrophone recordings"
    - "Energy-adaptive decoder improves over fixed hysteresis by tracking local signal envelope; biggest gains on gradual onset calls"
    - "Pass 1 padding_sec=2.0 is optimal; values above 2.5 cause region overlap and confuse Pass 2"

  dead_ends:
    - area: "pass_3_loss_function"
      attempts: 3
      detail: "Focal loss (gamma 1.0, 2.0, 3.0) did not improve over BCE at current gold set size (67 events). May be worth revisiting when gold set exceeds 150 events."
    - area: "pass_2_decoder_peak_picking"
      attempts: 2
      detail: "Peak-picking decoder consistently missed low-confidence events that hysteresis catches. Not viable as standalone; could work as ensemble."
    - area: "pass_1_low_threshold_below_0.30"
      attempts: 2
      detail: "Low threshold below 0.30 causes region bleed into noise floor. Composite score degrades due to false positive regions."

  recommendations:
    - type: "tier_3"
      detail: "Consider passing Pass 2 event confidence scores to Pass 3 as an additional input feature — low-confidence boundaries correlate with classification errors"
    - type: "next_run"
      detail: "Pass 2 boundary accuracy is now the bottleneck. Next run should focus pass_focus=[2] with Tier 2b to explore alternative decoder architectures"
```

#### How the Agent Uses Memory

At session start, the agent reads all memory files chronologically. This informs:

- **Dead end avoidance**: If prior runs consistently failed with focal loss at the current gold set size, the agent skips it rather than re-discovering the dead end. The `detail` field provides context for when the dead end might be worth revisiting (e.g., "when gold set exceeds 150 events").
- **Building on prior work**: The champion configuration from the most recent run becomes the baseline. The agent builds on what worked rather than starting from scratch.
- **Strategic focus**: Recommendations from prior runs guide where to spend the budget. If the last run identified Pass 2 as the bottleneck, the current run can start there.
- **Avoiding redundant authoring**: If a prior run already wrote and tested a peak-picking decoder that underperformed, the agent knows not to write a similar one — though it might write a different decoder that addresses the specific failure mode noted in the dead end.

Memory files are append-only across runs — the agent never modifies a prior run's memory. This preserves the historical record and lets the agent reason about trends ("the last 3 runs all identified Pass 2 as the bottleneck — is there a structural issue?").

#### Memory vs. Experiment Log

The experiment log is the detailed record: every experiment spec, every metric, every accept/reject decision. The run memory is the distilled summary: what mattered, what to avoid, what to try next. The agent reads memory for strategic direction and consults the experiment log for detailed evidence when needed.

---

## 6. Agent Architecture

### Orchestrator

A Claude-based orchestrator agent owns the experiment loop:

```
Analyze -> Hypothesize -> Design experiment -> Execute -> Evaluate -> Log -> Decide next step
```

The orchestrator delegates to specialized tools that wrap the existing API and training infrastructure.

### Agent Configuration

```yaml
agent:
  model: "claude-opus-4-6"
  effort: "max"
  time_budget_hours: 4
  max_experiments: 30
  seed_count: 2

  strategy:
    exploration_fraction: 0.3
    retrain_budget_fraction: 0.6
    allow_retraining: true
    allowed_tiers: [1, "2a", "2b"]   # which control plane tiers the agent may use
    pass_focus: null

  guardrails:
    composite_score_floor: null
    max_consecutive_rejects: 5
    tier_2a_switches_per_run: 3
    max_authored_algorithms_per_run: 5
    require_test_pass: true

  memory:
    memory_dir: "autoresearch_memory"  # relative to experiment artifacts root
    read_previous: true                # load memories from all prior runs at session start
    write_summary: true                # persist a memory file at session end
```

- **Model choice**: Opus for deep cross-pass reasoning; Sonnet for faster mechanical iterations.
- **`allow_retraining: false`**: Quick mode for threshold-only exploration (<1 hour).
- **`allowed_tiers`**: Controls which control plane tiers the agent may use in this run. Examples:
  - `[1]` — numeric dials only. Cheapest, fastest. Good for initial threshold sweeps.
  - `[1, "2a"]` — dials + built-in strategy switches. No code authoring.
  - `["2b"]` — algorithm authoring only. Locks all numeric dials and built-in switches to their current values; the agent can only improve things by writing new algorithms. Useful for exploring whether the algorithms themselves are the bottleneck.
  - `[1, "2a", "2b"]` — full access (default). Agent chooses the right tool for each experiment.
- **`pass_focus`**: Lock satisfied passes, focus budget on the bottleneck.
- **`max_consecutive_rejects`**: Forces strategy reassessment after repeated failures.
- **`tier_2a_switches_per_run`**: Max built-in strategy switches to explore in one run.
- **`max_authored_algorithms_per_run`**: Bounds novelty budget — prevents the agent from rewriting everything in one session.
- **`require_test_pass`**: Authored algorithms must pass existing unit tests before use (should always be true; toggle exists for debugging).
- **`memory.read_previous`**: When true, the agent reads memory files from all prior runs before starting. When false, the agent starts with no institutional knowledge (useful for A/B testing the memory system itself).
- **`memory.write_summary`**: When true, the agent writes a memory file at session end summarizing findings and dead ends.

### Tool Inventory

| Tool | Purpose | Maps to |
|------|---------|---------|
| `read_run_memories` | Load memory files from all prior autoresearch runs | Memory directory |
| `read_correction_history` | Fetch corrections grouped by pass and pattern | Queries correction tables |
| `read_experiment_log` | Fetch past experiment specs + results | Experiment log storage |
| `read_gold_set_summary` | Gold set composition, size, type coverage | Gold promotion records |
| `run_pass_1` | Region detection with given config on gold audio | Region detection API |
| `run_pass_2_inference` | Segmentation with existing model + decoder config | Segmentation API |
| `run_pass_2_training` | Train new SegmentationCRNN | Training API |
| `run_pass_3_inference` | Classification with existing model | Classification API |
| `run_pass_3_training` | Train new EventClassifierCNN | Training API |
| `evaluate_against_gold` | Compare pipeline output to gold truth | New evaluation endpoint |
| `write_algorithm` | Write a new algorithm implementation for a Protocol slot | Saves to experiment artifacts, runs unit tests, reports pass/fail |
| `list_algorithm_slots` | List available Protocol slots with their contracts and existing implementations | Reads Protocol definitions + registry |
| `log_experiment` | Persist experiment spec + results + reasoning | New logging endpoint |
| `write_run_memory` | Persist run memory file at session end | Memory directory |

### Safety Guardrails

- The agent cannot modify the gold set
- The agent cannot delete existing models — it only creates new ones
- Tier 1 and Tier 2a dials have bounded ranges defined in the experiment spec schema
- Tier 2b authored algorithms must pass unit tests and conform to Protocol interfaces before use
- Authored algorithm code may only import from an allowlist: numpy, scipy, torch, librosa, standard library
- Authored algorithms are sandboxed in experiment artifact directories — never written to the main codebase
- Tier 3 changes are logged as recommendations, never executed
- Each experiment is logged before execution for full audit trail
- The agent must run at least `seed_count` seeds before accepting a change

### Budget Management

Typical execution times:
- Threshold-only experiments (no retraining): ~5 minutes
- Authored algorithm experiments (no retraining): ~5-10 minutes (write + test + inference + evaluate)
- Single-pass retrain: ~20-40 minutes
- Single-pass retrain with authored algorithm: ~25-50 minutes (write + test + train + evaluate)
- Full pipeline retrain (Pass 2 + Pass 3): ~60-90 minutes

The agent should front-load cheap experiments (threshold sweeps) to narrow the search space, then invest in expensive retrains for the most promising configurations.

---

## 7. Training / Inference Consistency

### Within-Pass Consistency (Must-Align Behaviors)

| Pass | Behavior | Risk if misaligned |
|------|----------|-------------------|
| 2 | Feature extraction (sample_rate, n_fft, hop_length, n_mels, fmin, fmax) | Model sees different spectrograms at inference than training |
| 2 | Normalization method + parameters | Stats from training won't match inference |
| 2 | Decoder algorithm + parameters | Evaluation must use same decoder as deployment |
| 3 | Feature extraction | Same risk as Pass 2 |
| 3 | Normalization method | Same risk as Pass 2 |
| 3 | Per-type thresholds | Must come from the same training run |

**Enforcement**: Feature config and normalization strategy are specified once per experiment and applied to both training and inference. The experiment runner validates consistency before execution. When an authored algorithm replaces a built-in (e.g., a custom normalizer), the same authored implementation must be used in both training and inference paths — the runner enforces this by loading the authored module once and injecting it into both pipelines.

### Cross-Pass Consistency

| Interaction | Risk |
|-------------|------|
| Pass 2 retrained, Pass 3 not | Pass 3 trained on old boundary shapes degrades silently on new boundaries |
| Pass 2 feature config changed, Pass 3 not | Shared spectrogram pipeline diverges |
| Pass 1 thresholds changed | Different regions produce different downstream training data |
| Gold evaluated with corrected boundaries, model trained on predicted boundaries | Training/evaluation gap |

### Cascade Invalidation Rule

```
Pass 1 change -> invalidates Pass 2 eval -> invalidates Pass 3 eval
Pass 2 change -> invalidates Pass 3 eval
Pass 3 change -> self-contained
```

The experiment runner enforces this: if an upstream pass config differs from the parent experiment, all downstream passes must be re-run before evaluation.

---

## 8. Required Components

The following components are needed to enable the autoresearch control plane. Implementation phasing and ordering are deferred to the implementation plan.

### Pass 1 Correction Workflow

- `region_boundary_corrections` table (mirrors `event_boundary_corrections`)
- Read-time overlay function (mirrors `load_corrected_events()`)
- Pass 2 worker reads corrected regions instead of raw `regions.parquet`
- Review UI editing mode on existing region detection timeline viewer
- API endpoints: `GET/POST /region-detection-jobs/{id}/corrections`

### Gold Standard System

- Gold promotion table linking audio segments to verified ground truth at all three pass levels
- Held-out enforcement in training data assembly
- `evaluate_against_gold` endpoint: runs full pipeline on gold audio, compares to truth, returns per-pass metrics + composite score
- Gold set summary endpoint

### Tier 2a Strategy Registries

Registry pattern: string key selects implementation, config dict passes parameters.

- Normalization registry (Pass 2 + 3): zscore, PCEN, per-channel energy norm
- Decoder registry (Pass 2): hysteresis, peak-picking
- Loss function registry (Pass 2 + 3): BCE, focal loss, dice loss
- Architecture variant registries (lower priority)

### Tier 2b Algorithm Plugin Infrastructure

Protocol-based plugin system for agent-authored algorithms.

- Python Protocol definitions for each algorithm slot (see §4 Tier 2b for full list)
- Dynamic module loader: imports authored Python files from experiment artifact directories
- Test harness: runs existing unit tests against authored implementations before use
- Import allowlist enforcement: numpy, scipy, torch, librosa, standard library
- Artifact directory structure: `{experiment_artifacts_dir}/authored_algorithms/{experiment_name}/{slot_name}.py`

### Experiment Infrastructure

- Experiment spec schema (validated Pydantic model) with `authored_algorithms` fields per pass
- Experiment log storage (includes authored algorithm source code for reproducibility)
- Experiment runner with cascade invalidation logic and authored algorithm injection
- Tier enforcement: runner validates that each experiment only uses dials/switches/authoring from `allowed_tiers`
- Agent configuration schema with `allowed_tiers` and memory settings

### Run Memory System

- Memory file schema (YAML with structured sections: champion, findings, dead ends, recommendations)
- Memory directory at `{experiment_artifacts_dir}/autoresearch_memory/`
- Memory reader: loads all prior memory files chronologically at session start
- Memory writer: agent produces a summary memory file at session end
- Memory files are append-only across runs — agents never modify prior run memories

### Agent Tool Layer

- Thin wrappers around existing API endpoints as Claude tool definitions
- Correction history reader
- Experiment log reader/writer
- Pipeline execution tools
- Gold set evaluation and summary tools
- Algorithm authoring tools: `write_algorithm` (write + test), `list_algorithm_slots` (discover available Protocol slots)

### What Does Not Change

- Existing Pass 2/3 model architectures (become the default strategy)
- Existing correction workflows at Pass 2 and Pass 3
- Existing training data assembly (add gold exclusion filter only)
- Parquet artifact format
- Worker infrastructure
- API route structure (new endpoints alongside existing)
