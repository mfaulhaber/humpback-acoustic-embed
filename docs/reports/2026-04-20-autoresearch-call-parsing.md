# Teaching an AI to Optimize Whale Call Parsing

How we're building a self-improving pipeline for humpback whale vocalization analysis.

## The Problem

Identifying and classifying humpback whale vocalizations in underwater audio is a multi-step process. Our system runs audio through three stages: detect regions of whale activity, segment individual call events within those regions, and classify each event by vocalization type (moans, whups, cries, etc.).

Each stage has a trained model with dozens of configurable parameters. Getting these right requires expertise, intuition, and a lot of trial and error. Change a threshold in stage 1 and it ripples through everything downstream. Retrain the segmentation model and suddenly the classifier sees different inputs. The interactions between stages make manual optimization slow and unpredictable.

Meanwhile, every time a researcher reviews the system's output, they make corrections — adjusting event boundaries, fixing misclassified call types, marking missed detections. These corrections contain exactly the signal needed to improve the system, but turning that signal into better models requires someone to analyze patterns, form hypotheses, design experiments, retrain models, and evaluate results. That's hours of skilled work per improvement cycle.

## The Idea: Let Claude Run the Experiments

What if we gave a Claude agent access to all the knobs in the pipeline and let it run optimization experiments for hours at a time?

The agent would:

1. **Read the full history of human corrections** to understand where the pipeline is failing
2. **Form a hypothesis** about what to change ("researchers are correcting a lot of boundaries on quiet calls — maybe the segmentation model needs PCEN normalization instead of z-score")
3. **Design and run an experiment** — adjust parameters, retrain models if needed, run the pipeline
4. **Evaluate against a gold standard** of human-verified examples
5. **Log what it tried, what happened, and why** — then decide what to try next

This isn't blind hyperparameter search. Claude can reason about *why* corrections are happening, spot patterns across pipeline stages, and make targeted changes. An optimizer like Optuna can sweep numbers efficiently, but it can't look at correction data and realize "the issue isn't the threshold — it's that the normalization method doesn't handle low-SNR recordings well."

## The Three-Stage Pipeline

```
Audio ──> [Pass 1: Detection] ──> [Pass 2: Segmentation] ──> [Pass 3: Classification]
              |                        |                          |
         Regions of              Individual call              Vocalization type
         whale activity          event boundaries             labels per event
```

**Pass 1 (Region Detection)** runs a pre-trained Perch model over the audio and uses hysteresis thresholding to find regions where whales are vocalizing. No trainable model here — just thresholds and padding parameters. But these settings determine everything downstream. Miss a region and those calls are invisible to later stages. Detect too aggressively and you flood the pipeline with noise.

**Pass 2 (Event Segmentation)** takes each detected region and finds the individual call events within it using a convolutional recurrent neural network (CRNN). The model predicts frame-by-frame whether it's inside or outside an event, then a decoder converts those predictions into discrete event boundaries. Both the model and the decoder have tunable parameters.

**Pass 3 (Event Classification)** crops the audio at each event's boundaries, computes a spectrogram, and runs it through a CNN that predicts vocalization types. This is a multi-label classifier — a single event can contain multiple vocalization types. Per-type confidence thresholds determine the final labels.

### Error Propagation

Errors cascade forward through the pipeline. A conservative detection threshold in Pass 1 means Pass 2 never sees certain regions. An aggressive segmentation decoder that merges events means Pass 3 classifies combined calls it wasn't trained on. This cascade is why optimizing one stage in isolation often fails — what's locally optimal for detection may produce inputs that are hostile to classification.

The autoresearch agent must reason about these cross-stage interactions, not just tune each stage independently.

## What the Agent Controls

We organize the agent's controls into three tiers:

### Numeric Dials

These are scalar parameters the agent can sweep — thresholds, learning rates, architectural sizes, decoder sensitivity. About 15 parameters across the three stages. These are the cheapest experiments: change a number, re-run inference, evaluate. Five minutes per experiment.

### Strategy Switches

These are choices between alternative algorithms. For example, the segmentation model currently normalizes spectrograms using z-score statistics. An alternative is PCEN (Per-Channel Energy Normalization), which is widely used in bioacoustics because it handles varying noise floors better than z-score. The agent can switch between these approaches via a registry system — no code changes needed, just a configuration choice.

Other strategy switches include the decoder algorithm (hysteresis vs. peak-picking), the loss function (standard binary cross-entropy vs. focal loss for handling class imbalance), and model architecture variants. These experiments are more expensive because they usually require retraining, but they can unlock step-change improvements that no amount of numeric tuning would find.

### Agent-Authored Algorithms

This is where the system gets interesting. Beyond switching between pre-built options, the agent can write entirely new algorithm implementations.

Every pluggable algorithm in the pipeline has a defined contract — a Python Protocol that specifies exactly what goes in and what comes out. For example, the region detector contract is: "given an array of per-window confidence scores and a config dict, return a list of regions with start/end times." The event decoder contract is: "given an array of framewise probabilities, return a list of discrete events with boundaries."

The agent can write a Python file that implements any of these contracts. Before the implementation is used in an experiment, it must pass the existing unit test suite for the algorithm slot it replaces. This is the safety mechanism — the tests define the behavioral contract, and any replacement must satisfy it.

There are 11 algorithm slots across the pipeline that the agent can target:

| Slot | Pass | What it does | Current approach |
|------|------|-------------|-----------------|
| Region detector | 1 | Convert scores to regions | Hysteresis state machine |
| Region shaper | 1 | Pad, merge, filter regions | Sorted merge with overlap |
| Event decoder | 2 | Frame probabilities to events | Framewise hysteresis |
| Feature normalizer | 2+3 | Normalize spectrograms | Per-region z-score |
| Inference windower | 2 | Handle long audio | 50% overlap averaging |
| Training loss | 2 | Segmentation loss function | Masked BCE |
| Train/val splitter | 2+3 | Partition training data | Audio-source-disjoint |
| Event matcher | eval | Match predictions to truth | Greedy IoU |
| Threshold optimizer | 3 | Find per-type thresholds | Grid search |
| Class weighter | 3 | Compute loss weights | Positive/negative ratio |
| Post-processor | 3 | Logits to typed events | Threshold + fallback |

Each of these is under 100 lines of code, a pure function or lightweight module, and has existing tests. The agent isn't rewriting the entire pipeline — it's swapping out small, well-defined components.

For example, the current region detector uses a simple hysteresis state machine: open a region when confidence exceeds a high threshold, keep it open while above a low threshold, close it when confidence drops below. The agent might write an adaptive version that adjusts thresholds based on local noise floor estimates, or a peak-picking approach that finds confidence maxima and grows regions outward until the signal drops. As long as it takes scores in and produces regions out, and passes the tests, it's valid.

Authored algorithms live in experiment artifact directories, never in the main codebase. If an agent-written algorithm produces consistently better results across multiple sessions, a human can review the code and decide whether to promote it into the codebase permanently.

### Restricted Operations

Some changes are too structural for the agent to make autonomously — promoting authored algorithms into the main codebase, changing fundamental signal processing parameters like sample rate, modifying how data flows between stages, or adding entirely new algorithm slots. The agent can propose these as recommendations in its log, but a human decides whether to implement them. This keeps the system safe while still capturing the agent's insights.

## Focusing the Search

Not every autoresearch run needs to explore everything. The agent configuration includes an `allowed_tiers` setting that controls which parts of the control plane the agent can touch:

- **Tier 1 only** — the agent can only adjust numeric parameters: thresholds, learning rates, decoder sensitivity. This is the cheapest mode. Experiments take about 5 minutes each. Good for answering "are the current algorithms fine, just misconfigured?"

- **Tier 2a only** — the agent can switch between pre-built strategy alternatives (PCEN vs. z-score, focal loss vs. BCE) but can't tune numeric dials or write new code. Good for answering "is one of our built-in alternatives clearly better?"

- **Tier 2b only** — the most interesting mode. All numeric dials and built-in switches are locked to their current values. The only way the agent can improve things is by writing new algorithm implementations. This directly answers "are the algorithms themselves the bottleneck, or is it just configuration?" If a Tier 2b-only run produces significant gains, it proves the algorithms needed rethinking. If it doesn't, the team knows to focus on data quality or configuration instead.

- **All tiers** — full access. The agent chooses the right tool for each experiment. This is the default for long runs where the agent has hours to explore.

This tiered control also enables a useful debugging pattern: if a full-access run produces improvements, you can re-run with each tier isolated to attribute the gains. "Was it the threshold change or the new decoder that mattered?"

## The Gold Standard: How We Know It's Working

The agent needs a reliable way to measure whether a change is actually better. We can't use the same corrections for both training and evaluation — that would be testing on training data.

The solution is a **gold standard evaluation set**: a collection of audio segments where a human has verified the correct answer at every pipeline level — correct region boundaries, correct event boundaries, and correct vocalization type labels.

Gold items aren't created through a separate annotation effort. They're promoted from the normal review workflow. When a researcher reviews a segment and corrects it through all three stages, that segment becomes a gold candidate. Once promoted, it's permanently held out from model training and used only for evaluation.

This means the gold set grows organically as the system is used. Early autoresearch runs might evaluate against 50 events; after months of use, the gold set could contain thousands. The agent's evaluation confidence improves over time without any dedicated annotation effort.

## A Concrete Example

Here's what an autoresearch session might look like:

**Hour 1 — Analysis and cheap experiments.** The agent reads the correction history and notices that 40% of boundary corrections are on regions shorter than 3 seconds, where researchers are expanding the boundaries. It hypothesizes that `padding_sec=1.0` is too tight. It runs five threshold-only experiments with padding values from 1.0 to 3.0, evaluating each against the gold set. Padding of 2.0 reduces boundary corrections by 15% without hurting detection precision. Accepted.

**Hour 2 — Strategy exploration.** With the new padding, boundary corrections are still elevated on quiet recordings. The agent tries switching Pass 2 normalization from z-score to PCEN, which handles low signal-to-noise ratios better. This requires retraining the segmentation model. After 25 minutes of training, boundary IoU on the gold set improves from 0.72 to 0.78. The agent re-runs Pass 3 classification on the new boundaries — type label F1 also improves from 0.69 to 0.73 because the classifier now sees more accurately cropped spectrograms. Accepted.

**Hour 3 — Algorithm authoring.** Threshold tuning on the Pass 2 decoder is hitting diminishing returns. The agent decides to write a new decoder algorithm. The current approach uses fixed-threshold hysteresis — but the agent hypothesizes that an energy-adaptive decoder, which adjusts its sensitivity based on the local signal envelope, would handle quiet passages better. It writes a 60-line Python implementation of the `EventDecoder` Protocol, runs the existing decoder test suite (12 tests), all pass. It plugs the new decoder into the pipeline — boundary IoU improves from 0.78 to 0.82 on the gold set, with the biggest gains on low-SNR regions. Accepted.

**Hour 4 — Consolidation and targeted tuning.** The agent tries focal loss for Pass 3, but it doesn't improve over standard BCE. Rejected. It fine-tunes Pass 3 per-type thresholds with the new decoder's boundaries — small gains on rare vocalization types. Accepted. Final pass: re-run the full pipeline with best configuration across both seeds, verify reproducibility, log the final composite score. Summary: "4 accepted changes, composite score improved from 1.84 to 2.52 (+37%). Primary gains from increased padding, PCEN normalization, and energy-adaptive decoder. Authored decoder implementation saved to experiment artifacts for human review."

## The Compound Metric

The optimization target isn't just accuracy — it's accuracy relative to human effort:

```
composite_score = event_F1 / correction_rate
```

A system that classifies perfectly but requires a human to review every segment scores poorly. A system that's somewhat less accurate but rarely needs correction scores well. This metric directly captures the goal: maximize the useful work the pipeline does autonomously.

The agent also tracks per-stage diagnostics (detection recall, boundary IoU, per-type F1) so it can pinpoint where the bottleneck is. If the composite score is low because of boundary errors, it focuses on Pass 2. If boundaries are fine but type labels are wrong, it focuses on Pass 3.

## Why Claude, Not Optuna

Traditional hyperparameter optimizers like Optuna are efficient at searching numeric parameter spaces. They can run hundreds of trials with Bayesian optimization, tree-structured Parzen estimators, or evolutionary strategies. For Tier 1 numeric dials, they would likely find good values faster than Claude.

But the interesting improvements in a cascaded pipeline come from strategy switches, algorithm invention, and understanding *why* things fail. Claude can:

- Read correction patterns and form hypotheses ("boundary errors concentrate on low-frequency moans — the current fmin of 20 Hz might be cutting off relevant signal")
- **Write new algorithm implementations** that go beyond the pre-built options ("the fixed-threshold decoder can't handle varying noise floors — let me write an adaptive version that estimates local energy")
- Reason about cross-stage interactions ("if I change Pass 2 normalization, I should also change Pass 3 normalization for consistency")
- Make strategic budget decisions ("I've spent 2 hours on Pass 2, but the gold set metrics show Pass 1 recall is now the bottleneck — time to shift focus")
- Know when to stop ("three consecutive rejected experiments on focal loss — this isn't the path, try something else")
- Propose Tier 3 recommendations for changes it can't make itself

The algorithm authoring capability is the clearest differentiator from traditional AutoML. Optuna can find that `threshold=0.63` is better than `threshold=0.50`, but it can't look at the decoder's failure cases and write an entirely different decoder that addresses the underlying problem. Claude can read the hysteresis state machine, understand why it fails on gradual onsets, and write a peak-picking variant that handles those cases — all within the safety boundary of the Protocol contract and existing test suite.

The experiment log — with its hypothesis, reasoning, accept/reject decisions, and authored source code — becomes institutional knowledge that persists across autoresearch sessions. A future session can read that "energy-adaptive decoder outperformed hysteresis on 2026-04-20" and build on that algorithm rather than re-discovering it.

## What We Need to Build

The autoresearch system builds on the existing call parsing pipeline. The core new components are:

- **Pass 1 correction workflow** — currently the only pass without human review capability. Needed for gold standard completeness.
- **Gold standard system** — promotion workflow, held-out enforcement, evaluation endpoint.
- **Strategy registries** — pluggable implementations for normalization, decoders, loss functions. Configuration-driven, no code changes per experiment.
- **Algorithm plugin infrastructure** — Python Protocol definitions for each algorithm slot, dynamic module loader, test harness for validating authored code, import allowlist enforcement. This is what enables the agent to write new algorithms safely.
- **Experiment infrastructure** — spec schema, runner with cascade invalidation, structured logging (including authored source code for reproducibility).
- **Agent tool layer** — Claude-callable wrappers around the pipeline APIs, plus algorithm authoring tools.

Each component is independently useful. The gold standard system improves evaluation even without autoresearch. Strategy registries make manual experimentation easier. The Protocol definitions formalize algorithm boundaries that benefit the codebase regardless of autoresearch. The experiment log captures knowledge whether a human or an agent runs the experiments.

## The Long View

The first autoresearch runs will be modest — a small gold set, a handful of strategy options, conservative guardrails. But the system is designed to compound. Every human review session grows the gold set. Every autoresearch run logs what worked and what didn't. New strategy options can be added to registries without changing the agent. The agent configuration itself is tunable — model choice, time budget, exploration vs. exploitation balance.

Over time, the human role shifts from manually tuning the pipeline to reviewing the agent's work, promoting gold examples, and reviewing agent-authored algorithms for potential promotion into the codebase. The corrections that researchers already make as part of their normal workflow become the fuel that drives the system forward.

## Learning Across Runs

A single autoresearch session produces useful improvements. But the real power emerges across multiple sessions. Each run writes a memory file — a structured summary of what it found, what failed, and what to try next.

When a new session starts, it reads the memories from all prior runs. This creates a form of institutional knowledge:

**Dead end avoidance.** If three prior runs all tried focal loss for Pass 3 and it never improved over standard BCE, the next run knows to skip it. But the memory includes context — "did not improve at gold set size of 67 events" — so when the gold set grows to 200 events, a future run can recognize that the conditions have changed and the dead end may be worth revisiting.

**Building on what worked.** Each memory file records the best configuration found in that run. The next session starts from the best known configuration rather than the original defaults. Over multiple runs, the pipeline ratchets toward better performance.

**Strategic continuity.** A run might end with the recommendation: "Pass 2 boundary accuracy is now the bottleneck; next run should focus on Tier 2b decoder authoring." The next session reads this and can act on it immediately rather than spending an hour of its budget rediscovering where the bottleneck is.

**Trend detection.** When an agent reads that the last five runs all identified the same bottleneck despite different approaches, it can flag this as a potential Tier 3 issue — something that might require structural changes beyond what the control plane can express.

The memory system is deliberately simple: YAML files, append-only, one per run. Agents never modify prior memories. This preserves the historical record and avoids the risk of an agent editing away evidence of past failures.

The most striking aspect of this architecture is what the agent produces beyond just better numbers. Each autoresearch session generates a log of hypotheses, experiments, and authored algorithms — a record of what was tried, what worked, what didn't, and why. This is institutional knowledge that a traditional hyperparameter sweep never creates. When a researcher reads that "an energy-adaptive decoder improved boundary accuracy by 5% on low-SNR recordings because the fixed hysteresis threshold can't track gradual signal onsets," they learn something about the acoustic properties of their data that informs their own research — not just their pipeline configuration.
