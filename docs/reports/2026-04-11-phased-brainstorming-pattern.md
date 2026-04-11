# Phased Brainstorming Pattern for Multi-Component Projects

**Date**: 2026-04-11
**Context**: Discovered while planning the humpback call parsing pipeline (a 4-pass project: detect → segment → classify → export). A single brainstorming session for the full project would have produced an unwieldy spec conflating architecture with internal design details, required re-litigating the architecture for each subsystem, and burned context window re-loading the same concepts across many sessions. The pattern below let us lock the architecture once and scope each subsequent brainstorm to internal design only.

## The Pattern

Break a large multi-subsystem project into:

1. **Phase 0** — a dedicated first phase that ships the architecture contract plus a working-but-empty scaffold. No subsystem logic.
2. **Subsequent phases** — one per subsystem, each its own feature branch with its own brainstorm/spec/plan/implement cycle scoped only to internal design.

Plus four supporting artifacts produced during Phase 0 planning:
- A Phase 0 design spec that is the binding contract
- A Phase 0 executable plan
- **Skeletal plans for every subsequent phase**, committed alongside Phase 0's own plan
- A project memory file that auto-surfaces the contract in every future session

## Phase 0 Scope

**Phase 0 ships:**
- Data model / SQL tables / Alembic migration
- API surface (routes exist but return 501 for unimplemented pieces)
- Worker shells (claim → fail with `NotImplementedError`, wired into priority order)
- Repo layout (new packages created, module files reserved, imports wired up)
- Shared infrastructure (e.g., training harness, storage helpers, dataclasses, parquet schemas)
- Dependency additions (e.g., new ML frameworks bundled into existing extras)
- Behavior-preserving refactors that extract helpers the subsequent phases will consume
- Documentation updates (CLAUDE.md, DECISIONS.md ADR)

**Phase 0 does NOT ship:**
- Any subsystem logic
- Any model training or inference
- Any frontend work
- Any design decisions that depend on data or experimentation

**End state:** the project compiles, tests pass, migrations apply cleanly, and the scaffold can be exercised (e.g., a parent entity can be created via API) even though nothing beyond the contract works.

## Skeletal Plans for Subsequent Phases

During Phase 0 planning, write one skeletal plan per subsequent phase and commit them alongside the Phase 0 plan. Each skeletal plan contains four sections:

1. **Goal** — one sentence.
2. **Inherited from Phase 0 (do NOT re-derive)** — the tables, helpers, modules, dataclasses, API stubs, and worker shells this phase can assume exist. This is the load-bearing section: it prevents a future session from re-brainstorming the architecture.
3. **Brainstorm checklist — TBDs** — the exhaustive list of design decisions this phase's brainstorm must resolve. Grouped by topic (model architecture, input features, training data, evaluation, operational). Each item is a checkbox.
4. **Skeletal tasks** — the structural tasks that are already known (new migration, new module file, new endpoint) with TBD-marked acceptance criteria for the design-heavy parts.

Plus the standard verification section at the end.

The skeletal plan is NOT executable. It's a handoff artifact that lets a future session pick up the phase with minimal context-loading: read the Phase 0 spec once, read this plan once, brainstorm only the TBDs, then elaborate the tasks.

## Project Memory File

Write a `project_<name>.md` memory file in `~/.claude/projects/<project>/memory/` pointing at all the phase artifacts, stating the architecture contract, and including the **rule**: *"When resuming work on any phase, do NOT re-derive the table structure, worker priority, parquet layout, or API surface — those are committed. Brainstorm only the TBDs listed in that phase's skeletal plan."*

Add a one-line pointer in `MEMORY.md` so it auto-loads in every future session.

## Cross-Phase Feedback Loop

**This is the second half of the pattern and the piece that makes phased planning self-correcting over time.** Phase 0 cannot anticipate everything. Phase N's implementation will surface discoveries that should inform Phase N+1:

- The helper extracted in Phase 0 ends up with a different signature than the skeletal plan assumed
- A table column decided in Phase 0 turns out to need a different type
- Phase N+1 gains a new TBD because Phase N uncovered an assumption that was wrong
- Phase N resolves a TBD that Phase N+1 was worried about (e.g., Phase 1 settles on a padding convention that Phase 2 can inherit)

### Mechanism — Downstream Review at Session-End

At `session-end` of each phase, **before merging the feature branch to main**:

1. **Read the next phase's skeletal plan.** (Or every downstream phase, if the changes are architectural.)
2. **Compare actual against planned.** Where does the just-completed phase's reality diverge from the downstream plan's "Inherited from Phase N" section? Common divergences: helper signatures, new config columns, new fields on dataclasses, renamed modules, changed storage paths.
3. **Update the downstream plan's "Inherited" section** to reflect reality.
4. **Add a "Post-Phase N findings" subsection** to the downstream plan's brainstorm checklist listing:
   - Newly-discovered TBDs the brainstorm should now cover
   - Previously-listed TBDs that Phase N has implicitly resolved
   - Lessons learned that should shape downstream design (e.g., "Phase 1 found that the score trace compresses 20× with minimal quality loss — Pass 2 can assume a decimated input")
5. **Update the project memory file** if any architecture-level facts changed (new ADR, new worker, new table, new naming convention).
6. **Commit the plan updates on the current phase's branch** so they merge to main with the code — phase branches never merge without their downstream hand-off.

### Example (hypothetical, for illustration)

*Phase 0 plans `compute_hysteresis_events(audio, sr, perch_model, classifier, config) -> tuple[list[WindowScore], list[HysteresisEvent]]`.*

During Phase 0 implementation the refactor reveals the helper needs a fourth return value (a `WindowSelectionMode` diagnostic flag that the existing detector's snap-merge path depends on). The downstream review:

- Updates the Pass 1 skeletal plan's "Inherited from Phase 0" section with the real 4-tuple signature
- Adds to the Pass 1 brainstorm checklist: *"Decide whether region detection consumes the diagnostic flag or discards it"*
- No update to Passes 2–4 (the helper is Pass 1's consumer only)
- Commits the Pass 1 plan update on the Phase 0 branch

*Another example, hypothetical.*

*During Pass 1 implementation, we find that regions with many overlapping padded events produce very dense trace parquet files — a full run on a 4-hour hydrophone archive hits ~200 MB of trace data. The downstream review:*

- Adds to the Pass 2 brainstorm checklist: *"Pass 2 input is much denser than expected (~200MB/4hr). Decide whether to decimate at read time or at write time."*
- Updates the project memory file with the observation
- Potentially opens an ADR discussion about whether to add trace decimation to the Phase 0 writer retroactively (a backport to Phase 0's behavior)

## When This Pattern Applies

- Multi-subsystem projects where subsystems share a common data contract but have internally independent design decisions
- Projects where architecture can be decided upfront but internal design depends on data, labels, or experimentation
- Projects that span multiple sessions/days/weeks and where token efficiency matters
- Projects where the subsystem boundaries are clean (each has a clear input/output contract)

## When It Doesn't

- Small single-component tasks (one or two files, one session) — the overhead of Phase 0 exceeds the benefit
- Research experiments where the architecture IS the design (e.g., comparing model variants that can't share a common contract)
- Projects where subsystems are tightly coupled and can't be built independently

## Ingredients Checklist (reusable)

For a future project considering this pattern:

- [ ] Decide the Phase 0 scope: contract + scaffold, not logic
- [ ] Brainstorm the Phase 0 architecture contract (in conversation, no spec on disk yet)
- [ ] Write Phase 0 spec under `docs/specs/YYYY-MM-DD-<project>-phase0-design.md` covering tables, workers, APIs, repo layout, infrastructure, dependencies
- [ ] Write Phase 0 executable plan under `docs/plans/YYYY-MM-DD-<project>-phase0.md`
- [ ] Write skeletal plans for Phases 1..N under `docs/plans/YYYY-MM-DD-<project>-phaseN-<name>.md`, each with "Inherited from Phase 0" + "Brainstorm checklist" + skeletal tasks
- [ ] Write the project memory file under `~/.claude/projects/<project>/memory/project_<name>.md` pointing at all artifacts and stating the re-derivation rule
- [ ] Add a one-line pointer in `MEMORY.md`
- [ ] Commit all of the above on the Phase 0 feature branch as a single planning commit
- [ ] At `session-end` of each subsequent phase, run **downstream review**: update every downstream skeletal plan with findings, update the project memory file, commit the updates on the phase branch before merging

## Worked Example

The humpback call parsing pipeline was planned using this pattern on 2026-04-11.

**Project:** 4-pass humpback vocalization pipeline (detect → segment → classify → export). Large enough that a single brainstorm would have been unwieldy.

**Phase 0 artifacts (committed on `feature/call-parsing-pipeline-phase0`):**
- `docs/specs/2026-04-11-call-parsing-pipeline-phase0-design.md` (architecture contract)
- `docs/plans/2026-04-11-call-parsing-pipeline-phase0.md` (8 executable tasks)
- `docs/plans/2026-04-11-call-parsing-pass1-region-detector.md` (skeletal)
- `docs/plans/2026-04-11-call-parsing-pass2-segmentation.md` (skeletal, heaviest — introduces PyTorch)
- `docs/plans/2026-04-11-call-parsing-pass3-event-classifier.md` (skeletal)
- `docs/plans/2026-04-11-call-parsing-pass4-sequence-export.md` (skeletal, lightest)
- `~/.claude/projects/-Users-michael-development-humpback-acoustic-embed/memory/project_call_parsing_pipeline.md`
- ADR-048 (added during Phase 0 Task 8, not yet landed)

**Commits:**
- `cd27f16` — Phase 0 spec + executable plan
- `821cf33` — Four Pass 1–4 skeletal plans

**Status at time of writing:** Phase 0 implementation not yet started. Passes 1–4 each await their own brainstorm/spec/plan/implement cycle, which will be much smaller than Phase 0's brainstorm because the architecture contract is already locked in.

## Related Project Conventions

This pattern layers on top of the existing session workflow (CLAUDE.md §10.1):

```
session-begin → brainstorm → session-plan → session-implement → [session-debug]* → session-review → session-end
```

- Phase 0's planning cycle uses the standard flow plus extras: skeletal plans and memory file are written during the session-plan step rather than during brainstorming, since brainstorming in this project does not write to disk
- Each subsequent phase runs the full standard flow, with the phase's own brainstorm producing a per-phase spec
- The **downstream review** happens during each phase's `session-end`, before the feature branch merges

## Open Questions

- **Automation:** Could the downstream review be partly automated by a skill that reads the current phase's diff and the downstream plan's "Inherited" section, then flags mismatches? Worth exploring if we use this pattern on another large project.
- **Backporting:** When Phase N uncovers something that should change Phase 0's already-shipped contract, what's the right process? Amend Phase 0's spec and open a follow-up migration, or add a new ADR documenting the delta? Needs a worked example to settle.
- **Phase-0 sizing:** How big can a Phase 0 get before it itself needs decomposition? The humpback call parsing Phase 0 is 8 tasks and feels right-sized. A 20-task Phase 0 would probably need its own sub-phases.
