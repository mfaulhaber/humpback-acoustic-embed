# Development Plans

---

## Active

No active plans.

---

### [COMPLETED] Overlapping Window Inference + Hysteresis Event Detection

Full plan: `~/.claude/plans/shimmying-foraging-lynx.md`

Configurable `hop_seconds` for overlapping windows, hysteresis dual-threshold event merging, and per-event diagnostics. Implemented across 6 phases (core logic, DB/model, API, frontend, tests, docs). See ADR-005.

---

## Backlog

- Explore GPU-accelerated batch processing for large audio libraries
- Add WebSocket push for real-time job status updates (replace polling)
- Investigate multi-model ensemble clustering

---

## Completed

- Overlap-back windowing (ADR-001)
- In-place folder import (ADR-002)
- Balanced class weights for detection (ADR-003)
- Negative embedding sets for training (ADR-004)
- Multi-agent memory framework migration
- Overlapping window inference + hysteresis event detection (ADR-005)
- Incremental detection rendering with per-file progress (ADR-006)
