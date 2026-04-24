# Call Parsing Correction Recovery Design

## Context

PR #139 unified vocalization corrections under `vocalization_corrections`, but
existing legacy `event_type_corrections` data was not migrated into the live
database. Recovery is possible from the April 19 backup DB plus the persisted
Pass 2/Pass 3 artifacts under `data/call_parsing/`.

While validating the recovered rows end-to-end, one UI-facing gap remains:
saved vocalization labels for boundary-added events do not survive reload in the
Classify review workspace because the frontend only remaps saved corrections
onto typed events that already exist in `typed_events.parquet`.

## Goals

- Provide repeatable local recovery tooling for legacy correction data.
- Support safe dry-run previews before writing to the live DB.
- Verify the target DB after recovery so the operator gets an explicit success
  or failure result.
- Restore UI-facing saved labels for boundary-added events in Classify review.

## Non-Goals

- Auto-running recovery during migrations.
- Attempting to recover orphaned legacy rows that no longer have enough DB or
  artifact linkage to reconstruct exact modern rows.
- Broad UI redesign of the Classify review flow.

## Design

### Recovery Scripts

- Keep the recovery tools in `scripts/` so they can be run directly against
  local SQLite backups.
- Provide one script for `vocalization_corrections` recovery and include the
  companion boundary-correction recovery tool in the same feature branch.
- Default to dry-run behavior; writes only happen with `--apply`.
- Build a deterministic recovery plan first, then either:
  - preview target inserts/updates/unchanged rows, or
  - apply the plan and verify that the expected rows are present afterward.

### Vocalization Recovery Strategy

- Read legacy `event_type_corrections` rows from the backup DB.
- Recover event bounds in priority order:
  1. `typed_events.parquet`
  2. matching legacy `event_boundary_corrections`
  3. `events.parquet`
- Convert legacy positives to unified `add` rows.
- Convert legacy negatives (`type_name IS NULL`) to unified `remove` rows only
  when the legacy predicted type can be reconstructed from typed-event output.
- Report unrecoverable rows explicitly instead of guessing.

### Classify Review Fix

- Extract the saved-correction merge logic into a small helper.
- Include saved boundary-added events when building the saved event-bounds map.
- Use the same synthetic event-id shape already used by the review workspace for
  saved adds so reloaded labels attach to the same UI events.

## Risks

- Legacy negative corrections can remain unrecoverable when the old predicted
  type can no longer be inferred from surviving artifacts.
- Boundary-added events rely on overlap matching by saved bounds, so extremely
  close neighboring added events could still require careful regression testing.

## Validation

- Unit-test the recovery planning/apply/verify path.
- Unit-test the Classify review merge helper for saved boundary-added events.
- Run targeted dry-run/apply checks against the known backup/live DB pair.
- Run frontend TypeScript verification after the UI fix.
