# Call Parsing References

Read these only when the task needs the detail.

## Reference Docs

- `docs/reference/call-parsing-api.md` for endpoint surfaces.
- `docs/reference/behavioral-constraints.md` for pass contracts,
  corrections, feedback training, and effective-event rules.
- `docs/reference/storage-layout.md` for call parsing artifacts.
- `docs/reference/data-model.md` for call parsing tables.
- `docs/reference/signal-processing.md` for segmentation/windowing parameters.

## ADR Headings

- ADR-048: Four-pass call parsing pipeline (Phase 0 scaffold)
- ADR-049: Call parsing Pass 1 - algorithmic defaults and streaming architecture
- ADR-050: Call parsing Pass 2 - bootstrap-era design decisions
- ADR-051: Call parsing Pass 3 - event classifier architecture
- ADR-052: Chunk artifact system applies to hydrophone path only
- ADR-053: Feedback training architecture - correction tables and bootstrap cleanup
- ADR-054: Read-time correction overlay for downstream consumers
- ADR-062: Segmentation-scoped effective event identity
