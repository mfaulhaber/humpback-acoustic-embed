# Sequence Models References

Read these only when the task needs the detail.

## Reference Docs

- `docs/reference/sequence-models-api.md` for Continuous Embedding, Event
  Encoder, and Event Encoder timeline endpoints.
- `docs/reference/storage-layout.md` for continuous embedding and event encoder
  artifacts.
- `docs/reference/data-model.md` for `continuous_embedding_jobs` and
  `event_encoder_jobs`.
- `docs/reference/behavioral-constraints.md` for Continuous Embedding and Event
  Encoder idempotency and source-mode rules.
- `docs/reference/call-parsing-api.md` when upstream Pass 1/Pass 2 contracts
  matter.

## ADR Headings

- ADR-056: Sequence Models track parallel to Call Parsing pipeline
- ADR-057: CRNN region-based chunk embeddings as second Sequence Models source
- ADR-062: Segmentation-scoped effective event identity
