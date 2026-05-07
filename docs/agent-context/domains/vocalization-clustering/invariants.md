# Vocalization Clustering Invariants

- `"(Negative)"` is reserved and cannot be used as a vocalization type name.
- `"(Negative)"` is mutually exclusive with positive vocalization type labels.
- Vocalization training excludes unlabeled detection windows.
- `"(Negative)"` labels become empty positive-label sets for training.
- Detection rows and vocalization labels are linked by stable `row_id`.
- Deleting a detection row cascade-deletes associated vocalization labels.
- Detection jobs with vocalization labels or training dataset references are
  guarded from deletion.
- Multi-label vocalization classification uses binary relevance.
