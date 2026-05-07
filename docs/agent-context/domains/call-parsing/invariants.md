# Call Parsing Invariants

- Pass 1 source identity is exactly one local `audio_file_id` or one hydrophone
  triple: `hydrophone_id`, `start_timestamp`, `end_timestamp`.
- Pass 1 hydrophone chunks align to whole window-size boundaries so streaming
  scoring matches a single-buffer run.
- Pass 1 hydrophone jobs preserve chunk artifacts for resume; file-based jobs do
  not use chunk artifacts.
- Pass 2 and Pass 3 resolve source identity through upstream jobs rather than
  duplicating source columns.
- Pass 2 writes immutable raw `events.parquet`.
- Pass 3 writes immutable `typed_events.parquet`.
- Human corrections live in SQL overlay tables and do not mutate parquet
  inference artifacts.
- New downstream consumers must choose raw events or effective events
  explicitly.
- Boundary corrections must not create overlapping effective events within the
  same segmentation job and region.
- Event type corrections are single-label at the event level.
- Feedback training is hydrophone-only and resolves audio through the job chain.
