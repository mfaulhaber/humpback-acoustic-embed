# Development Backlog

- Agile Modeling Phase 1b: search by uploaded audio clip by embedding the clip on the fly with a selected model, then searching existing embedding sets.
- Agile Modeling Phase 3: connect search-result labeling into classifier training and the retrain loop.
- Agile Modeling Phase 4: prioritize labeling suggestions using model uncertainty signals such as entropy or margin.
- Smoke-test `tf-linux-gpu` on a real Ubuntu/NVIDIA host, including `uv sync --extra tf-linux-gpu`, TensorFlow import, and GPU device visibility.
- Generalize legacy hydrophone API and frontend naming toward archive-source terminology now that NOAA Glacier Bay shares the same backend surfaces.
- Explore GPU-accelerated batch processing for large audio libraries.
- Add WebSocket push for real-time job status updates to replace polling.
- Investigate multi-model ensemble clustering.
- Optimize `/audio/{id}/spectrogram` to avoid materializing all windows when only one index is requested.
- Optimize hydrophone incremental lookback discovery to avoid repeated full S3 folder scans during startup.
- Add an integration and performance harness for hydrophone S3 prefetch so worker defaults can be tuned on real S3-backed runs.
- Investigate a lower-overhead Orcasound decode path, likely chunk-level or persistent-stream decode, and treat it as a signal-processing/runtime change that needs validation plus an ADR.
- Make `hydrophone_id` optional for local-cache detection jobs in the backend API, service layer, and worker.
- Remove vestigial `output_tsv_path` and `output_row_store_path` fields from the detection model, schema, and database via migration.
