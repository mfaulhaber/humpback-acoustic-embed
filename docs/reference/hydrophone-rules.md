# Hydrophone Design Rules

> Read this when working on hydrophone detection, playback, extraction, or timeline features.

## Extraction Path Convention
Hydrophone labeled-sample extraction groups by species/category first, then hydrophone:
- positives: `{positive_output_path}/{humpback|orca}/{hydrophone_id}/YYYY/MM/DD/*.flac`
- negatives: `{negative_output_path}/{ship|background}/{hydrophone_id}/YYYY/MM/DD/*.flac`
- hydrophone extraction should over-fetch a small real-audio guard band and
  hard-trim clips to the expected sample count when archive audio exists;
  never zero-pad short archive clips just to satisfy a window length
- every extracted labeled clip (local and hydrophone) must also write a sibling
  `.png` spectrogram sidecar using the same marker-free base rendering as the UI
  spectrogram popup for that extracted clip window

## Timeline Assembly
Hydrophone detection, playback, and extraction must use the same bounded stream timeline:
- segment ordering must be numeric by segment suffix (never plain lexicographic)
- playlist (`live.m3u8`) duration metadata should be used when available
- sparse local cache segment sets must preserve playlist timeline offsets
  (do not assume the first cached segment starts at folder timestamp)
- folder discovery should start at the requested range and expand backward
  by configurable hour increments (default 4h), up to configurable max
  lookback (default 168h), stopping once overlap at the requested start
  boundary is found
- processing/playback/extraction must stay within `[start_timestamp, end_timestamp]`
- legacy playback compatibility for older jobs may fall back to `job.start_timestamp`
- Orcasound HLS playback/extraction is local-cache-authoritative: resolve from local HLS cache only, with no S3 listing/fetch fallback
- Non-HLS archive providers may use their own direct-fetch playback/extraction path when explicitly configured (for example NOAA GCS `.aif`)
- hydrophone extraction should build/reuse timeline metadata once per extraction run (avoid rebuilding per labeled row)
- hydrophone detection jobs with no overlapping stream audio in the requested range
  must fail with an explicit error message (never silently complete with zero windows)

## Detection TSV Metadata
Hydrophone detection TSV output should carry canonical event metadata:
- canonical `start_sec`/`end_sec` represent snapped clip bounds (window-size multiples)
- include `raw_start_sec`/`raw_end_sec` and `merged_event_count` for audit/debug provenance
- include `detection_filename` for hydrophone rows (`{start_utc}_{end_utc}.flac`, snapped canonical bounds)
- keep `extract_filename` as a legacy alias to the same canonical filename for compatibility; explicit legacy `.wav` values must remain readable
- include `hydrophone_name` for hydrophone rows (short form, e.g., `rpi_north_sjc`)
- persist positive-selection provenance in `positive_selection_*` columns plus
  `positive_extract_filename`; positive extraction seeds from the best 5-second
  scored window and may widen in 5-second chunks when adjacent chunks remain
  above the configured extension threshold
- local detection TSV rows follow the same canonical snapped bounds + raw audit metadata

## Job Lifecycle
Hydrophone detection jobs support the following status transitions:
- `queued` → `running` (worker claims job)
- `running` → `paused` (user pauses via API/UI)
- `paused` → `running` (user resumes via API/UI)
- `running` or `paused` → `canceled` (user cancels; partial results preserved)
- `running` → `complete` (normal completion)
- `running` → `failed` (error during processing)
- TF2 SavedModel hydrophone detection must run in a short-lived subprocess so
  TensorFlow/Metal memory is reclaimed between jobs; the parent worker remains
  responsible for progress, diagnostics, alerts, and pause/resume/cancel state
- Paused jobs remain in the Active Job panel; the worker thread blocks until resumed or canceled
- Paused jobs with partial TSV output remain readable through
  `/classifier/detection-jobs/{id}/content`
- Canceled jobs are fully functional in the Previous Jobs panel (expandable, downloadable, label-editable, extractable)
