# Sequence Models Invariants

- Continuous Embedding and Event Encoder are the active Sequence Models runtime
  surfaces.
- HMM, Masked Transformer, and motif-extraction runtime surfaces are retired.
- One canonical Continuous Embedding row/output exists per `encoding_signature`.
- Re-submitting an in-flight or complete signature reuses the existing row.
- Re-submitting a failed or canceled signature resets that row to `queued`.
- SurfPerch Continuous Embedding can read raw or effective event source modes.
- Effective event mode must include correction revision identity in the
  `encoding_signature`.
- CRNN region Continuous Embedding requires completed Pass 1 and matching Pass 2
  context plus a segmentation CRNN model.
- CRNN region source rejects effective event mode because the artifact is
  region-scoped rather than event-padded.
- One canonical Event Encoder row/output exists per `tokenization_signature`.
- Event Encoder reuses in-flight or complete signatures and resets failed or
  canceled signatures to `queued`.
- Event Encoder requires a completed Pass 2 segmentation job and a completed
  matching `region_crnn` Continuous Embedding job.
- Event Encoder raw mode reads immutable Pass 2 `events.parquet`; effective
  mode reads `load_effective_events()` and includes correction revision identity
  in the `tokenization_signature`.
- Event Encoder recomputes event/chunk overlap from timestamps and never treats
  CRNN `nearest_event_id` as authoritative.
- Completed Event Encoder timeline views are artifact-authoritative: they read
  the job's token/vector parquet artifacts and do not reload current Pass 2
  raw or effective events.
- Event Encoder token ids and `Txx` labels are job-local and k-local, not a
  stable global vocabulary.
- The active v3 Event Encoder non-CRNN descriptor order is `duration`,
  `log_energy`, `peak_frequency`, `spectral_centroid`, `bandwidth`,
  `spectral_entropy`, `ridge_log_frequency_slope`, `gap_to_previous`,
  `median_f0`, `f0_range`, `voicing_fraction`, `inflection_count`,
  `pulse_rate`, `pulse_rate_slope`, `ridge_median_frequency`,
  `ridge_low_frequency`, `ridge_high_frequency`, `ridge_frequency_span`,
  `ridge_coverage`, `ridge_energy_ratio`, `band_limited_peak_frequency`,
  `high_band_energy_ratio`.
- Existing v2 Event Encoder artifacts with the 14-entry descriptor order remain
  readable because timeline endpoints derive descriptor fields from each
  artifact manifest.
- Event Encoder descriptor vectors are robust-z normalized, clipped to
  `descriptor_clip_value` (default 3.0, null disables clipping), then multiplied
  by `descriptor_weight`.
- Event Encoder ridge descriptors persist scalar summaries only, including
  trimmed low/high ridge frequency bounds for piano roll display. Full STFT
  matrices and frame-level ridge contours are not stored in Continuous
  Embedding artifacts for this feature.
- Event Encoder piano-roll Ridge mode may expand a trusted ridge band's display
  top to the spectral centroid when scalar spectral-envelope descriptors show a
  broad tonal high-band event; this remains display-only and artifact-backed.
- Piano Roll Notes is a sidecar worker keyed on
  `(event_encoder_job_id, extractor_version)`. It reads completed Event Encoder
  outputs and the source audio, writes per-event MIDI notes to
  `event_notes_{extractor_version}.parquet`, and never modifies Event Encoder
  outputs. A complete Event Encoder job auto-enqueues a Piano Roll Notes job at
  the current default `extractor_version`; the auto-enqueue hook swallows
  conflicts so an in-flight or completed sidecar never blocks the encoder
  completion. `DEFAULT_EXTRACTOR_VERSION = "v7"` (ADR-074). Legacy
  `v1`, `v2`, `v3`, `v4`, and `v5` rows remain queryable through the existing
  API and pinned exports.
- Piano Roll Notes v4 (ADR-070) drops the STFT ridge band floor from
  100 Hz to 30 Hz and replaces the v3 octave-halving subharmonic
  refinement (`SubharmonicParams`) with HPS-style harmonic-stack F0
  scoring (`HPSParams`). The ridge tracker still seeds frame presence;
  HPS chooses which divisor `d ∈ {1..6}` of the ridge represents the
  true F0 each frame, scoring candidates by total harmonic-stack
  support across the first 8 partials with per-harmonic peak,
  noise-floor, and dynamic-range gates. Sub-100 Hz candidates need
  ≥ 3 surviving harmonics; ≥ 100 Hz candidates need ≥ 2. Frames where
  no candidate clears the gate fall back to the ridge as F0. v3
  `SubharmonicParams` stays in the v3 module for `params_json`
  round-trip parsing on historical rows.
- The `subharmonic_octave` column in `event_note_contours_*.parquet`
  records different quantities for v3, v4, and v5: v3 stores the
  octave halving count (0..3) chosen by `_refine_subharmonic`; v4
  stores `chosen_divisor − 1` (0..5) chosen by `_score_f0_candidates`;
  v5 reserves the column and always writes 0 (the harmonic-Viterbi
  algorithm has no divisor concept — ADR-071). The column name is
  preserved across versions because all three encodings answer the
  same diagnostic question ("how far did we shift the ridge to get
  F0?"). Renderers using it for diagnostic display work unchanged.
- Piano Roll Notes v5 (ADR-071) estimates F0 directly from the CQT
  via per-frame harmonic-sum emission plus log-frequency Viterbi
  smoothing. Per-frame `H_t(f₀) = Σ_{k=1..K} w_k · max(0, CQT_log[bin
  (f₀·k), t] − floor_t)` with K=4 (sacrifices higher harmonics for
  F0 fidelity per ADR-071 §4.1), `w_k = 1/√k`, and the v4
  noise-floor + 3-bin local-peak gates plus a `min_harmonics_present`
  gate and an `max_h1_below_strongest` H1-prominence gate. Voicing
  is a CQT-peakedness oracle (`column.max - noise_floor > tau_voicing`);
  background subtraction in `"pad"` mode samples per-bin chronic noise
  from the pad-zone CQT frames (frames outside the segmented event)
  and removes it before peakedness. STFT ridge band floor of 30 Hz
  is inherited from v4 but the v5 extractor does not consume the
  ridge sidecar — `ridge_sidecar_rows` is accepted for signature
  parity and ignored. Worker `pad_seconds` default for v5 is 0.25 s
  (was 0.05 s in v3/v4) so background subtraction has noise frames
  to sample. v3 and v4 sidecars on disk remain reachable via explicit
  `extractor_version` pinning.
- Piano Roll Notes v6 (ADR-072) is v5's decode plus a slope-based F0
  contour de-spike pass applied to each decoded F0 segment before note
  building. `extract_notes_v6` reuses v5's `_decode_f0` and v3's note
  builders unchanged. The de-spike is a slew-rate anchor walk: walk
  frames left→right holding a trusted anchor; when a later frame returns
  to within the anchor's slope envelope (`|Δlog₂f|` ≤ `max_slope_oct_per_s
  · dt · (frames since anchor)`), the intervening out-of-envelope frames
  were an out-and-back spike and are excised + linearly bridged so the
  note stays one continuous contour. Only out-and-back excursions are
  bridged (return-to-baseline): an excursion that never returns within
  `max_spike_frames` is a genuine level change (register jump, or a
  signal drop that resumes at a different pitch) — the walk re-anchors
  past it WITHOUT bridging and the real contour is left intact (added
  after the `cb23dfcd…` over-bridging finding, ADR-072 amendment). The
  one exception is a short non-returning excursion at the very end of a
  segment (≤ `max_trailing_trim_frames`): as a call's energy fades the
  tracker drops to a sub-fundamental, so that spurious tail is trimmed
  and the note (and its harmonics) end at the call (ADR-072 Amendment 2,
  events `2054e6de…` / `c82fa1fc…`); a leading non-returning excursion
  and a sustained end-of-call level change are still kept.
  `DespikeParams` defaults:
  `enabled = True`, `max_slope_oct_per_s = 6.0`, `max_spike_frames = 12`,
  `max_trailing_trim_frames = 4`;
  `enabled = False` makes v6 byte-identical to v5. Harmonic ribbons are
  corrected for free because harmonic presence is searched at
  `n · (cleaned f0)` and bends reuse the cleaned F0 cents. v6 inherits
  v5's worker defaults (30 Hz STFT floor, `min_break_frames = 6`,
  `pad_seconds = 0.25`) and emits `event_notes_v6.parquet` /
  `event_note_contours_v6.parquet` (schemas identical to v3–v5;
  `subharmonic_octave` reserved / written as 0). No auto-backfill of v6
  for completed v5 jobs.
- Piano Roll Notes v7 (ADR-074) is v6's decode/de-spike plus two
  post-decode passes before note building. Residual discontinuity splitting
  cuts adjacent retained F0 frames whose actual slope exceeds
  `max_continuous_slope_oct_per_s` (default 6.0 octaves/s), dropping
  fragments shorter than `segmentation.min_note_frames`; this turns
  surviving non-returning branch jumps into note boundaries rather than
  continuous MPE pitch bends. Ridge rescue rewrites flat decoded F0
  segments (`max_decoded_span_semitones = 2.0`) from the persisted Event
  Encoder STFT ridge when the overlap has at least `min_overlap_frames = 8`,
  the ridge span is at least 5 semitones, and the ridge is smooth after a
  linear trend fit (`max_ratio_mad_semitones = 2.0`). Rewritten frames
  preserve original timing, `frame_index`, strength, and
  `subharmonic_octave = 0`; harmonic siblings inherit the final F0 cents
  through the shared note builders. v7 inherits v5/v6 worker defaults
  (30 Hz STFT floor, `min_break_frames = 6`, `pad_seconds = 0.25`) and
  emits `event_notes_v7.parquet` / `event_note_contours_v7.parquet` with
  schemas identical to v3–v6.
- The MIDI export resolver picks the highest `complete` notes-job
  version by lexicographic ordering on `extractor_version`
  (`desc(extractor_version)`), so `"v7" > "v6" > "v5" > "v4" > "v3" >
  "v2" > "v1"`. A complete v7 row wins automatically when present; explicit version
  pinning via the `extractor_version=` argument still selects an older
  row. The MPE Lower Zone synthesizer detects MPE by the presence of
  the `note_uid` column, so v3-v7 sidecars route to the MPE path.
- Piano Roll Notes v3 uses the shared STFT ridge tracker
  (`humpback.processing.ridge_path.compute_ridge_path()`) as the canonical
  F0 source. The Event Encoder worker persists per-event ridge contours to
  `event_encoders/{job_id}/event_ridges_{tokenizer_version}.parquet`; the
  notes worker consumes the sidecar when present and falls back to
  in-process recompute when it is absent. Both paths use the same 6 kHz
  `max_frequency_hz` so the fallback ridges are indistinguishable from
  the persisted sidecar (ADR-069 §10). Encoder descriptor outputs are
  byte-identical before and after the extraction.
- Piano Roll Notes v3 pitch contours are sub-semitone and persisted: one
  row per frame per note in
  `event_encoders/{job_id}/event_note_contours_v3.parquet`, keyed on
  `note_uid`. The schema carries `time_offset_s`, `cents_from_pitch`
  (clamped to ±9600), `harmonic_strength`, and the smoothed
  `subharmonic_octave` produced by §5.2 of the ADR-069 spec. Cents are
  clamped even though the MPE bend range is ±2400 cents — the headroom
  guards against outlier frames before the bend quantizer rounds.
- Piano Roll Notes v3 emits one MIDI note per coherent F0 contour. The
  contour splits into separate notes only at energy gaps (≥ 3 frames below
  the per-frame amplitude floor) or surviving octave jumps from
  subharmonic refinement. Harmonic siblings are derived structurally at
  `n · f₀(t)` for `n ∈ {2..16}` with ±75¢ CQT-peak tolerance; harmonic
  notes carry `partial_index = n - 1` and `partial_index = -1` is no
  longer reachable.
- Harmonic notes inherit their parent F0's bend trajectory in cents
  (cents conservation): `1200 · log₂(n·f / n·f_nominal) =
  1200 · log₂(f / f_nominal)`. The CQT peak is used to validate harmonic
  presence only and never drives the bend stream. The harmonic contour
  reads the F0 cents keyed by the **original segment frame index**, not
  the 0-based `ContourFrame.frame_index` row position (ADR-073). The
  earlier key mismatch made harmonics of any event whose segment started
  after leading silence/pad borrow F0 cents from a time-shifted frame
  (or fall back to 0), producing an upper-harmonic "slope spike" ladder
  while the F0 itself was fine. The fix lives in the shared
  `_build_harmonic_notes`, so it corrects v3–v6 alike; on-disk parquet is
  untouched, but re-running v3/v4/v5 now yields corrected harmonics (the
  "byte-identical on re-run" property no longer holds for harmonic rows).
- Piano Roll Notes v3 rows carry a deterministic `note_uid` (UUID v5 of
  `(job_id, event_id, partial_index, track_id, start_utc_rounded_ms)`)
  plus `f0_track_id` and `contour_frame_count`. The MIDI pitch range
  widens to `[12, 120]` and contour frames render as curved ribbons in
  the Notes view by default.
- The Piano Roll Notes v1/v2 harmonic prior
  (`label_harmonics` in `src/humpback/processing/piano_roll_tracker.py`)
  and the `HarmonicParams` dataclass are retired by ADR-069. Existing
  v1/v2 rows on disk still resolve to the legacy code path because the
  notes worker branches on `job.extractor_version`; new jobs run v3.
- The Piano Roll MIDI export uses MPE Lower Zone (ADR-069) when the
  resolved notes-job version is `v3`: 15 member channels (1–15) with a
  per-member ±24-semitone pitch-bend range, deterministic longest-idle
  channel allocator with FIFO voice steal, per-note `program_change`
  (F0→0, H2→11, H3→12, H4→10, H5→8, H6..H16→88), CC 74 = `partial_index
  * 16`, master-track `MetaMessage("text", "pN")` events, and a 4¢
  bend-quantizer. The SMF has 17 tracks total (tempo + MPE Master + 15
  voice tracks); empty voice tracks still emit `track_name` and
  `end_of_track`. Identical parquet + contour sidecar produces
  byte-identical SMF bytes. Legacy `v1`/`v2` exports retain the slim
  seven-channel layout from ADR-067 (F0 → channel 1, 2nd–5th harmonics
  → channels 2–5, higher harmonics → channel 6, unmatched → channel 7,
  GM drum channel 10 intentionally empty); the synthesizer detects
  format by the presence of `note_uid` in the input parquet.
- Piano Roll exports are windowed and bundled (ADR-068). One canonical
  pair of artifacts exists per `(event_encoder_job_id, extractor_version)`:
  a `.mid` whose tick-0 origin equals the row's `window_start_utc`, and a
  co-exported `.flac` (32 kHz mono 16-bit PCM, NOT loudness-normalized)
  covering the same `[window_start_utc, window_end_utc)`. Re-export
  overwrites both files; the row's `window_*` columns are NOT NULL.
- The export window's duration must be strictly positive and ≤ 1800 s
  (30 min). The schema validator, the service layer, the API layer, and
  the frontend button all enforce this cap; the API also rejects windows
  that do not overlap the encoder's resolved data range (the
  `EventEncoderJob → EventSegmentationJob → RegionDetectionJob` chain).
- The MIDI synth's `time_origin_utc` argument defaults to the
  earliest-note shift (legacy behavior). When supplied, it anchors tick 0
  to that absolute UTC second so the windowed `.mid` lines up sample-wise
  with the co-exported `.flac` when both are imported into a DAW at the
  same project position with project tempo matching the file's tempo
  event.
