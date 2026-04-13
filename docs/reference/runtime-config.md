# Runtime Configuration

`Settings` reads `HUMPBACK_`-prefixed environment variables.
API and worker entrypoints load the repo-root `.env`; direct `Settings()` does not.

## Core Settings

- `api_host` defaults to `0.0.0.0`, `api_port` to `8000`.
- `allowed_hosts` defaults to `*`. `HUMPBACK_ALLOWED_HOSTS` uses Starlette wildcard syntax.
- `positive_sample_path`, `negative_sample_path`, `s3_cache_path` derive from `storage_root` when unset.

## Timeline Settings

- `timeline_cache_max_jobs` defaults to `15`. `HUMPBACK_TIMELINE_CACHE_JOBS` controls how many detection jobs keep fully cached timeline tiles on disk (~8-16 GB at default). LRU eviction removes the oldest job when exceeded.
- `timeline_prepare_workers` defaults to `2`; startup/full tile batches may render through a bounded worker pool.
- `timeline_startup_radius_tiles` defaults to `2`; the Timeline button now triggers startup-scoped cache warming around the initial viewport rather than a full all-zoom warmup.
- `timeline_startup_coarse_levels` defaults to `1`, `timeline_neighbor_prefetch_radius` defaults to `1`, `timeline_tile_memory_cache_items` defaults to `256`, `timeline_manifest_memory_cache_items` defaults to `8`, and `timeline_pcm_memory_cache_mb` defaults to `128` for bounded in-memory timeline reuse.

## Replay & Tolerance

- `replay_metric_tolerance` defaults to `0.01`. `HUMPBACK_REPLAY_METRIC_TOLERANCE` controls the absolute tolerance for rate metrics (precision, recall, fp_rate, high_conf_fp_rate) during replay verification of candidate-backed training. Count metrics (tp, fp, fn, tn) must match exactly.

## PCEN (Timeline Spectrogram)

- `pcen_time_constant_sec` (`2.0`), `pcen_gain` (`0.98`), `pcen_bias` (`2.0`), `pcen_power` (`0.5`), `pcen_eps` (`1e-6`): PCEN parameters applied per timeline tile via `librosa.pcen`. Overridable via `HUMPBACK_PCEN_*`. The per-bin low-pass filter state is pre-initialized to the first STFT frame's magnitude (scaled `lfilter_zi`), eliminating librosa's default unit-step cold-start transient that would otherwise paint a dark strip at the left edge of every tile.
- `pcen_warmup_sec` defaults to `2.0`. Each tile's audio fetch is extended backward by this amount so the PCEN low-pass filter can settle before the first rendered frame; the warm-up frames are trimmed off the output. Redundant with warm-zi initialization for stationary signals, but still useful when the signal changes across the tile boundary.
- `pcen_vmin` (`0.0`) / `pcen_vmax` (`1.0`): fixed colormap range for PCEN-normalized tiles. Because PCEN's output is bounded, there is no per-job `ref_db` computation.

## Playback

- `playback_target_rms_dbfs` defaults to `-20.0`. `HUMPBACK_PLAYBACK_TARGET_RMS_DBFS` controls the RMS level that timeline playback chunks are scaled to before MP3/WAV encoding.
- `playback_ceiling` defaults to `0.95`. `HUMPBACK_PLAYBACK_CEILING` is the `tanh` soft-clip ceiling applied after RMS scaling to prevent harsh clipping on transients.
