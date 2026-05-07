# Signal Timeline Invariants

- PCEN normalization applies to timeline spectrogram rendering only.
- Playback audio normalization applies to listening endpoints and export chunks,
  not to source audio or ML feature extraction.
- Timeline tile cache migrations preserve `.prepare_plan.json`,
  `.audio_manifest.json`, and `.last_access`.
- `TimelineProvider` owns playback, zoom, pan, viewport state, and keyboard
  shortcuts for compound timeline components.
- Consumers should use provider context or the playback ref handle instead of
  creating parallel audio elements or duplicate zoom state.
- Overlays render inside the `Spectrogram` coordinate system and use the overlay
  context for positioning.
- Tooltip overlays that must escape the canvas use the provided portal target.
- UTC labeling is mandatory for operational time ranges.
