"""Unit tests for sample_builder Stage 10 — full pipeline orchestrator."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.sample_builder.pipeline import (
    SampleBuilderConfig,
    build_samples_for_recording,
)

SR = 32000


def _make_recording_with_annotations(
    tmp_dir: Path,
    duration_sec: float = 30.0,
) -> tuple[Path, list[RavenAnnotation]]:
    """Create a synthetic recording + annotations for testing."""
    rng = np.random.default_rng(42)

    # Background noise
    audio = rng.normal(0, 0.001, int(duration_sec * SR)).astype(np.float32)

    # Inject tonal calls at annotation positions
    annotations = [
        RavenAnnotation(1, 5.0, 6.0, 200, 2000, "Whup"),
        RavenAnnotation(2, 15.0, 16.0, 200, 2000, "Chirp"),
        RavenAnnotation(3, 25.0, 26.0, 200, 2000, "Whup"),
    ]
    for ann in annotations:
        t = np.arange(int((ann.end_time - ann.begin_time) * SR)) / SR
        call = 0.1 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
        start = int(ann.begin_time * SR)
        audio[start : start + len(call)] += call

    audio_path = tmp_dir / "test_recording.wav"
    sf.write(str(audio_path), audio, SR)

    return audio_path, annotations


class TestBuildSamplesForRecording:
    def test_produces_results_for_all_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, annotations = _make_recording_with_annotations(tmp_path)

            config = SampleBuilderConfig(target_sr=SR)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            assert len(results) == len(annotations)

    def test_accepted_samples_have_correct_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, annotations = _make_recording_with_annotations(tmp_path)

            config = SampleBuilderConfig(target_sr=SR, window_size=5.0)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            for result in results:
                if result.accepted:
                    assert result.audio is not None
                    assert len(result.audio) == int(5.0 * SR)
                    assert result.sr == SR

    def test_rejected_samples_have_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, annotations = _make_recording_with_annotations(tmp_path)

            config = SampleBuilderConfig(target_sr=SR)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            for result in results:
                if not result.accepted:
                    assert result.rejection_reason is not None

    def test_call_types_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, annotations = _make_recording_with_annotations(tmp_path)

            config = SampleBuilderConfig(target_sr=SR)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            call_types = {r.call_type for r in results}
            assert "Whup" in call_types
            assert "Chirp" in call_types

    def test_source_filename_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, annotations = _make_recording_with_annotations(tmp_path)

            config = SampleBuilderConfig(target_sr=SR)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            for result in results:
                assert result.source_filename == "test_recording"

    def test_invalid_annotation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, _ = _make_recording_with_annotations(tmp_path)

            # Too-short annotation
            annotations = [
                RavenAnnotation(1, 5.0, 5.1, 200, 2000, "Whup"),  # 0.1s < 0.3s min
            ]
            config = SampleBuilderConfig(target_sr=SR)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            assert len(results) == 1
            assert results[0].accepted is False
            assert results[0].rejection_reason == "invalid_annotation"

    def test_empty_annotations_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, _ = _make_recording_with_annotations(tmp_path)

            results = build_samples_for_recording(audio_path, [], sr=SR)
            assert results == []

    def test_marine_recording_accepts_annotations(self) -> None:
        """Pipeline should accept annotations from a marine-like pink noise recording."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rng = np.random.default_rng(42)

            # Generate 60s pink noise recording (ocean ambient)
            n = int(60.0 * SR)
            white = rng.normal(0, 1, n).astype(np.float64)
            spectrum = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n, 1.0 / SR)
            freqs[0] = 1.0
            spectrum /= np.sqrt(freqs)
            audio = np.fft.irfft(spectrum, n=n)
            audio_rms = float(np.sqrt(np.mean(audio**2)))
            if audio_rms > 0:
                audio *= 0.01 / audio_rms
            audio = audio.astype(np.float32)

            # Embed 3 whale-like tonal calls
            annotations = [
                RavenAnnotation(1, 10.0, 11.0, 200, 2000, "Whup"),
                RavenAnnotation(2, 30.0, 31.0, 200, 2000, "Chirp"),
                RavenAnnotation(3, 50.0, 51.0, 200, 2000, "Whup"),
            ]
            for ann in annotations:
                t = np.arange(int((ann.end_time - ann.begin_time) * SR)) / SR
                call = 0.05 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
                start = int(ann.begin_time * SR)
                audio[start : start + len(call)] += call

            audio_path = tmp_path / "marine_recording.wav"
            sf.write(str(audio_path), audio, SR)

            config = SampleBuilderConfig(target_sr=SR)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            accepted = [r for r in results if r.accepted]
            assert len(accepted) >= 2, (
                f"Only {len(accepted)}/3 accepted. Rejections: "
                + ", ".join(
                    f"{r.call_type}@{r.annotation.begin_time if r.annotation else '?'}s"
                    f"={r.rejection_reason}"
                    for r in results
                    if not r.accepted
                )
            )

    def test_metadata_populated_for_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path, annotations = _make_recording_with_annotations(tmp_path)

            config = SampleBuilderConfig(target_sr=SR)
            results = build_samples_for_recording(
                audio_path, annotations, sr=SR, config=config
            )

            for result in results:
                if result.accepted:
                    assert result.metadata is not None
                    assert result.metadata.window_size_sec == 5.0
                    assert result.metadata.target_duration_sec > 0
