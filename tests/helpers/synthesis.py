"""Audio synthesis helpers retained for synthesis quality regression tests."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthesizedSample:
    """A synthesized fixed-duration audio clip."""

    audio_segment: np.ndarray
    sr: int
    call_type: str
    treatment: str
    source_filename: str
    start_sec: float
    end_sec: float
    peak_score: float


def _raised_cosine_fade(length: int) -> np.ndarray:
    """Return a half-cosine ramp from 0 to 1 of *length* samples."""
    from humpback.processing.dsp import raised_cosine_fade

    return raised_cosine_fade(length)


def synthesize_clean_window(
    call_segment: np.ndarray,
    background_segment: np.ndarray,
    sr: int,
    window_size: float = 5.0,
    placement_sec: float | None = None,
    crossfade_ms: float = 50.0,
) -> SynthesizedSample | None:
    """Place a call segment into a background window with crossfade splicing."""
    expected_samples = int(window_size * sr)
    if len(background_segment) < expected_samples:
        return None

    bg = background_segment[:expected_samples].copy().astype(np.float32)
    call = call_segment.astype(np.float32)

    call_samples = len(call)
    if call_samples >= expected_samples:
        return SynthesizedSample(
            audio_segment=call[:expected_samples],
            sr=sr,
            call_type="",
            treatment="synthesized",
            source_filename="",
            start_sec=0.0,
            end_sec=window_size,
            peak_score=0.0,
        )

    if placement_sec is None:
        placement_sec = (window_size - call_samples / sr) / 2.0
    insert_sample = int(max(0.0, placement_sec) * sr)
    insert_sample = min(insert_sample, expected_samples - call_samples)

    bg_rms = float(np.sqrt(np.mean(bg**2))) + 1e-10
    call_rms = float(np.sqrt(np.mean(call**2))) + 1e-10
    target_bg_rms = 0.2 * call_rms
    bg *= target_bg_rms / bg_rms

    bg_peak = float(np.max(np.abs(bg)))
    bg_peak_limit = 6.0 * target_bg_rms
    if bg_peak > bg_peak_limit:
        np.clip(bg, -bg_peak_limit, bg_peak_limit, out=bg)

    fade_len = max(1, int(crossfade_ms / 1000.0 * sr))
    end_sample = insert_sample + call_samples

    pre_fade = min(fade_len, insert_sample, call_samples)
    post_fade = min(fade_len, expected_samples - end_sample, call_samples)

    if pre_fade + post_fade > call_samples:
        half = call_samples // 2
        pre_fade = min(pre_fade, half)
        post_fade = min(post_fade, call_samples - pre_fade)

    result = bg.copy()

    if pre_fade > 0:
        ramp = _raised_cosine_fade(pre_fade)
        idx = slice(insert_sample, insert_sample + pre_fade)
        result[idx] = bg[idx] * (1.0 - ramp) + call[:pre_fade] * ramp

    pure_start = insert_sample + pre_fade
    pure_end = end_sample - post_fade
    if pure_start < pure_end:
        result[pure_start:pure_end] = call[pre_fade : call_samples - post_fade]

    if post_fade > 0:
        ramp = _raised_cosine_fade(post_fade)
        idx = slice(end_sample - post_fade, end_sample)
        result[idx] = call[-post_fade:] * (1.0 - ramp) + bg[idx] * ramp

    return SynthesizedSample(
        audio_segment=result[:expected_samples],
        sr=sr,
        call_type="",
        treatment="synthesized",
        source_filename="",
        start_sec=0.0,
        end_sec=window_size,
        peak_score=0.0,
    )


def synthesize_variants(
    call_segment: np.ndarray,
    backgrounds: list[np.ndarray],
    sr: int,
    window_size: float = 5.0,
    crossfade_ms: float = 50.0,
    n_variants: int = 3,
    bg_offset: int = 0,
) -> list[SynthesizedSample]:
    """Generate up to *n_variants* placement variants of a call."""
    call_dur = len(call_segment) / sr
    placements = [
        0.75,
        (window_size - call_dur) / 2.0,
        window_size - call_dur - 0.75,
    ]
    placements = [max(0.0, min(p, window_size - call_dur)) for p in placements]

    results: list[SynthesizedSample] = []
    for i in range(min(n_variants, len(placements))):
        bg_idx = (bg_offset + i) % len(backgrounds) if backgrounds else -1
        bg = backgrounds[bg_idx] if bg_idx >= 0 else None
        if bg is None:
            continue
        sample = synthesize_clean_window(
            call_segment,
            bg,
            sr,
            window_size=window_size,
            placement_sec=placements[i],
            crossfade_ms=crossfade_ms,
        )
        if sample is not None:
            results.append(sample)

    return results
