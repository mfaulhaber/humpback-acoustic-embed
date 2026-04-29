"""Pass 2 CRNN BiGRU activations as chunk embeddings.

Sole touchpoint between the new Sequence Models region-CRNN source and
Pass 2 internals. Loads a ``SegmentationCRNN`` checkpoint, runs windowed
inference via the shared ``iter_inference_windows`` helper, captures
BiGRU activations through a non-invasive forward hook (no edits to
``SegmentationCRNN.forward()``), upsamples them to the spectrogram frame
rate the model already restores its frame head to, stitches overlapping
windows by keeping the centre half, slices the stitched activations
into N-frame chunks at the requested chunk hop, concatenates the
per-chunk frames, applies a ``ChunkProjection``, and emits a
``ChunkEmbeddingResult`` with per-chunk timestamps and per-chunk mean
call probability.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from humpback.call_parsing.segmentation.features import (
    extract_logmel,
    normalize_per_region_zscore,
)
from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.call_parsing.segmentation.window_iter import (
    MAX_WINDOW_SEC,
    WINDOW_HOP_SEC,
    iter_inference_windows,
)
from humpback.ml.checkpointing import load_checkpoint
from humpback.schemas.call_parsing import SegmentationFeatureConfig
from humpback.sequence_models.chunk_projection import ChunkProjection

# The Pass 2 CRNN currently ships with ``gru_hidden=64`` and a stride-2
# time pool in the last conv block, giving a BiGRU output width of 128
# at the spectrogram frame rate (~32 fps with hop_length=512 / sr=16k).
# These are the values the embedding extractor was designed against; if
# a checkpoint deviates, the producer fails fast at load time.
EXPECTED_BIGRU_WIDTH = 128
EXPECTED_FRAME_RATE_HZ = 32.0
FRAME_RATE_TOLERANCE_HZ = 1.5


@dataclass(frozen=True)
class ChunkEmbeddingResult:
    """Per-chunk extractor output for one region.

    All arrays are aligned along axis 0 with one entry per chunk.
    ``chunk_starts`` and ``chunk_ends`` are seconds relative to the
    region's audio (i.e. ``audio[0]`` is ``t = 0``).
    """

    embeddings: np.ndarray  # (T_chunks, projection.output_dim) float32
    call_probabilities: np.ndarray  # (T_chunks,) float32
    chunk_starts: np.ndarray  # (T_chunks,) float64 seconds
    chunk_ends: np.ndarray  # (T_chunks,) float64 seconds


def compute_checkpoint_sha256(checkpoint_path: str | Path) -> str:
    """Return the hex SHA-256 digest of the checkpoint file."""
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instantiate_from_config(model_config: dict[str, Any]) -> SegmentationCRNN:
    """Build a ``SegmentationCRNN`` matching the persisted architecture."""
    n_mels = int(model_config.get("n_mels", 64))
    conv_channels_raw = model_config.get("conv_channels", [32, 64, 96, 128])
    conv_channels = [int(c) for c in conv_channels_raw]
    gru_hidden = int(model_config.get("gru_hidden", 64))
    gru_layers = int(model_config.get("gru_layers", 2))
    return SegmentationCRNN(
        n_mels=n_mels,
        conv_channels=conv_channels,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
    )


def _validate_model(
    model: SegmentationCRNN, feature_config: SegmentationFeatureConfig
) -> None:
    """Assert the BiGRU width and feature frame rate match expectations."""
    bigru_width = 2 * model.gru_hidden
    if bigru_width != EXPECTED_BIGRU_WIDTH:
        raise ValueError(
            "CRNN extractor expects BiGRU output width "
            f"{EXPECTED_BIGRU_WIDTH} (gru_hidden=64, bidirectional=True); "
            f"got width={bigru_width} (gru_hidden={model.gru_hidden})"
        )
    frame_rate = feature_config.sample_rate / feature_config.hop_length
    if abs(frame_rate - EXPECTED_FRAME_RATE_HZ) > FRAME_RATE_TOLERANCE_HZ:
        raise ValueError(
            "CRNN extractor expects a feature frame rate of "
            f"~{EXPECTED_FRAME_RATE_HZ} Hz (got {frame_rate:.3f} Hz from "
            f"sample_rate={feature_config.sample_rate}, "
            f"hop_length={feature_config.hop_length})"
        )


@dataclass(frozen=True)
class LoadedCRNN:
    """A loaded segmentation CRNN ready for chunk extraction."""

    model: SegmentationCRNN
    feature_config: SegmentationFeatureConfig
    checkpoint_sha256: str


def load_crnn_for_extraction(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LoadedCRNN:
    """Load a SegmentationCRNN checkpoint and validate it for extraction.

    The caller is responsible for placing the model on the right device;
    this function moves it onto ``device`` and switches to ``eval`` mode
    so the extractor never re-touches gradient state.
    """
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config_raw = payload.get("config", {})
    if not isinstance(model_config_raw, dict):
        raise ValueError(f"checkpoint {path} has non-dict config")
    feature_config_raw = model_config_raw.get("feature_config") or {}
    if not isinstance(feature_config_raw, dict):
        raise ValueError(f"checkpoint {path} has non-dict feature_config")
    feature_config = SegmentationFeatureConfig(**feature_config_raw)

    model = _instantiate_from_config(model_config_raw)
    load_checkpoint(path, model)
    model.eval()
    model.to(device)

    _validate_model(model, feature_config)

    return LoadedCRNN(
        model=model,
        feature_config=feature_config,
        checkpoint_sha256=compute_checkpoint_sha256(path),
    )


def _run_window_with_hook(
    model: SegmentationCRNN,
    chunk_audio: np.ndarray,
    feature_config: SegmentationFeatureConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward one window through the model, return ``(activations, probs)``.

    ``activations`` shape is ``(T_feat, 128)`` at the spectrogram frame
    rate (the BiGRU output is upsampled the same way the model upsamples
    its frame logits, mirroring ``SegmentationCRNN.forward``).
    ``probs`` shape is ``(T_feat,)``.
    """
    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        # ``GRU.forward`` returns ``(output, h_n)`` — we want ``output``,
        # of shape (B, T_conv, 2*hidden_size).
        if isinstance(output, tuple):
            captured.append(output[0].detach())
        else:
            captured.append(output.detach())

    handle = model.gru.register_forward_hook(_hook)
    try:
        logmel = normalize_per_region_zscore(
            extract_logmel(chunk_audio, feature_config)
        )
        features_t = (
            torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).float().to(device)
        )
        with torch.no_grad():
            logits = model(features_t)
            probs_t = torch.sigmoid(logits).squeeze(0)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("BiGRU forward hook never fired")
    bigru_out = captured[-1]  # (1, T_conv, 128)

    # Upsample BiGRU output along the time axis to the spectrogram frame
    # count using nearest-neighbor — same operation the model applies to
    # its frame logits.
    target_t = int(probs_t.shape[-1])
    activations = (
        F.interpolate(
            bigru_out.transpose(1, 2),  # (1, 128, T_conv)
            size=target_t,
            mode="nearest",
        )
        .squeeze(0)  # (128, T_feat)
        .transpose(0, 1)  # (T_feat, 128)
    )

    return (
        activations.cpu().numpy().astype(np.float32),
        probs_t.detach().cpu().numpy().astype(np.float32),
    )


def _stitch_centre_half(
    pieces: list[tuple[int, np.ndarray]],
    total_frames: int,
) -> np.ndarray:
    """Stitch overlapping window outputs by ownership-mid-point.

    Each ``pieces`` entry is ``(frame_offset, array)`` where ``array``
    has shape ``(W_frames, ...)``. For interior windows, the "centre
    half" rule reduces to: the boundary between two adjacent windows is
    the midpoint of their overlap. The first window owns from 0; the
    last window owns through ``total_frames``.
    """
    if not pieces:
        raise ValueError("Cannot stitch empty pieces")
    feat_dim = pieces[0][1].shape[1] if pieces[0][1].ndim == 2 else 0
    out_shape: tuple[int, ...]
    if pieces[0][1].ndim == 2:
        out_shape = (total_frames, feat_dim)
    else:
        out_shape = (total_frames,)
    out = np.zeros(out_shape, dtype=pieces[0][1].dtype)

    n = len(pieces)
    for i, (offset, arr) in enumerate(pieces):
        end = offset + arr.shape[0]
        if i == 0:
            own_start = 0
        else:
            prev_offset, prev_arr = pieces[i - 1]
            prev_end = prev_offset + prev_arr.shape[0]
            own_start = (offset + prev_end) // 2
        if i == n - 1:
            own_end = total_frames
        else:
            next_offset, next_arr = pieces[i + 1]
            own_end = (next_offset + end) // 2

        own_start = max(own_start, offset)
        own_end = min(own_end, end, total_frames)
        if own_start >= own_end:
            continue

        local_start = own_start - offset
        local_end = own_end - offset
        out[own_start:own_end] = arr[local_start:local_end]

    return out


def _frames_for_seconds(seconds: float, frame_rate: float) -> int:
    """Round seconds to an integer frame count."""
    return int(round(seconds * frame_rate))


def _iter_chunk_slices(
    n_frames: int, frames_per_chunk: int, frame_hop: int
) -> Iterator[tuple[int, int]]:
    """Yield ``(start_frame, end_frame)`` for each non-truncated chunk."""
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be positive")
    if frame_hop <= 0:
        raise ValueError("frame_hop must be positive")
    start = 0
    while start + frames_per_chunk <= n_frames:
        yield start, start + frames_per_chunk
        start += frame_hop


def extract_chunk_embeddings(
    *,
    model: SegmentationCRNN,
    audio: np.ndarray,
    feature_config: SegmentationFeatureConfig,
    chunk_size_seconds: float,
    chunk_hop_seconds: float,
    projection: ChunkProjection,
    device: torch.device,
    window_seconds: float = MAX_WINDOW_SEC,
    window_hop_seconds: float = WINDOW_HOP_SEC,
) -> ChunkEmbeddingResult:
    """Run the CRNN over ``audio`` and emit per-chunk embeddings.

    ``audio`` is the region's full padded waveform at
    ``feature_config.sample_rate``. ``projection`` must already be
    fitted (or be an ``IdentityProjection``); the producer is
    responsible for that.
    """
    _validate_model(model, feature_config)
    if chunk_size_seconds <= 0 or chunk_hop_seconds <= 0:
        raise ValueError("chunk_size_seconds and chunk_hop_seconds must be > 0")

    sr = feature_config.sample_rate
    hop_length = feature_config.hop_length
    frame_rate = sr / hop_length
    frames_per_chunk = _frames_for_seconds(chunk_size_seconds, frame_rate)
    frame_hop = _frames_for_seconds(chunk_hop_seconds, frame_rate)
    if frames_per_chunk < 1:
        raise ValueError(
            f"chunk_size_seconds={chunk_size_seconds} resolves to <1 frame at "
            f"frame_rate={frame_rate:.3f} Hz"
        )
    if frame_hop < 1:
        raise ValueError(
            f"chunk_hop_seconds={chunk_hop_seconds} resolves to <1 frame at "
            f"frame_rate={frame_rate:.3f} Hz"
        )

    total_samples = len(audio)
    if total_samples == 0:
        empty_emb = np.zeros((0, projection.output_dim), dtype=np.float32)
        empty_t = np.zeros((0,), dtype=np.float64)
        empty_p = np.zeros((0,), dtype=np.float32)
        return ChunkEmbeddingResult(empty_emb, empty_p, empty_t, empty_t)

    # ``+ 1`` mirrors librosa's ``center=True`` frame-count formula used
    # by ``extract_logmel``.
    total_frames = 1 + total_samples // hop_length

    activation_pieces: list[tuple[int, np.ndarray]] = []
    prob_pieces: list[tuple[int, np.ndarray]] = []
    for window_audio, frame_offset in iter_inference_windows(
        audio,
        sample_rate=sr,
        frame_hop_samples=hop_length,
        window_seconds=window_seconds,
        hop_seconds=window_hop_seconds,
    ):
        activations, probs = _run_window_with_hook(
            model, window_audio, feature_config, device
        )
        activation_pieces.append((frame_offset, activations))
        prob_pieces.append((frame_offset, probs))

    stitched_activations = _stitch_centre_half(activation_pieces, total_frames)
    stitched_probs = _stitch_centre_half(prob_pieces, total_frames)

    chunk_indices = list(_iter_chunk_slices(total_frames, frames_per_chunk, frame_hop))
    if not chunk_indices:
        # Region too short to fit one chunk.
        empty_emb = np.zeros((0, projection.output_dim), dtype=np.float32)
        empty_t = np.zeros((0,), dtype=np.float64)
        empty_p = np.zeros((0,), dtype=np.float32)
        return ChunkEmbeddingResult(empty_emb, empty_p, empty_t, empty_t)

    n_chunks = len(chunk_indices)
    bigru_width = stitched_activations.shape[1]
    concat_dim = frames_per_chunk * bigru_width
    concat = np.empty((n_chunks, concat_dim), dtype=np.float32)
    call_probs = np.empty((n_chunks,), dtype=np.float32)
    chunk_starts = np.empty((n_chunks,), dtype=np.float64)
    chunk_ends = np.empty((n_chunks,), dtype=np.float64)

    frame_period = 1.0 / frame_rate
    for i, (start_f, end_f) in enumerate(chunk_indices):
        slice_act = stitched_activations[start_f:end_f]
        concat[i] = slice_act.reshape(-1)
        call_probs[i] = float(stitched_probs[start_f:end_f].mean())
        chunk_starts[i] = start_f * frame_period
        chunk_ends[i] = end_f * frame_period

    embeddings = projection.transform(concat).astype(np.float32, copy=False)
    if embeddings.shape != (n_chunks, projection.output_dim):
        raise RuntimeError(
            "Projection produced unexpected shape "
            f"{embeddings.shape} (expected {(n_chunks, projection.output_dim)})"
        )

    return ChunkEmbeddingResult(
        embeddings=embeddings,
        call_probabilities=call_probs,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
    )
