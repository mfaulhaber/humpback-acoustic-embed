"""Extract labeled audio samples from detection TSV files."""

import csv
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from humpback.classifier.archive import ArchiveProvider
from humpback.classifier.detection_rows import (
    POSITIVE_SELECTION_FIELDNAMES,
    POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK,
    POSITIVE_SELECTION_SCORE_SOURCE_STORED,
    ROW_STORE_FIELDNAMES,
    PositiveSelectionResult,
    blank_positive_selection_fields,
    derive_detection_filename,
    is_negative_row,
    is_positive_row,
    negative_labels_for_row,
    parse_recording_timestamp,
    positive_labels_for_row,
    read_detection_row_store,
    safe_float,
    safe_float_list,
    safe_optional_float,
    select_positive_window,
    selection_result_to_row_update,
    write_detection_row_store,
)
from humpback.classifier.detector import read_window_diagnostics_table
from humpback.classifier.s3_stream import (
    AUDIO_SLICE_GUARD_SAMPLES,
    build_stream_timeline,
    expected_audio_samples,
    resolve_audio_slice,
)
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.spectrogram import generate_spectrogram_png
from humpback.processing.windowing import slice_windows_with_metadata

logger = logging.getLogger(__name__)

_COMPACT_TS_FORMAT = "%Y%m%dT%H%M%SZ"
_KNOWN_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
_OUTPUT_AUDIO_EXTENSION = ".flac"
DEFAULT_SPECTROGRAM_HOP_LENGTH = 256
DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB = 80.0
DEFAULT_SPECTROGRAM_WIDTH_PX = 640
DEFAULT_SPECTROGRAM_HEIGHT_PX = 320
DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW = 3
DEFAULT_POSITIVE_SELECTION_MIN_SCORE = 0.70
DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE = 0.60


def _date_folder_from_epoch(epoch: float) -> str:
    """Return YYYY/MM/dd path component from a UTC epoch float."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%Y/%m/%d")


def _file_base_epoch(filepath: Path) -> float:
    """Return the UTC epoch for a file: from timestamp in name, mtime, or 0.0."""
    ts = parse_recording_timestamp(filepath.name)
    if ts is not None:
        return ts.timestamp()
    try:
        return os.path.getmtime(filepath)
    except OSError:
        return 0.0


def _build_local_audio_index(
    audio_folder: Path,
) -> list[tuple[float, Path]]:
    """Build a sorted index of (base_epoch, path) for audio files in a folder."""
    index: list[tuple[float, Path]] = []
    for child in sorted(audio_folder.iterdir()):
        if child.suffix.lower() in _KNOWN_AUDIO_EXTENSIONS and child.is_file():
            index.append((_file_base_epoch(child), child))
    index.sort(key=lambda x: x[0])
    return index


def _resolve_local_audio_for_row(
    start_utc: float,
    audio_index: list[tuple[float, Path]],
) -> tuple[Path, float, float] | None:
    """Map start_utc to (file_path, base_epoch, offset_sec).

    Returns the audio file whose base_epoch is largest among those <= start_utc.
    """
    best: tuple[Path, float, float] | None = None
    for base_epoch, path in audio_index:
        if base_epoch <= start_utc + 1e-6:
            best = (path, base_epoch, start_utc - base_epoch)
        else:
            break
    return best


def _normalize_local_rows_to_utc(
    rows: list[dict[str, str]],
    audio_index: list[tuple[float, Path]],
) -> None:
    """Ensure every row has ``start_utc``/``end_utc``.

    For rows already carrying these fields (row-store) this is a no-op.
    For legacy TSV rows we derive UTC from ``filename`` + ``start_sec``.
    """
    for row in rows:
        if row.get("start_utc"):
            continue
        filename = row.get("filename", "")
        start_sec = safe_float(row.get("start_sec"), 0.0)
        end_sec = safe_float(row.get("end_sec"), 0.0)
        base_epoch = 0.0
        if filename:
            ts = parse_recording_timestamp(filename)
            if ts is not None:
                base_epoch = ts.timestamp()
            else:
                for epoch, path in audio_index:
                    if path.name == filename:
                        base_epoch = epoch
                        break
        row["start_utc"] = str(base_epoch + start_sec)
        row["end_utc"] = str(base_epoch + end_sec)


def _normalize_hydrophone_rows_to_utc(
    rows: list[dict[str, str]],
    window_size_seconds: float,
) -> None:
    """Ensure every hydrophone row has ``start_utc``/``end_utc``.

    For rows already carrying these fields (row-store) this is a no-op.
    For legacy TSV rows we derive UTC from ``detection_filename``,
    ``extract_filename``, or ``filename`` + ``start_sec``/``end_sec``.
    """
    for row in rows:
        if row.get("start_utc"):
            continue
        filename = row.get("filename", "")
        recording_ts = parse_recording_timestamp(filename) if filename else None
        start_sec = safe_float(row.get("start_sec"), 0.0)
        end_sec = safe_float(row.get("end_sec"), 0.0)

        # Try detection_filename or extract_filename for exact UTC bounds.
        for field in ("detection_filename", "extract_filename"):
            candidate = row.get(field, "").strip()
            if candidate:
                parsed = _parse_compact_range_filename(candidate)
                if parsed is not None:
                    abs_start, abs_end = parsed
                    row["start_utc"] = str(abs_start.timestamp())
                    row["end_utc"] = str(abs_end.timestamp())
                    break
        else:
            if recording_ts is not None:
                import math

                snap_start = (
                    math.floor(start_sec / window_size_seconds) * window_size_seconds
                )
                snap_end = (
                    math.ceil(end_sec / window_size_seconds) * window_size_seconds
                )
                if snap_end <= snap_start:
                    snap_end = snap_start + window_size_seconds
                base_epoch = recording_ts.timestamp()
                row["start_utc"] = str(base_epoch + snap_start)
                row["end_utc"] = str(base_epoch + snap_end)
            else:
                row["start_utc"] = str(start_sec)
                row["end_utc"] = str(end_sec)


def _with_output_audio_extension(filename: str) -> str:
    """Return filename with the configured extracted-audio extension."""
    from humpback.classifier.detection_rows import strip_known_audio_extension

    return f"{strip_known_audio_extension(filename)}{_OUTPUT_AUDIO_EXTENSION}"


def _spectrogram_sidecar_path(audio_output_path: Path) -> Path:
    """Return the sidecar spectrogram path for an extracted audio output."""
    return audio_output_path.with_suffix(".png")


def _validate_positive_selection_config(
    smoothing_window: int,
    min_score: float,
    extend_min_score: float,
) -> None:
    """Validate positive-window selection parameters."""
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError(
            "positive_selection_smoothing_window must be an odd integer >= 1"
        )
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("positive_selection_min_score must be between 0.0 and 1.0")
    if not 0.0 <= extend_min_score <= 1.0:
        raise ValueError(
            "positive_selection_extend_min_score must be between 0.0 and 1.0"
        )


def _read_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read all TSV rows while preserving the original field order."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _read_detection_rows(
    tsv_path: Path,
    *,
    row_store_path: Path | None = None,
) -> tuple[list[str], list[dict[str, str]], bool]:
    """Read detection rows from the canonical row store when available."""
    if row_store_path is not None and row_store_path.exists():
        fieldnames, rows = read_detection_row_store(row_store_path)
        return fieldnames, rows, True
    fieldnames, rows = _read_tsv_rows(tsv_path)
    return fieldnames, rows, False


def _write_detection_rows(
    rows: list[dict[str, str]],
    *,
    row_store_path: Path | None = None,
) -> None:
    """Persist detection rows to the row store.

    When row_store_path is None (legacy TSV-only mode) the update is a no-op;
    callers that need persistence must supply a row_store_path.
    """
    if row_store_path is not None:
        write_detection_row_store(row_store_path, rows)


def _ensure_fieldnames(fieldnames: list[str], required: list[str]) -> list[str]:
    """Append required fieldnames without disturbing existing order."""
    for field in required:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def _selection_from_effective_row(
    row: dict[str, str],
) -> PositiveSelectionResult | None:
    decision = row.get("positive_selection_decision", "").strip()
    start_utc = safe_optional_float(row.get("positive_selection_start_utc"))
    end_utc = safe_optional_float(row.get("positive_selection_end_utc"))
    if not decision:
        return None
    return PositiveSelectionResult(
        score_source=row.get("positive_selection_score_source", "").strip()
        or POSITIVE_SELECTION_SCORE_SOURCE_STORED,
        decision="positive" if decision == "positive" else "skip",
        offsets=safe_float_list(row.get("positive_selection_offsets")) or [],
        raw_scores=safe_float_list(row.get("positive_selection_raw_scores")) or [],
        smoothed_scores=safe_float_list(row.get("positive_selection_smoothed_scores"))
        or [],
        start_utc=start_utc,
        end_utc=end_utc,
        peak_score=safe_optional_float(row.get("positive_selection_peak_score")),
    )


def _load_window_records(
    diagnostics_path: Path | None,
    *,
    filename: str,
    base_epoch: float = 0.0,
) -> list[dict[str, float]] | None:
    """Load stored diagnostics rows for a source filename, shifted to UTC.

    When *base_epoch* is non-zero the file-relative ``offset_sec``/``end_sec``
    values are shifted to absolute UTC epoch seconds so they are directly
    comparable with ``start_utc``/``end_utc`` row identity fields.
    """
    if diagnostics_path is None or not diagnostics_path.exists():
        return None
    try:
        table = read_window_diagnostics_table(diagnostics_path, filename=filename)
    except Exception:
        logger.debug(
            "Failed to read stored window diagnostics for %s from %s",
            filename,
            diagnostics_path,
            exc_info=True,
        )
        return None

    records: list[dict[str, float]] = []
    for i in range(table.num_rows):
        records.append(
            {
                "offset_sec": float(table.column("offset_sec")[i].as_py()) + base_epoch,
                "end_sec": float(table.column("end_sec")[i].as_py()) + base_epoch,
                "confidence": float(table.column("confidence")[i].as_py()),
            }
        )
    return records


def _score_segment_windows(
    audio_segment: np.ndarray,
    *,
    source_sr: int,
    row_start_utc: float,
    pipeline: Any,
    model: EmbeddingModel,
    window_size_seconds: float,
    target_sample_rate: int,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
    hop_seconds: float = 1.0,
) -> list[dict[str, float]]:
    """Fallback: score candidate windows by re-running the classifier on a clip."""
    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")
    audio = (
        resample(audio_segment, source_sr, target_sample_rate)
        if source_sr != target_sample_rate
        else audio_segment
    )

    raw_windows: list[np.ndarray] = []
    offsets: list[float] = []
    for window, meta in slice_windows_with_metadata(
        audio,
        target_sample_rate,
        window_size_seconds,
        hop_seconds=hop_seconds,
    ):
        raw_windows.append(window)
        offsets.append(row_start_utc + meta.offset_sec)

    if not raw_windows:
        return []

    if input_format == "waveform":
        batch_items: list[np.ndarray] = raw_windows
    else:
        batch_items = extract_logmel_batch(
            raw_windows,
            target_sample_rate,
            n_mels=128,
            hop_length=1252,
            target_frames=128,
            normalization=normalization,
        )

    batch_size = 64
    embeddings: list[np.ndarray] = []
    for i in range(0, len(batch_items), batch_size):
        batch = np.stack(batch_items[i : i + batch_size])
        embeddings.append(model.embed(batch))
    all_emb = np.vstack(embeddings)
    proba = pipeline.predict_proba(all_emb)[:, 1]

    return [
        {
            "offset_sec": offset_sec,
            "end_sec": offset_sec + window_size_seconds,
            "confidence": float(conf),
        }
        for offset_sec, conf in zip(offsets, proba.tolist())
    ]


def _resolve_positive_output_path(
    root: Path,
    *,
    label: str,
    clip_name: str,
    source_id: str | None = None,
) -> Path:
    """Resolve the final positive artifact path from its filename."""
    clip_start = parse_recording_timestamp(clip_name)
    date_folder = (
        clip_start.strftime("%Y/%m/%d") if clip_start is not None else "unknown_date"
    )
    if source_id:
        return root / label / source_id / date_folder / clip_name
    return root / label / date_folder / clip_name


def _delete_stale_positive_outputs(
    root: Path,
    *,
    clip_name: str,
    labels: list[str],
    source_id: str | None = None,
) -> None:
    """Remove previously extracted positive artifacts for one row."""
    if not clip_name:
        return
    for label in labels:
        path = _resolve_positive_output_path(
            root,
            label=label,
            clip_name=_with_output_audio_extension(clip_name),
            source_id=source_id,
        )
        _delete_output_artifacts(path)


def _delete_output_artifacts(audio_output_path: Path) -> None:
    """Remove an extracted audio artifact and its sidecar spectrogram."""
    for path in (audio_output_path, _spectrogram_sidecar_path(audio_output_path)):
        if path.exists():
            path.unlink()


def _write_bytes_atomic(output_path: Path, data: bytes) -> None:
    """Atomically write bytes to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(output_path.parent), suffix=output_path.suffix
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_flac_file(audio_segment: np.ndarray, sr: int, output_path: Path) -> None:
    """Write a float32 audio segment as 16-bit PCM FLAC."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Peak normalize
    peak = np.max(np.abs(audio_segment))
    if peak > 0:
        audio_segment = audio_segment / peak
    sf.write(
        str(output_path),
        audio_segment.astype(np.float32),
        sr,
        format="FLAC",
        subtype="PCM_16",
    )


def write_spectrogram_png_file(
    audio_segment: np.ndarray,
    sr: int,
    output_path: Path,
    *,
    hop_length: int = DEFAULT_SPECTROGRAM_HOP_LENGTH,
    dynamic_range_db: float = DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
    width_px: int = DEFAULT_SPECTROGRAM_WIDTH_PX,
    height_px: int = DEFAULT_SPECTROGRAM_HEIGHT_PX,
) -> None:
    """Write a spectrogram PNG sidecar for an extracted audio segment."""
    png_bytes = generate_spectrogram_png(
        audio_segment,
        sr,
        hop_length=hop_length,
        dynamic_range_db=dynamic_range_db,
        width_px=width_px,
        height_px=height_px,
    )
    _write_bytes_atomic(output_path, png_bytes)


def _output_artifacts_complete(output_path: Path) -> bool:
    """Return whether both extracted audio and spectrogram sidecar exist."""
    return output_path.exists() and _spectrogram_sidecar_path(output_path).exists()


def ensure_output_artifacts(
    audio_segment: np.ndarray,
    sr: int,
    output_path: Path,
    *,
    spectrogram_hop_length: int = DEFAULT_SPECTROGRAM_HOP_LENGTH,
    spectrogram_dynamic_range_db: float = DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
    spectrogram_width_px: int = DEFAULT_SPECTROGRAM_WIDTH_PX,
    spectrogram_height_px: int = DEFAULT_SPECTROGRAM_HEIGHT_PX,
) -> bool:
    """Ensure an extracted audio artifact and its sidecar PNG both exist.

    Returns True when the audio file itself was newly written, or False when the
    audio already existed and only the sidecar check/backfill was needed.
    """
    audio_exists = output_path.exists()
    png_path = _spectrogram_sidecar_path(output_path)
    png_exists = png_path.exists()

    if not audio_exists:
        write_flac_file(audio_segment, sr, output_path)
    if not audio_exists or not png_exists:
        write_spectrogram_png_file(
            audio_segment,
            sr,
            png_path,
            hop_length=spectrogram_hop_length,
            dynamic_range_db=spectrogram_dynamic_range_db,
            width_px=spectrogram_width_px,
            height_px=spectrogram_height_px,
        )

    return not audio_exists


def write_wav_file(audio_segment: np.ndarray, sr: int, output_path: Path) -> None:
    """Backward-compatible alias for callers that imported the old helper name."""
    write_flac_file(audio_segment, sr, output_path)


def extract_labeled_samples(
    tsv_path: str | Path,
    audio_folder: str | Path,
    positive_output_path: str | Path,
    negative_output_path: str | Path,
    window_size_seconds: float = 5.0,
    window_diagnostics_path: str | Path | None = None,
    positive_selection_smoothing_window: int = DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
    positive_selection_min_score: float = DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    positive_selection_extend_min_score: float = (
        DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE
    ),
    fallback_pipeline: Any | None = None,
    fallback_model: EmbeddingModel | None = None,
    fallback_target_sample_rate: int = 32000,
    fallback_input_format: str = "spectrogram",
    fallback_feature_config: dict | None = None,
    row_store_path: str | Path | None = None,
    spectrogram_hop_length: int = DEFAULT_SPECTROGRAM_HOP_LENGTH,
    spectrogram_dynamic_range_db: float = DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
    spectrogram_width_px: int = DEFAULT_SPECTROGRAM_WIDTH_PX,
    spectrogram_height_px: int = DEFAULT_SPECTROGRAM_HEIGHT_PX,
) -> dict:
    """Extract labeled audio segments from a detection TSV.

    Reads the TSV, filters to rows with at least one label=1,
    slices audio, and writes WAV files to the appropriate directories.

    Returns a summary dict with counts per label.
    """
    tsv_path = Path(tsv_path)
    audio_folder = Path(audio_folder)
    positive_output_path = Path(positive_output_path)
    negative_output_path = Path(negative_output_path)
    row_store_path = Path(row_store_path) if row_store_path is not None else None
    diagnostics_path = (
        Path(window_diagnostics_path) if window_diagnostics_path is not None else None
    )
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")
    _validate_positive_selection_config(
        positive_selection_smoothing_window,
        positive_selection_min_score,
        positive_selection_extend_min_score,
    )

    fieldnames, all_rows, using_row_store = _read_detection_rows(
        tsv_path,
        row_store_path=row_store_path,
    )
    if using_row_store:
        fieldnames = list(ROW_STORE_FIELDNAMES)
    _ensure_fieldnames(fieldnames, POSITIVE_SELECTION_FIELDNAMES)
    counts = {
        "n_humpback": 0,
        "n_orca": 0,
        "n_ship": 0,
        "n_background": 0,
        "n_skipped": 0,
        "n_positive_selected": 0,
        "n_positive_selection_skipped": 0,
    }
    if not all_rows:
        return counts

    process_rows = [
        row
        for row in all_rows
        if is_positive_row(row)
        or is_negative_row(row)
        or bool(row.get("positive_extract_filename", "").strip())
    ]
    if not process_rows:
        return counts

    # Build audio file index and normalize legacy rows to UTC.
    audio_index = _build_local_audio_index(audio_folder)
    _normalize_local_rows_to_utc(process_rows, audio_index)

    # Group rows by resolved source file.
    by_file: dict[Path, tuple[float, list[dict[str, str]]]] = {}
    for row in process_rows:
        row_start_utc = safe_float(row.get("start_utc"), 0.0)
        resolved = _resolve_local_audio_for_row(row_start_utc, audio_index)
        if resolved is None:
            logger.warning("No source audio resolved for start_utc=%.1f", row_start_utc)
            continue
        file_path, base_epoch, _ = resolved
        entry = by_file.get(file_path)
        if entry is None:
            by_file[file_path] = (base_epoch, [row])
        else:
            entry[1].append(row)

    for source_path, (base_epoch, file_rows) in by_file.items():
        needs_audio = any(
            is_positive_row(row) or is_negative_row(row) for row in file_rows
        )
        audio: np.ndarray | None = None
        sr: int | None = None
        if needs_audio:
            if not source_path.is_file():
                logger.warning("Source audio not found: %s", source_path)
                continue
            audio, sr = decode_audio(source_path)

        stored_records = _load_window_records(
            diagnostics_path,
            filename=source_path.name,
            base_epoch=base_epoch,
        )

        for row in file_rows:
            row_start_utc = safe_float(row.get("start_utc"), 0.0)
            row_end_utc = safe_float(row.get("end_utc"), 0.0)
            row_offset = row_start_utc - base_epoch
            row_end_offset = row_end_utc - base_epoch
            pos_labels = positive_labels_for_row(row)
            neg_labels = negative_labels_for_row(row)
            old_positive_filename = row.get("positive_extract_filename", "").strip()

            if pos_labels:
                selection = (
                    _selection_from_effective_row(row) if using_row_store else None
                )
                if selection is None:
                    selection = (
                        select_positive_window(
                            row_start_utc=row_start_utc,
                            row_end_utc=row_end_utc,
                            window_size_seconds=window_size_seconds,
                            window_records=stored_records,
                            smoothing_window=positive_selection_smoothing_window,
                            min_score=positive_selection_min_score,
                            extend_min_score=positive_selection_extend_min_score,
                            score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                        )
                        if stored_records is not None
                        else None
                    )
                if (selection is None or not selection.offsets) and (
                    not using_row_store
                    and (
                        fallback_pipeline is not None
                        and fallback_model is not None
                        and audio is not None
                        and sr is not None
                    )
                ):
                    row_start_sample = min(int(row_offset * sr), len(audio))
                    row_end_sample = min(int(row_end_offset * sr), len(audio))
                    row_segment = audio[row_start_sample:row_end_sample]
                    fallback_records = _score_segment_windows(
                        row_segment,
                        source_sr=sr,
                        row_start_utc=row_start_utc,
                        pipeline=fallback_pipeline,
                        model=fallback_model,
                        window_size_seconds=window_size_seconds,
                        target_sample_rate=fallback_target_sample_rate,
                        input_format=fallback_input_format,
                        feature_config=fallback_feature_config,
                    )
                    selection = select_positive_window(
                        row_start_utc=row_start_utc,
                        row_end_utc=row_end_utc,
                        window_size_seconds=window_size_seconds,
                        window_records=fallback_records,
                        smoothing_window=positive_selection_smoothing_window,
                        min_score=positive_selection_min_score,
                        extend_min_score=positive_selection_extend_min_score,
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK,
                    )
                if selection is None:
                    selection = PositiveSelectionResult(
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                        decision="skip",
                        offsets=[],
                        raw_scores=[],
                        smoothed_scores=[],
                        start_utc=None,
                        end_utc=None,
                        peak_score=None,
                    )

                if (
                    selection.decision == "positive"
                    and selection.start_utc is not None
                    and selection.end_utc is not None
                    and audio is not None
                    and sr is not None
                ):
                    clip_name = (
                        derive_detection_filename(
                            selection.start_utc, selection.end_utc
                        )
                        or ""
                    )
                    date_folder = _date_folder_from_epoch(selection.start_utc)
                    if old_positive_filename and old_positive_filename != clip_name:
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=old_positive_filename,
                            labels=["humpback", "orca"],
                        )
                    sel_offset = selection.start_utc - base_epoch
                    sel_end_offset = selection.end_utc - base_epoch
                    selected_start = min(int(sel_offset * sr), len(audio))
                    selected_end = min(int(sel_end_offset * sr), len(audio))
                    selected_segment = audio[selected_start:selected_end]
                    if len(selected_segment) == 0:
                        selection = PositiveSelectionResult(
                            score_source=selection.score_source,
                            decision="skip",
                            offsets=selection.offsets,
                            raw_scores=selection.raw_scores,
                            smoothed_scores=selection.smoothed_scores,
                            start_utc=selection.start_utc,
                            end_utc=selection.end_utc,
                            peak_score=selection.peak_score,
                        )
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=clip_name,
                            labels=["humpback", "orca"],
                        )
                        if using_row_store:
                            row["positive_extract_filename"] = ""
                        else:
                            row.update(
                                selection_result_to_row_update(
                                    selection,
                                    positive_extract_filename=None,
                                )
                            )
                        counts["n_positive_selection_skipped"] += 1
                    else:
                        for label_name in pos_labels:
                            out_dir = positive_output_path / label_name / date_folder
                            out_path = out_dir / clip_name
                            if _output_artifacts_complete(out_path):
                                counts["n_skipped"] += 1
                                continue
                            wrote_audio = ensure_output_artifacts(
                                selected_segment,
                                sr,
                                out_path,
                                spectrogram_hop_length=spectrogram_hop_length,
                                spectrogram_dynamic_range_db=spectrogram_dynamic_range_db,
                                spectrogram_width_px=spectrogram_width_px,
                                spectrogram_height_px=spectrogram_height_px,
                            )
                            if wrote_audio:
                                counts[f"n_{label_name}"] += 1
                            else:
                                counts["n_skipped"] += 1
                        if using_row_store:
                            row["positive_extract_filename"] = clip_name
                        else:
                            row.update(
                                selection_result_to_row_update(
                                    selection,
                                    positive_extract_filename=clip_name,
                                )
                            )
                        counts["n_positive_selected"] += 1
                else:
                    if old_positive_filename:
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=old_positive_filename,
                            labels=["humpback", "orca"],
                        )
                    if using_row_store:
                        row["positive_extract_filename"] = ""
                    else:
                        row.update(
                            selection_result_to_row_update(
                                selection,
                                positive_extract_filename=None,
                            )
                        )
                    counts["n_positive_selection_skipped"] += 1
            else:
                if old_positive_filename:
                    _delete_stale_positive_outputs(
                        positive_output_path,
                        clip_name=old_positive_filename,
                        labels=["humpback", "orca"],
                    )
                if using_row_store:
                    row["positive_extract_filename"] = ""
                else:
                    row.update(blank_positive_selection_fields())

            if neg_labels and audio is not None and sr is not None:
                start_sample = min(int(row_offset * sr), len(audio))
                end_sample = min(int(row_end_offset * sr), len(audio))
                segment = audio[start_sample:end_sample]
                if len(segment) == 0:
                    continue
                clip_name = derive_detection_filename(row_start_utc, row_end_utc) or ""
                date_folder = _date_folder_from_epoch(row_start_utc)
                for label_name in neg_labels:
                    out_dir = negative_output_path / label_name / date_folder
                    out_path = out_dir / clip_name
                    if _output_artifacts_complete(out_path):
                        counts["n_skipped"] += 1
                        continue
                    wrote_audio = ensure_output_artifacts(
                        segment,
                        sr,
                        out_path,
                        spectrogram_hop_length=spectrogram_hop_length,
                        spectrogram_dynamic_range_db=spectrogram_dynamic_range_db,
                        spectrogram_width_px=spectrogram_width_px,
                        spectrogram_height_px=spectrogram_height_px,
                    )
                    if wrote_audio:
                        counts[f"n_{label_name}"] += 1
                    else:
                        counts["n_skipped"] += 1

    _write_detection_rows(
        all_rows,
        row_store_path=row_store_path,
    )
    return counts


def _fetch_audio_range(
    provider: ArchiveProvider,
    abs_start_ts: float,
    abs_end_ts: float,
    target_sr: int,
    *,
    expected_samples: int | None = None,
    guard_samples: int = AUDIO_SLICE_GUARD_SAMPLES,
) -> np.ndarray | None:
    """Fetch and decode provider audio covering a time range."""
    fetch_end_ts = abs_end_ts
    if expected_samples is not None:
        fetch_end_ts += guard_samples / target_sr

    timeline = provider.build_timeline(abs_start_ts, fetch_end_ts)
    all_audio: list[np.ndarray] = []
    for segment in timeline:
        try:
            seg_bytes = provider.fetch_segment(segment.key)
            audio = provider.decode_segment(seg_bytes, target_sr)
        except Exception:
            continue
        if len(audio) == 0:
            continue
        decoded_end_ts = segment.start_ts + (len(audio) / target_sr)
        start_ts = max(segment.start_ts, abs_start_ts)
        end_ts = min(decoded_end_ts, fetch_end_ts)
        if end_ts <= start_ts:
            continue
        start_sample = max(0, int(round((start_ts - segment.start_ts) * target_sr)))
        end_sample = min(
            len(audio), int(round((end_ts - segment.start_ts) * target_sr))
        )
        if end_sample <= start_sample:
            continue
        all_audio.append(audio[start_sample:end_sample])
    if not all_audio:
        return None
    combined = np.concatenate(all_audio)
    if expected_samples is not None and len(combined) > expected_samples:
        combined = combined[:expected_samples]
    return combined


def _format_compact_ts(dt: datetime) -> str:
    """Format datetime as YYYYMMDDTHHMMSSz (compact, no microseconds)."""
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_compact_range_filename(
    filename: str,
) -> tuple[datetime, datetime] | None:
    """Parse compact UTC range filename into (start, end) datetimes."""
    from humpback.classifier.detection_rows import strip_known_audio_extension

    base = strip_known_audio_extension(filename)
    parts = base.split("_")
    if len(parts) != 2:
        return None
    try:
        start = datetime.strptime(parts[0], _COMPACT_TS_FORMAT).replace(
            tzinfo=timezone.utc
        )
        end = datetime.strptime(parts[1], _COMPACT_TS_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    if end <= start:
        return None
    return start, end


def parse_hydrophone_clip_range(
    filename: str,
) -> tuple[datetime, datetime] | None:
    """Parse a compact UTC hydrophone clip filename into absolute bounds."""
    return _parse_compact_range_filename(filename)


def fetch_hydrophone_audio_range(
    provider: ArchiveProvider,
    abs_start_ts: float,
    abs_end_ts: float,
    target_sr: int,
) -> np.ndarray | None:
    """Fetch a hydrophone clip by absolute UTC range with exact-length trimming."""
    duration_sec = abs_end_ts - abs_start_ts
    if duration_sec <= 0:
        return None
    return _fetch_audio_range(
        provider,
        abs_start_ts,
        abs_end_ts,
        target_sr,
        expected_samples=expected_audio_samples(duration_sec, target_sr),
    )


def extract_hydrophone_labeled_samples(
    tsv_path: str | Path,
    provider: ArchiveProvider,
    positive_output_path: str | Path,
    negative_output_path: str | Path,
    target_sample_rate: int = 32000,
    window_size_seconds: float = 5.0,
    stream_start_timestamp: float | None = None,
    stream_end_timestamp: float | None = None,
    window_diagnostics_path: str | Path | None = None,
    positive_selection_smoothing_window: int = DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
    positive_selection_min_score: float = DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    positive_selection_extend_min_score: float = (
        DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE
    ),
    fallback_pipeline: Any | None = None,
    fallback_model: EmbeddingModel | None = None,
    fallback_input_format: str = "spectrogram",
    fallback_feature_config: dict | None = None,
    row_store_path: str | Path | None = None,
    spectrogram_hop_length: int = DEFAULT_SPECTROGRAM_HOP_LENGTH,
    spectrogram_dynamic_range_db: float = DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
    spectrogram_width_px: int = DEFAULT_SPECTROGRAM_WIDTH_PX,
    spectrogram_height_px: int = DEFAULT_SPECTROGRAM_HEIGHT_PX,
) -> dict:
    """Extract labeled audio segments from a hydrophone detection TSV.

    Similar to extract_labeled_samples but reconstructs audio from HLS segments
    instead of reading from a local audio folder.

    Returns a summary dict with counts per label.
    """
    tsv_path = Path(tsv_path)
    positive_output_path = Path(positive_output_path)
    negative_output_path = Path(negative_output_path)
    row_store_path = Path(row_store_path) if row_store_path is not None else None
    diagnostics_path = (
        Path(window_diagnostics_path) if window_diagnostics_path is not None else None
    )
    _validate_positive_selection_config(
        positive_selection_smoothing_window,
        positive_selection_min_score,
        positive_selection_extend_min_score,
    )

    fieldnames, all_rows, using_row_store = _read_detection_rows(
        tsv_path,
        row_store_path=row_store_path,
    )
    if using_row_store:
        fieldnames = list(ROW_STORE_FIELDNAMES)
    _ensure_fieldnames(fieldnames, POSITIVE_SELECTION_FIELDNAMES)
    counts = {
        "n_humpback": 0,
        "n_orca": 0,
        "n_ship": 0,
        "n_background": 0,
        "n_skipped": 0,
        "n_positive_selected": 0,
        "n_positive_selection_skipped": 0,
    }
    if not all_rows:
        return counts

    process_rows = [
        row
        for row in all_rows
        if is_positive_row(row)
        or is_negative_row(row)
        or bool(row.get("positive_extract_filename", "").strip())
    ]
    if not process_rows:
        return counts

    # Normalize legacy rows to UTC.
    _normalize_hydrophone_rows_to_utc(process_rows, window_size_seconds)

    # Load diagnostics once (keyed by filename for legacy compat).
    # For new-schema rows the filename column is absent; diagnostics
    # are loaded per-file when needed below.
    _diag_cache: dict[str, list[dict[str, float]] | None] = {}

    use_stream_resolver = (
        stream_start_timestamp is not None and stream_end_timestamp is not None
    )
    stream_timeline = None
    stream_start_ts_value: float | None = None
    stream_end_ts_value: float | None = None

    if use_stream_resolver:
        assert stream_start_timestamp is not None
        assert stream_end_timestamp is not None

        stream_start_ts_value = float(stream_start_timestamp)
        stream_end_ts_value = float(stream_end_timestamp)

        try:
            stream_timeline = build_stream_timeline(
                provider=provider,
                stream_start_ts=stream_start_ts_value,
                stream_end_ts=stream_end_ts_value,
            )
        except Exception as exc:
            logger.warning(
                "Hydrophone extraction timeline unavailable for %s [%.1f, %.1f]: %s",
                provider.source_id,
                stream_start_ts_value,
                stream_end_ts_value,
                exc,
            )
            stream_timeline = []

    for row in process_rows:
        row_start_utc = safe_float(row.get("start_utc"), 0.0)
        row_end_utc = safe_float(row.get("end_utc"), 0.0)
        pos_labels = positive_labels_for_row(row)
        old_positive_filename = row.get("positive_extract_filename", "").strip()

        # Load stored diagnostics (cached per filename when available).
        diag_filename = row.get("filename", "").strip() or None
        if diag_filename is not None:
            if diag_filename not in _diag_cache:
                recording_ts = parse_recording_timestamp(diag_filename)
                base_epoch = recording_ts.timestamp() if recording_ts else 0.0
                _diag_cache[diag_filename] = _load_window_records(
                    diagnostics_path,
                    filename=diag_filename,
                    base_epoch=base_epoch,
                )
            stored_records = _diag_cache[diag_filename]
        else:
            stored_records = None

        if pos_labels:
            selection = _selection_from_effective_row(row) if using_row_store else None
            if selection is None:
                selection = (
                    select_positive_window(
                        row_start_utc=row_start_utc,
                        row_end_utc=row_end_utc,
                        window_size_seconds=window_size_seconds,
                        window_records=stored_records,
                        smoothing_window=positive_selection_smoothing_window,
                        min_score=positive_selection_min_score,
                        extend_min_score=positive_selection_extend_min_score,
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                    )
                    if stored_records is not None
                    else None
                )
            if (selection is None or not selection.offsets) and (
                not using_row_store
                and (fallback_pipeline is not None and fallback_model is not None)
            ):
                row_duration = max(0.0, row_end_utc - row_start_utc)
                fallback_segment: np.ndarray | None = None
                if use_stream_resolver:
                    if stream_timeline:
                        assert stream_start_ts_value is not None
                        assert stream_end_ts_value is not None
                        try:
                            fallback_segment = resolve_audio_slice(
                                provider=provider,
                                stream_start_ts=stream_start_ts_value,
                                stream_end_ts=stream_end_ts_value,
                                start_utc=row_start_utc,
                                duration_sec=row_duration,
                                target_sr=target_sample_rate,
                                timeline=stream_timeline,
                            )
                        except Exception:
                            fallback_segment = None
                else:
                    fallback_segment = fetch_hydrophone_audio_range(
                        provider,
                        row_start_utc,
                        row_end_utc,
                        target_sample_rate,
                    )
                if fallback_segment is not None and len(fallback_segment) > 0:
                    fallback_records = _score_segment_windows(
                        fallback_segment,
                        source_sr=target_sample_rate,
                        row_start_utc=row_start_utc,
                        pipeline=fallback_pipeline,
                        model=fallback_model,
                        window_size_seconds=window_size_seconds,
                        target_sample_rate=target_sample_rate,
                        input_format=fallback_input_format,
                        feature_config=fallback_feature_config,
                    )
                    selection = select_positive_window(
                        row_start_utc=row_start_utc,
                        row_end_utc=row_end_utc,
                        window_size_seconds=window_size_seconds,
                        window_records=fallback_records,
                        smoothing_window=positive_selection_smoothing_window,
                        min_score=positive_selection_min_score,
                        extend_min_score=positive_selection_extend_min_score,
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK,
                    )
            if selection is None:
                selection = PositiveSelectionResult(
                    score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                    decision="skip",
                    offsets=[],
                    raw_scores=[],
                    smoothed_scores=[],
                    start_utc=None,
                    end_utc=None,
                    peak_score=None,
                )

            positive_written = False
            if (
                selection.decision == "positive"
                and selection.start_utc is not None
                and selection.end_utc is not None
            ):
                clip_name = (
                    derive_detection_filename(selection.start_utc, selection.end_utc)
                    or ""
                )
                if clip_name:
                    if old_positive_filename and old_positive_filename != clip_name:
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=old_positive_filename,
                            labels=["humpback", "orca"],
                            source_id=provider.source_id,
                        )
                    duration = selection.end_utc - selection.start_utc
                    segment: np.ndarray | None = None
                    if use_stream_resolver:
                        if stream_timeline:
                            assert stream_start_ts_value is not None
                            assert stream_end_ts_value is not None
                            try:
                                segment = resolve_audio_slice(
                                    provider=provider,
                                    stream_start_ts=stream_start_ts_value,
                                    stream_end_ts=stream_end_ts_value,
                                    start_utc=selection.start_utc,
                                    duration_sec=duration,
                                    target_sr=target_sample_rate,
                                    timeline=stream_timeline,
                                )
                            except Exception:
                                segment = None
                    else:
                        segment = fetch_hydrophone_audio_range(
                            provider,
                            selection.start_utc,
                            selection.end_utc,
                            target_sample_rate,
                        )

                    if segment is None or len(segment) == 0:
                        selection = PositiveSelectionResult(
                            score_source=selection.score_source,
                            decision="skip",
                            offsets=selection.offsets,
                            raw_scores=selection.raw_scores,
                            smoothed_scores=selection.smoothed_scores,
                            start_utc=selection.start_utc,
                            end_utc=selection.end_utc,
                            peak_score=selection.peak_score,
                        )
                    else:
                        for label_name in pos_labels:
                            out_path = _resolve_positive_output_path(
                                positive_output_path,
                                label=label_name,
                                clip_name=clip_name,
                                source_id=provider.source_id,
                            )
                            if _output_artifacts_complete(out_path):
                                counts["n_skipped"] += 1
                                continue
                            wrote_audio = ensure_output_artifacts(
                                segment,
                                target_sample_rate,
                                out_path,
                                spectrogram_hop_length=spectrogram_hop_length,
                                spectrogram_dynamic_range_db=spectrogram_dynamic_range_db,
                                spectrogram_width_px=spectrogram_width_px,
                                spectrogram_height_px=spectrogram_height_px,
                            )
                            if wrote_audio:
                                counts[f"n_{label_name}"] += 1
                            else:
                                counts["n_skipped"] += 1
                        if using_row_store:
                            row["positive_extract_filename"] = clip_name
                        else:
                            row.update(
                                selection_result_to_row_update(
                                    selection,
                                    positive_extract_filename=clip_name,
                                )
                            )
                        counts["n_positive_selected"] += 1
                        positive_written = True
            if not positive_written:
                if old_positive_filename:
                    _delete_stale_positive_outputs(
                        positive_output_path,
                        clip_name=old_positive_filename,
                        labels=["humpback", "orca"],
                        source_id=provider.source_id,
                    )
                if using_row_store:
                    row["positive_extract_filename"] = ""
                else:
                    row.update(
                        selection_result_to_row_update(
                            selection,
                            positive_extract_filename=None,
                        )
                    )
                counts["n_positive_selection_skipped"] += 1
        else:
            if old_positive_filename:
                _delete_stale_positive_outputs(
                    positive_output_path,
                    clip_name=old_positive_filename,
                    labels=["humpback", "orca"],
                    source_id=provider.source_id,
                )
            if using_row_store:
                row["positive_extract_filename"] = ""
            else:
                row.update(blank_positive_selection_fields())

        if row_end_utc <= row_start_utc:
            counts["n_skipped"] += 1
            continue

        # Output filename and folder from UTC
        det_filename = derive_detection_filename(row_start_utc, row_end_utc)
        if det_filename is None:
            counts["n_skipped"] += 1
            continue
        clip_name = _with_output_audio_extension(det_filename)
        date_folder = _date_folder_from_epoch(row_start_utc)

        # Route to label-specific folders (species/category before hydrophone_id)
        labels_to_write: list[tuple[Path, str]] = []
        if row.get("ship", "").strip() == "1":
            out_dir = negative_output_path / "ship" / provider.source_id / date_folder
            labels_to_write.append((out_dir, "ship"))
        if row.get("background", "").strip() == "1":
            out_dir = (
                negative_output_path / "background" / provider.source_id / date_folder
            )
            labels_to_write.append((out_dir, "background"))

        pending_writes: list[tuple[Path, str]] = []
        for out_dir, label_name in labels_to_write:
            out_path = out_dir / clip_name
            if _output_artifacts_complete(out_path):
                counts["n_skipped"] += 1
                continue
            pending_writes.append((out_path, label_name))

        if not pending_writes:
            continue

        duration = row_end_utc - row_start_utc
        segment = None

        if use_stream_resolver:
            if not stream_timeline:
                counts["n_skipped"] += len(pending_writes)
                continue
            assert stream_start_ts_value is not None
            assert stream_end_ts_value is not None
            try:
                segment = resolve_audio_slice(
                    provider=provider,
                    stream_start_ts=stream_start_ts_value,
                    stream_end_ts=stream_end_ts_value,
                    start_utc=row_start_utc,
                    duration_sec=duration,
                    target_sr=target_sample_rate,
                    timeline=stream_timeline,
                )
            except Exception as exc:
                logger.warning(
                    "No hydrophone audio for start_utc=%.1f (%.1f-%.1f): %s",
                    row_start_utc,
                    row_start_utc,
                    row_end_utc,
                    exc,
                )
        else:
            combined = fetch_hydrophone_audio_range(
                provider,
                row_start_utc,
                row_end_utc,
                target_sample_rate,
            )
            if combined is not None:
                segment = combined

        if segment is None or len(segment) == 0:
            counts["n_skipped"] += len(pending_writes)
            continue

        for out_path, label_name in pending_writes:
            wrote_audio = ensure_output_artifacts(
                segment,
                target_sample_rate,
                out_path,
                spectrogram_hop_length=spectrogram_hop_length,
                spectrogram_dynamic_range_db=spectrogram_dynamic_range_db,
                spectrogram_width_px=spectrogram_width_px,
                spectrogram_height_px=spectrogram_height_px,
            )
            if wrote_audio:
                counts[f"n_{label_name}"] += 1
            else:
                counts["n_skipped"] += 1

    _write_detection_rows(
        all_rows,
        row_store_path=row_store_path,
    )
    return counts
