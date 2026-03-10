"""Extract labeled audio samples from detection TSV files."""

import csv
import logging
import math
import re
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from humpback.processing.audio_io import decode_audio

logger = logging.getLogger(__name__)

# Regex patterns for parsing timestamps from filenames
# e.g. "20250115T143022Z_..." or "20250115T143022.123456Z_..."
_TS_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(?:\.(\d+))?Z")
_COMPACT_TS_FORMAT = "%Y%m%dT%H%M%SZ"


def parse_recording_timestamp(filename: str) -> datetime | None:
    """Extract a recording start timestamp from a filename.

    Looks for ISO-like pattern YYYYMMDDTHHMMSSZ or YYYYMMDDTHHMMSS.ffffffZ.
    Returns a timezone-aware UTC datetime, or None if no pattern found.
    """
    m = _TS_PATTERN.search(filename)
    if m is None:
        return None
    year, month, day, hour, minute, second = (int(g) for g in m.groups()[:6])
    frac_str = m.group(7)
    microsecond = 0
    if frac_str:
        # Pad or truncate to 6 digits
        frac_str = frac_str[:6].ljust(6, "0")
        microsecond = int(frac_str)
    return datetime(
        year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc
    )


def _format_ts(dt: datetime) -> str:
    """Format datetime as YYYYMMDDTHHMMss.ffffffZ."""
    return dt.strftime("%Y%m%dT%H%M%S.%fZ")


def _date_folder(dt: datetime) -> str:
    """Return YYYY/MM/dd path component."""
    return dt.strftime("%Y/%m/%d")


def write_wav_file(audio_segment: np.ndarray, sr: int, output_path: Path) -> None:
    """Write a float32 audio segment as 16-bit PCM WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Peak normalize
    peak = np.max(np.abs(audio_segment))
    if peak > 0:
        audio_segment = audio_segment / peak
    pcm = (audio_segment * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def extract_labeled_samples(
    tsv_path: str | Path,
    audio_folder: str | Path,
    positive_output_path: str | Path,
    negative_output_path: str | Path,
    window_size_seconds: float = 5.0,
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
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")

    # Read TSV and filter to labeled rows
    labeled_rows: list[dict] = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            humpback = row.get("humpback", "").strip()
            ship = row.get("ship", "").strip()
            background = row.get("background", "").strip()
            if humpback == "1" or ship == "1" or background == "1":
                labeled_rows.append(row)

    if not labeled_rows:
        return {"n_humpback": 0, "n_ship": 0, "n_background": 0, "n_skipped": 0}

    # Group by source filename
    by_file: dict[str, list[dict]] = {}
    for row in labeled_rows:
        fn = row.get("filename", "")
        by_file.setdefault(fn, []).append(row)

    counts = {"n_humpback": 0, "n_ship": 0, "n_background": 0, "n_skipped": 0}

    for source_filename, rows in by_file.items():
        source_path = audio_folder / source_filename
        if not source_path.is_file():
            logger.warning("Source audio not found: %s", source_path)
            continue

        # Decode once per source file
        audio, sr = decode_audio(source_path)
        recording_ts = parse_recording_timestamp(source_filename)

        for row in rows:
            start_sec = float(row.get("start_sec", 0))
            end_sec = float(row.get("end_sec", 0))

            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            start_sample = min(start_sample, len(audio))
            end_sample = min(end_sample, len(audio))
            segment = audio[start_sample:end_sample]

            if len(segment) == 0:
                continue

            # Build filename
            if recording_ts:
                abs_start = recording_ts + timedelta(seconds=start_sec)
                abs_end = recording_ts + timedelta(seconds=end_sec)
                wav_name = f"{_format_ts(abs_start)}_{_format_ts(abs_end)}.wav"
                date_folder = _date_folder(abs_start)
            else:
                stem = Path(source_filename).stem
                wav_name = f"{stem}_{start_sec}_{end_sec}.wav"
                date_folder = "unknown_date"

            # Route to label-specific folders
            labels_to_write: list[tuple[Path, str]] = []
            if row.get("humpback", "").strip() == "1":
                out_dir = positive_output_path / "humpback" / date_folder
                labels_to_write.append((out_dir, "humpback"))
            if row.get("ship", "").strip() == "1":
                out_dir = negative_output_path / "ship" / date_folder
                labels_to_write.append((out_dir, "ship"))
            if row.get("background", "").strip() == "1":
                out_dir = negative_output_path / "background" / date_folder
                labels_to_write.append((out_dir, "background"))

            for out_dir, label_name in labels_to_write:
                out_path = out_dir / wav_name
                if out_path.exists():
                    counts["n_skipped"] += 1
                    continue
                write_wav_file(segment, sr, out_path)
                counts[f"n_{label_name}"] += 1

    return counts


def _fetch_audio_range(
    client,
    hydrophone_id: str,
    abs_start_ts: float,
    abs_end_ts: float,
    target_sr: int,
) -> np.ndarray | None:
    """Fetch and decode HLS segments covering a time range, return audio array."""
    from humpback.classifier.s3_stream import decode_ts_bytes

    folders = client.list_hls_folders(hydrophone_id, abs_start_ts, abs_end_ts)
    all_audio: list[np.ndarray] = []
    for folder_ts in folders:
        segs = client.list_segments(hydrophone_id, folder_ts)
        for seg_key in segs:
            try:
                seg_bytes = client.fetch_segment(seg_key)
                audio = decode_ts_bytes(seg_bytes, target_sr)
                all_audio.append(audio)
            except Exception:
                continue
    if not all_audio:
        return None
    return np.concatenate(all_audio)


def _format_compact_ts(dt: datetime) -> str:
    """Format datetime as YYYYMMDDTHHMMSSz (compact, no microseconds)."""
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_compact_range_filename(
    filename: str,
) -> tuple[datetime, datetime] | None:
    """Parse compact UTC range filename into (start, end) datetimes."""
    base = filename[:-4] if filename.endswith(".wav") else filename
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


def _snap_range_for_window(
    start_sec: float,
    end_sec: float,
    window_size_seconds: float,
) -> tuple[float, float]:
    """Snap bounds outward to window-size multiples."""
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")
    if end_sec <= start_sec:
        return start_sec, end_sec
    start_sec = math.floor(start_sec / window_size_seconds) * window_size_seconds
    end_sec = math.ceil(end_sec / window_size_seconds) * window_size_seconds
    if end_sec <= start_sec:
        end_sec = start_sec + window_size_seconds
    return float(start_sec), float(end_sec)


def extract_hydrophone_labeled_samples(
    tsv_path: str | Path,
    hydrophone_id: str,
    positive_output_path: str | Path,
    negative_output_path: str | Path,
    client,
    target_sample_rate: int = 32000,
    window_size_seconds: float = 5.0,
    stream_start_timestamp: float | None = None,
    stream_end_timestamp: float | None = None,
) -> dict:
    """Extract labeled audio segments from a hydrophone detection TSV.

    Similar to extract_labeled_samples but reconstructs audio from HLS segments
    instead of reading from a local audio folder.

    Returns a summary dict with counts per label.
    """
    tsv_path = Path(tsv_path)
    positive_output_path = Path(positive_output_path)
    negative_output_path = Path(negative_output_path)

    # Read TSV and filter to labeled rows
    labeled_rows: list[dict] = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            humpback = row.get("humpback", "").strip()
            ship = row.get("ship", "").strip()
            background = row.get("background", "").strip()
            if humpback == "1" or ship == "1" or background == "1":
                labeled_rows.append(row)

    if not labeled_rows:
        return {"n_humpback": 0, "n_ship": 0, "n_background": 0, "n_skipped": 0}

    # Group by source filename
    by_file: dict[str, list[dict]] = {}
    for row in labeled_rows:
        fn = row.get("filename", "")
        by_file.setdefault(fn, []).append(row)

    counts = {"n_humpback": 0, "n_ship": 0, "n_background": 0, "n_skipped": 0}
    use_stream_resolver = (
        stream_start_timestamp is not None and stream_end_timestamp is not None
    )
    stream_timeline = None
    processing_start_ts: float | None = None

    if use_stream_resolver:
        from humpback.classifier.s3_stream import (
            build_hydrophone_stream_timeline,
            resolve_hydrophone_audio_slice,
        )

        try:
            stream_timeline = build_hydrophone_stream_timeline(
                client=client,
                hydrophone_id=hydrophone_id,
                stream_start_ts=float(stream_start_timestamp),
                stream_end_ts=float(stream_end_timestamp),
            )
            processing_start_ts = max(
                float(stream_start_timestamp), stream_timeline[0].start_ts
            )
        except Exception as exc:
            logger.warning(
                "Hydrophone extraction timeline unavailable for %s [%.1f, %.1f]: %s",
                hydrophone_id,
                float(stream_start_timestamp),
                float(stream_end_timestamp),
                exc,
            )
            stream_timeline = []

    for source_filename, rows in by_file.items():
        recording_ts = parse_recording_timestamp(source_filename)
        if recording_ts is None:
            logger.warning("Cannot parse timestamp from %s, skipping", source_filename)
            continue

        for row in rows:
            start_sec = float(row.get("start_sec", 0))
            end_sec = float(row.get("end_sec", 0))
            detection_filename = row.get("detection_filename", "").strip() or None
            extract_filename = row.get("extract_filename", "").strip() or None
            parsed_detection_range = (
                _parse_compact_range_filename(detection_filename)
                if detection_filename
                else None
            )
            parsed_extract_range = (
                _parse_compact_range_filename(extract_filename)
                if extract_filename
                else None
            )

            # Resolve bounds with precedence:
            # detection_filename -> extract_filename -> snapped start/end fallback.
            if parsed_detection_range is not None:
                abs_start, abs_end = parsed_detection_range
                start_sec = (abs_start - recording_ts).total_seconds()
                end_sec = (abs_end - recording_ts).total_seconds()
            elif parsed_extract_range is not None:
                abs_start, abs_end = parsed_extract_range
                start_sec = (abs_start - recording_ts).total_seconds()
                end_sec = (abs_end - recording_ts).total_seconds()
                detection_filename = (
                    f"{_format_compact_ts(abs_start)}_{_format_compact_ts(abs_end)}.wav"
                )
            else:
                start_sec, end_sec = _snap_range_for_window(
                    start_sec, end_sec, window_size_seconds
                )
                abs_start = recording_ts + timedelta(seconds=start_sec)
                abs_end = recording_ts + timedelta(seconds=end_sec)
                detection_filename = (
                    f"{_format_compact_ts(abs_start)}_{_format_compact_ts(abs_end)}.wav"
                )

            if abs_end <= abs_start:
                counts["n_skipped"] += 1
                continue

            abs_start_ts = abs_start.timestamp()
            abs_end_ts = abs_end.timestamp()

            # Output filename and folder
            assert detection_filename is not None
            wav_name = detection_filename
            date_folder = _date_folder(abs_start)

            # Route to label-specific folders
            labels_to_write: list[tuple[Path, str]] = []
            if row.get("humpback", "").strip() == "1":
                out_dir = (
                    positive_output_path / hydrophone_id / "humpback" / date_folder
                )
                labels_to_write.append((out_dir, "humpback"))
            if row.get("ship", "").strip() == "1":
                out_dir = negative_output_path / hydrophone_id / "ship" / date_folder
                labels_to_write.append((out_dir, "ship"))
            if row.get("background", "").strip() == "1":
                out_dir = (
                    negative_output_path / hydrophone_id / "background" / date_folder
                )
                labels_to_write.append((out_dir, "background"))

            pending_writes: list[tuple[Path, str]] = []
            for out_dir, label_name in labels_to_write:
                out_path = out_dir / wav_name
                if out_path.exists():
                    counts["n_skipped"] += 1
                    continue
                pending_writes.append((out_path, label_name))

            if not pending_writes:
                continue

            duration = (abs_end - abs_start).total_seconds()
            segment: np.ndarray | None = None

            if use_stream_resolver:
                if not stream_timeline:
                    counts["n_skipped"] += len(pending_writes)
                    continue
                try:
                    segment = resolve_hydrophone_audio_slice(
                        client=client,
                        hydrophone_id=hydrophone_id,
                        stream_start_ts=float(stream_start_timestamp),
                        stream_end_ts=float(stream_end_timestamp),
                        filename=source_filename,
                        row_start_sec=start_sec,
                        duration_sec=duration,
                        target_sr=target_sample_rate,
                        legacy_anchor_start_ts=float(stream_start_timestamp),
                        timeline=stream_timeline,
                        processing_start_ts=processing_start_ts,
                    )
                except Exception as exc:
                    logger.warning(
                        "No hydrophone audio for %s (%.1f-%.1f): %s",
                        source_filename,
                        start_sec,
                        end_sec,
                        exc,
                    )
            else:
                # Backward-compatible fallback for direct callers that do not
                # provide stream bounds.
                combined = _fetch_audio_range(
                    client, hydrophone_id, abs_start_ts, abs_end_ts, target_sample_rate
                )
                if combined is not None:
                    end_sample = min(int(duration * target_sample_rate), len(combined))
                    segment = combined[:end_sample]

            if segment is None or len(segment) == 0:
                counts["n_skipped"] += len(pending_writes)
                continue

            for out_path, label_name in pending_writes:
                write_wav_file(segment, target_sample_rate, out_path)
                counts[f"n_{label_name}"] += 1

    return counts
