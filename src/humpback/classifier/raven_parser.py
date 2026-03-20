"""Parse Raven selection table TSV files and pair with audio recordings."""

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}

# Raven TSV column names
COL_SELECTION = "Selection"
COL_BEGIN_TIME = "Begin Time (s)"
COL_END_TIME = "End Time (s)"
COL_LOW_FREQ = "Low Freq (Hz)"
COL_HIGH_FREQ = "High Freq (Hz)"
COL_CALL_TYPE = "Call Type"

REQUIRED_COLUMNS = {COL_SELECTION, COL_BEGIN_TIME, COL_END_TIME, COL_CALL_TYPE}

# Typo/variant corrections: lowercase key -> canonical value
_CALL_TYPE_CORRECTIONS: dict[str, str] = {
    "chrip": "Chirp",
    "whuo": "Whup",
    "piccalo": "Piccolo",
}

# Suffix pattern for Raven selection table filenames
_RAVEN_SUFFIX_RE = re.compile(r"\.Table\.\d+\.selections$", re.IGNORECASE)


@dataclass
class RavenAnnotation:
    """A single annotation row from a Raven selection table."""

    selection: int
    begin_time: float  # seconds
    end_time: float  # seconds
    low_freq: float  # Hz
    high_freq: float  # Hz
    call_type: str  # normalized label


@dataclass
class AnnotatedRecording:
    """A paired audio recording and its annotations."""

    audio_path: Path
    annotation_path: Path
    annotations: list[RavenAnnotation]


def normalize_call_type(raw: str) -> str:
    """Normalize a call type label: strip whitespace, fix typos, title-case words."""
    cleaned = raw.strip()
    if not cleaned:
        return cleaned

    # Check correction map (case-insensitive)
    lower = cleaned.lower()
    if lower in _CALL_TYPE_CORRECTIONS:
        return _CALL_TYPE_CORRECTIONS[lower]

    # Title-case each word (preserves multi-word labels like "Descending moan" -> "Descending Moan")
    return cleaned.title()


def parse_raven_tsv(path: Path) -> list[RavenAnnotation]:
    """Parse a Raven selection table TSV file into annotations.

    Handles BOM prefixes and \\r\\n line endings.
    """
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")

    if reader.fieldnames is None:
        raise ValueError(f"No header row found in {path}")

    # Validate required columns present
    available = set(reader.fieldnames)
    missing = REQUIRED_COLUMNS - available
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    annotations: list[RavenAnnotation] = []
    for row_num, row in enumerate(reader, start=2):
        try:
            call_type_raw = row.get(COL_CALL_TYPE, "")
            if not call_type_raw or not call_type_raw.strip():
                logger.warning(
                    "Skipping row %d in %s: empty call type", row_num, path.name
                )
                continue

            annotations.append(
                RavenAnnotation(
                    selection=int(row[COL_SELECTION]),
                    begin_time=float(row[COL_BEGIN_TIME]),
                    end_time=float(row[COL_END_TIME]),
                    low_freq=float(row.get(COL_LOW_FREQ, "0") or "0"),
                    high_freq=float(row.get(COL_HIGH_FREQ, "0") or "0"),
                    call_type=normalize_call_type(call_type_raw),
                )
            )
        except (ValueError, KeyError) as e:
            logger.warning("Skipping malformed row %d in %s: %s", row_num, path.name, e)
            continue

    return annotations


def _annotation_stem(annotation_path: Path) -> str:
    """Extract the recording stem from a Raven annotation filename.

    Strips the '.Table.N.selections.txt' suffix to get the recording stem.
    Example: '211026-133018-OS-humpback-47min-clip.Table.1.selections.txt'
             -> '211026-133018-OS-humpback-47min-clip'
    """
    stem = annotation_path.stem  # removes .txt
    # Strip .Table.N.selections suffix
    stem = _RAVEN_SUFFIX_RE.sub("", stem)
    return stem


def pair_annotations_with_recordings(
    annotation_dir: Path,
    audio_dir: Path,
) -> list[AnnotatedRecording]:
    """Match Raven annotation files with audio recordings by filename stem.

    Returns paired recordings sorted by audio filename.
    Raises ValueError if no pairs are found.
    """
    # Index annotation files by stem
    annotation_files: dict[str, Path] = {}
    for p in sorted(annotation_dir.iterdir()):
        if p.suffix.lower() == ".txt" and p.is_file():
            stem = _annotation_stem(p)
            annotation_files[stem] = p

    if not annotation_files:
        raise ValueError(f"No annotation .txt files found in {annotation_dir}")

    # Index audio files by stem
    audio_files: dict[str, Path] = {}
    for p in sorted(audio_dir.iterdir()):
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file():
            audio_files[p.stem] = p

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    # Pair by matching stems
    pairs: list[AnnotatedRecording] = []
    unmatched_annotations: list[str] = []

    for stem, ann_path in sorted(annotation_files.items()):
        if stem in audio_files:
            annotations = parse_raven_tsv(ann_path)
            pairs.append(
                AnnotatedRecording(
                    audio_path=audio_files[stem],
                    annotation_path=ann_path,
                    annotations=annotations,
                )
            )
        else:
            unmatched_annotations.append(stem)

    if unmatched_annotations:
        logger.warning(
            "Annotation files without matching audio (%d): %s",
            len(unmatched_annotations),
            unmatched_annotations[:5],
        )

    unmatched_audio = set(audio_files.keys()) - set(annotation_files.keys())
    if unmatched_audio:
        logger.warning(
            "Audio files without annotations (%d): %s",
            len(unmatched_audio),
            sorted(unmatched_audio)[:5],
        )

    if not pairs:
        raise ValueError(
            f"No annotation-audio pairs found. "
            f"Annotation stems: {sorted(annotation_files.keys())[:5]}, "
            f"Audio stems: {sorted(audio_files.keys())[:5]}"
        )

    return pairs
