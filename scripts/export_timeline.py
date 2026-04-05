"""Export a detection job timeline as a self-contained static bundle.

Usage:
    uv run scripts/export_timeline.py --job-id <uuid> --output-dir /path/to/export
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _progress(stage: str, current: int, total: int) -> None:
    sys.stderr.write(f"\r  {stage}: {current}/{total}")
    if current == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


async def _run(job_id: str, output_dir: Path) -> None:
    from humpback.config import Settings
    from humpback.database import create_engine, create_session_factory
    from humpback.services.timeline_export import ExportError, export_timeline

    settings = Settings()
    engine = create_engine(settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        try:
            result = await export_timeline(
                job_id=job_id,
                output_dir=output_dir,
                db=session,
                settings=settings,
                progress_callback=_progress,
            )
        except ExportError as exc:
            sys.stderr.write(f"\nError: {exc}\n")
            sys.exit(1)

    summary = {
        "job_id": result.job_id,
        "output_path": result.output_path,
        "tile_count": result.tile_count,
        "audio_chunk_count": result.audio_chunk_count,
        "manifest_size_bytes": result.manifest_size_bytes,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a detection job timeline as a static bundle"
    )
    parser.add_argument("--job-id", required=True, help="Detection job UUID")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Parent directory for the export",
    )
    args = parser.parse_args()

    sys.stderr.write(f"Exporting timeline for job {args.job_id}...\n")
    asyncio.run(_run(args.job_id, args.output_dir))


if __name__ == "__main__":
    main()
