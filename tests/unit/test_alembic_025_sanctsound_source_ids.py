import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine, create_session_factory
from humpback.models.classifier import DetectionJob


def _db_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


async def _seed_detection_jobs(db_path: Path) -> None:
    engine = create_engine(_db_url(db_path))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        session.add_all(
            [
                DetectionJob(
                    classifier_model_id="model-ci",
                    hydrophone_id="sanctsound_ci01",
                    hydrophone_name="NOAA SanctSound (Channel Islands)",
                    detection_mode="windowed",
                ),
                DetectionJob(
                    classifier_model_id="model-oc",
                    hydrophone_id="sanctsound_oc01",
                    hydrophone_name="NOAA SanctSound (Olympic Coast)",
                    detection_mode="windowed",
                ),
                DetectionJob(
                    classifier_model_id="model-other",
                    hydrophone_id="noaa_glacier_bay",
                    hydrophone_name="NOAA Glacier Bay (Bartlett Cove)",
                    detection_mode="windowed",
                ),
            ]
        )
        await session.commit()

    await engine.dispose()


def test_alembic_025_normalizes_legacy_sanctsound_job_ids(tmp_path):
    db_path = tmp_path / "migration.db"
    asyncio.run(_seed_detection_jobs(db_path))

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('024')")
        conn.commit()
    finally:
        conn.close()

    repo_root = Path(__file__).resolve().parents[2]
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "alembic"))
    config.set_main_option("sqlalchemy.url", _db_url(db_path))
    command.upgrade(config, "025")

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT hydrophone_id, hydrophone_name
            FROM detection_jobs
            ORDER BY hydrophone_id
            """
        ).fetchall()
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        conn.close()

    assert rows == [
        ("noaa_glacier_bay", "NOAA Glacier Bay (Bartlett Cove)"),
        ("sanctsound_ci", "NOAA SanctSound (Channel Islands)"),
        ("sanctsound_oc", "NOAA SanctSound (Olympic Coast)"),
    ]
    assert version == ("025",)
