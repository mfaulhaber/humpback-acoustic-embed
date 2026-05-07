"""Shared Alembic and SQLite helpers for migration tests."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from alembic.config import Config

from humpback.database import Base, create_engine

REPO_ROOT = Path(__file__).resolve().parents[2]


def db_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


async def create_current_schema_db(db_path: Path) -> None:
    """Create a SQLite database from the currently registered SQLAlchemy models."""
    engine = create_engine(db_url(db_path))
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


def create_current_schema_db_sync(db_path: Path) -> None:
    asyncio.run(create_current_schema_db(db_path))


def alembic_config(db_path: Path) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", db_url(db_path))
    return config


def stamp_revision_row(db_path: Path, revision: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)", (revision,)
        )
        conn.commit()
    finally:
        conn.close()


def sqlite_tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def sqlite_table_exists(db_path: Path, table: str) -> bool:
    return table in sqlite_tables(db_path)


def sqlite_columns(db_path: Path, table: str) -> dict[str, dict[str, Any]]:
    """Return column metadata keyed by column name from PRAGMA table_info."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {
        r[1]: {"type": r[2], "notnull": bool(r[3]), "default": r[4], "pk": bool(r[5])}
        for r in rows
    }


def sqlite_column_names(db_path: Path, table: str) -> set[str]:
    return set(sqlite_columns(db_path, table))


def sqlite_indexes(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    finally:
        conn.close()
    return {r[1] for r in rows}
