import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UUIDMixin:
    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


def _ensure_sqlite_dir(url: str) -> None:
    """Create parent directory for SQLite database file if it doesn't exist."""
    if "sqlite" not in url:
        return
    # SQLAlchemy SQLite URLs: "sqlite:///relative/path" or "sqlite:////absolute/path"
    # Extract the path after the ":///" prefix.
    marker = ":///"
    idx = url.find(marker)
    if idx == -1:
        return
    db_path = url[idx + len(marker) :]
    if not db_path or db_path == ":memory:":
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def create_engine(url: str, **kwargs):
    _ensure_sqlite_dir(url)
    engine = create_async_engine(url, **kwargs)
    return engine


async def setup_sqlite_pragmas(engine):
    """Run SQLite PRAGMAs using an async connection."""
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")


def create_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session
