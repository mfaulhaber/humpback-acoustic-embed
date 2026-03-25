import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from humpback.config import Settings
from humpback.database import Base
from humpback.models import *  # noqa: F401,F403 — ensure all models registered

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
_DEFAULT_ALEMBIC_URL = "sqlite+aiosqlite:///data/humpback.db"


def _resolve_database_url() -> str:
    env_url = os.getenv("HUMPBACK_DATABASE_URL")
    if env_url:
        return env_url

    config_url = config.get_main_option("sqlalchemy.url")
    if config_url and config_url != _DEFAULT_ALEMBIC_URL:
        return config_url

    return Settings.from_repo_env().database_url


def run_migrations_offline():
    url = _resolve_database_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online():
    connectable = create_async_engine(_resolve_database_url())
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
