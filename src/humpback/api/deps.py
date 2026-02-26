from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session_factory(request: Request):
    return request.app.state.session_factory


async def get_session(
    request: Request,
) -> AsyncGenerator[AsyncSession, None]:
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
