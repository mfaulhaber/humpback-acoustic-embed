from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from humpback.api.routers import admin, audio, clustering, processing
from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory, setup_sqlite_pragmas

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    app = FastAPI(title="Humpback Acoustic Embedding Platform", version="0.1.0")
    app.state.settings = settings

    engine = create_engine(settings.database_url)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)

    @app.on_event("startup")
    async def startup():
        if "sqlite" in settings.database_url:
            await setup_sqlite_pragmas(engine)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        settings.storage_root.mkdir(parents=True, exist_ok=True)

    @app.on_event("shutdown")
    async def shutdown():
        await engine.dispose()

    app.include_router(audio.router)
    app.include_router(processing.router)
    app.include_router(clustering.router)
    app.include_router(admin.router)

    @app.get("/")
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def main():
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
