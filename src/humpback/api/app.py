from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from humpback.api.routers import admin, audio, classifier, clustering, processing
from humpback.config import Settings
from humpback.database import (
    Base,
    create_engine,
    create_session_factory,
    setup_sqlite_pragmas,
)
from humpback.services.model_registry_service import seed_default_model

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DIST_DIR = STATIC_DIR / "dist"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    if settings is None:
        settings = Settings.from_repo_env()

    app = FastAPI(title="Humpback Acoustic Embedding Platform", version="0.1.0")
    app.state.settings = settings
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

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
        # Seed default model if registry is empty
        async with app.state.session_factory() as session:
            await seed_default_model(session)

    @app.on_event("shutdown")
    async def shutdown():
        await engine.dispose()

    import humpback.models.classifier  # noqa: F401 — register tables

    app.include_router(audio.router)
    app.include_router(processing.router)
    app.include_router(clustering.router)
    app.include_router(classifier.router)
    app.include_router(admin.router)

    # Serve React SPA from dist/ if it exists, otherwise fall back to legacy index.html
    has_dist = DIST_DIR.is_dir() and (DIST_DIR / "index.html").exists()

    if has_dist:
        app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/")
    async def root():
        if has_dist:
            return FileResponse(DIST_DIR / "index.html")
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/app/{full_path:path}")
    async def spa_catchall(full_path: str):
        if has_dist:
            return FileResponse(DIST_DIR / "index.html")
        return FileResponse(STATIC_DIR / "index.html")

    # Keep legacy static mount for backward compat
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def main():
    settings = Settings.from_repo_env()
    app = create_app(settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
