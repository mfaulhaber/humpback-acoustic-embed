import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from humpback.api.routers import (
    admin,
    audio,
    classifier,
    clustering,
    processing,
)
from humpback.config import Settings
from humpback.database import (
    Base,
    create_engine,
    create_session_factory,
    setup_sqlite_pragmas,
)
from humpback.services.model_registry_service import seed_default_model

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DIST_DIR = STATIC_DIR / "dist"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    if settings is None:
        settings = Settings.from_repo_env()

    app = FastAPI(title="Humpback Acoustic Embedding Platform", version="0.1.0")
    app.state.settings = settings
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    logger.info("Database engine created: %s", settings.database_url)
    engine = create_engine(settings.database_url)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)

    @app.on_event("startup")
    async def startup():
        try:
            if "sqlite" in settings.database_url:
                await setup_sqlite_pragmas(engine)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            settings.storage_root.mkdir(parents=True, exist_ok=True)
            # Seed default model if registry is empty
            async with app.state.session_factory() as session:
                await seed_default_model(session)
            app.state.db_healthy = True
            logger.info("Database ready")
        except Exception as e:
            app.state.db_healthy = False
            app.state.db_error = str(e)
            logger.error("Database initialization failed: %s", e, exc_info=True)
            raise

    @app.on_event("shutdown")
    async def shutdown():
        await engine.dispose()

    @app.get("/health")
    async def health():
        healthy = getattr(app.state, "db_healthy", None)
        if healthy is True:
            return {"status": "ok", "db": "connected"}
        if healthy is False:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "db": "unavailable",
                    "detail": getattr(app.state, "db_error", "unknown error"),
                },
            )
        # startup not yet complete
        return {"status": "starting", "db": "unknown"}

    import humpback.models.classifier  # noqa: F401 — register tables
    import humpback.models.label_processing  # noqa: F401 — register tables
    import humpback.models.labeling  # noqa: F401 — register tables
    import humpback.models.vocalization  # noqa: F401 — register tables

    app.include_router(audio.router)
    app.include_router(processing.router)
    app.include_router(clustering.router)
    app.include_router(classifier.router)
    app.include_router(admin.router)
    from humpback.api.routers import label_processing, labeling, search, vocalization

    app.include_router(search.router)
    app.include_router(label_processing.router)
    app.include_router(labeling.router)
    app.include_router(vocalization.router)

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
