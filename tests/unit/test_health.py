import pytest
from httpx import ASGITransport, AsyncClient

from humpback.api.app import create_app
from humpback.config import Settings


@pytest.fixture
def app_settings(tmp_path):
    db_path = tmp_path / "test.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
    )


@pytest.mark.asyncio
async def test_health_ok(app_settings):
    """After a successful startup, /health returns 200 ok."""
    app = create_app(app_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await app.router.startup()
        resp = await ac.get("/health")
        await app.router.shutdown()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db"] == "connected"


@pytest.mark.asyncio
async def test_health_error_503(app_settings):
    """When db_healthy is False, /health returns 503 with error detail."""
    app = create_app(app_settings)
    app.state.db_healthy = False
    app.state.db_error = "unable to open database file: /bad/path/db"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["db"] == "unavailable"
    assert "bad/path" in body["detail"]


@pytest.mark.asyncio
async def test_health_starting(app_settings):
    """Before startup completes, /health returns 200 with status=starting."""
    app = create_app(app_settings)
    # db_healthy not yet set (startup not triggered)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "starting"
    assert body["db"] == "unknown"
