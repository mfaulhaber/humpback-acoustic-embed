import pytest
from httpx import ASGITransport, AsyncClient

from humpback.api.app import create_app
from humpback.config import Settings


@pytest.fixture
def trusted_host_settings(tmp_path):
    db_path = tmp_path / "trusted-hosts.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
        allowed_hosts=["*.trycloudflare.com", "localhost", "127.0.0.1"],
    )


async def test_trusted_hosts_allows_localhost(trusted_host_settings):
    app = create_app(trusted_host_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        await app.router.startup()
        resp = await client.get("/classifier/extraction-settings")
        await app.router.shutdown()

    assert resp.status_code == 200


async def test_trusted_hosts_allows_trycloudflare_subdomain(trusted_host_settings):
    app = create_app(trusted_host_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://abc.trycloudflare.com",
    ) as client:
        await app.router.startup()
        resp = await client.get("/classifier/extraction-settings")
        await app.router.shutdown()

    assert resp.status_code == 200


async def test_trusted_hosts_rejects_unlisted_host(trusted_host_settings):
    app = create_app(trusted_host_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://evil.example.com",
    ) as client:
        await app.router.startup()
        resp = await client.get("/classifier/extraction-settings")
        await app.router.shutdown()

    assert resp.status_code == 400
    assert resp.text == "Invalid host header"
