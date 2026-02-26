import struct
import wave
from pathlib import Path

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


@pytest.fixture
async def client(app_settings):
    app = create_app(app_settings)
    # Trigger startup
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Manually trigger startup event
        await app.router.startup()
        yield ac
        await app.router.shutdown()


@pytest.fixture
def wav_bytes() -> bytes:
    """Small WAV file as bytes."""
    import io
    import math

    sr = 16000
    duration = 2.0
    n = int(sr * duration)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()
