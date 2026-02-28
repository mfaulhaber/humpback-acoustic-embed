import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory


@pytest.fixture
def tmp_storage(tmp_path):
    return tmp_path


@pytest.fixture
def settings(tmp_path):
    db_path = tmp_path / "test.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
    )


@pytest.fixture
async def engine(settings):
    eng = create_engine(settings.database_url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    factory = create_session_factory(engine)
    async with factory() as sess:
        yield sess


@pytest.fixture
def test_wav(tmp_path) -> Path:
    """Generate a ~10 second 16kHz mono sine-wave WAV file."""
    path = tmp_path / "test_audio.wav"
    sample_rate = 16000
    duration = 10.0
    n_samples = int(sample_rate * duration)
    import math

    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return path


@pytest.fixture
def test_flac(tmp_path) -> Path:
    """Generate a ~10 second 16kHz mono sine-wave FLAC file."""
    import soundfile as sf

    path = tmp_path / "test_audio.flac"
    sample_rate = 16000
    duration = 10.0
    n_samples = int(sample_rate * duration)
    import math

    samples = np.array(
        [math.sin(2 * math.pi * 440 * i / sample_rate) for i in range(n_samples)],
        dtype=np.float32,
    )
    sf.write(str(path), samples, sample_rate, format="FLAC")
    return path
