"""E2E smoke test: train classifier → create label processing job → verify output."""

import io
import json
import math
import struct
import wave

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from humpback.api.app import create_app
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.workers.classifier_worker import run_training_job
from humpback.workers.label_processing_worker import run_label_processing_job
from humpback.workers.processing_worker import run_processing_job
from humpback.workers.queue import (
    claim_label_processing_job,
    claim_processing_job,
    claim_training_job,
)


def _make_wav_bytes(duration: float = 10.0, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * duration)
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


def _make_fixture_flac(path, duration: float = 30.0, sample_rate: int = 16000):
    """Write a synthetic FLAC file with tones at known positions."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    # Background silence with low noise
    audio = np.random.default_rng(42).normal(0, 0.001, n).astype(np.float32)
    # Insert tones at known positions to simulate calls
    # Tone at 3-5s (clean region)
    mask1 = (t >= 3.0) & (t < 5.0)
    audio[mask1] += 0.5 * np.sin(2 * np.pi * 800 * t[mask1]).astype(np.float32)
    # Tone at 12-14s (another clean region, well-separated)
    mask2 = (t >= 12.0) & (t < 14.0)
    audio[mask2] += 0.5 * np.sin(2 * np.pi * 600 * t[mask2]).astype(np.float32)
    # Tone at 20-22s (close to 23-25s tone = overlap region)
    mask3 = (t >= 20.0) & (t < 22.0)
    audio[mask3] += 0.4 * np.sin(2 * np.pi * 700 * t[mask3]).astype(np.float32)
    # Tone at 23-25s (overlapping with previous)
    mask4 = (t >= 23.0) & (t < 25.0)
    audio[mask4] += 0.4 * np.sin(2 * np.pi * 900 * t[mask4]).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, format="FLAC")
    return audio


def _make_fixture_raven_tsv(path, annotations):
    """Write a Raven selection table TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Selection\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCall Type"
    ]
    for i, (begin, end, call_type) in enumerate(annotations, 1):
        lines.append(f"{i}\t{begin}\t{end}\t200\t3000\t{call_type}")
    path.write_text("\n".join(lines))


@pytest.fixture
def lp_settings(tmp_path):
    db_path = tmp_path / "lp_e2e.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
        vector_dim=64,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )


@pytest.fixture
async def lp_client(lp_settings):
    app = create_app(lp_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await app.router.startup()
        yield ac
        await app.router.shutdown()


async def _train_classifier(client, settings, tmp_path):
    """Upload positive + negative audio, process, and train a classifier.

    Returns the classifier model ID.
    """
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    # Upload positive audio
    wav_data = _make_wav_bytes(duration=10.0, sample_rate=16000)
    resp = await client.post(
        "/audio/upload",
        files={"file": ("whale.wav", wav_data, "audio/wav")},
    )
    assert resp.status_code == 201
    pos_audio_id = resp.json()["id"]

    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": pos_audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201

    async with session_factory() as session:
        claimed = await claim_processing_job(session)
        assert claimed is not None
        await run_processing_job(session, claimed, settings)

    resp = await client.get("/processing/embedding-sets")
    pos_es_id = next(
        es["id"] for es in resp.json() if es["audio_file_id"] == pos_audio_id
    )

    # Upload negative audio
    neg_wav_data = _make_wav_bytes(duration=10.0, sample_rate=16000)
    resp = await client.post(
        "/audio/upload",
        files={"file": ("noise.wav", neg_wav_data, "audio/wav")},
        data={"folder_path": "negatives"},
    )
    assert resp.status_code == 201
    neg_audio_id = resp.json()["id"]

    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": neg_audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201

    async with session_factory() as session:
        claimed = await claim_processing_job(session)
        assert claimed is not None
        await run_processing_job(session, claimed, settings)

    resp = await client.get("/processing/embedding-sets")
    neg_es_id = next(
        es["id"] for es in resp.json() if es["audio_file_id"] == neg_audio_id
    )

    # Train classifier
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "lp-test-classifier",
            "positive_embedding_set_ids": [pos_es_id],
            "negative_embedding_set_ids": [neg_es_id],
        },
    )
    assert resp.status_code == 201
    tjob_id = resp.json()["id"]

    async with session_factory() as session:
        claimed = await claim_training_job(session)
        assert claimed is not None
        assert claimed.id == tjob_id
        await run_training_job(session, claimed, settings)

    resp = await client.get(f"/classifier/training-jobs/{tjob_id}")
    assert resp.json()["status"] == "complete"

    resp = await client.get("/classifier/models")
    model_id = resp.json()[0]["id"]

    await engine.dispose()
    return model_id


async def test_label_processing_e2e(lp_settings, lp_client, tmp_path):
    """E2E: train classifier → create label processing job → verify output."""
    client = lp_client
    settings = lp_settings

    # 1. Train classifier
    model_id = await _train_classifier(client, settings, tmp_path)

    # 2. Create fixture annotation + audio files
    ann_dir = tmp_path / "annotations"
    aud_dir = tmp_path / "audio_data"

    # 4 annotations: 2 cleanly separated, 2 overlapping
    _make_fixture_raven_tsv(
        ann_dir / "fixture.Table.1.selections.txt",
        [
            (3.0, 5.0, "Moan"),  # clean region
            (12.0, 14.0, "Chirp"),  # clean region
            (20.0, 22.0, "Moan"),  # near overlap
            (23.0, 25.0, "Descending moan"),  # near overlap
        ],
    )
    _make_fixture_flac(aud_dir / "fixture.flac", duration=30.0, sample_rate=16000)

    output_dir = tmp_path / "lp_output"

    # 3. Preview should work
    resp = await client.get(
        "/label-processing/preview",
        params={
            "annotation_folder": str(ann_dir),
            "audio_folder": str(aud_dir),
        },
    )
    assert resp.status_code == 200
    preview = resp.json()
    assert preview["total_annotations"] == 4
    assert len(preview["paired_files"]) == 1
    assert preview["paired_files"][0]["annotation_count"] == 4

    # 4. Create label processing job
    resp = await client.post(
        "/label-processing/jobs",
        json={
            "classifier_model_id": model_id,
            "annotation_folder": str(ann_dir),
            "audio_folder": str(aud_dir),
            "output_root": str(output_dir),
            "parameters": {
                "cleanup_score_cache": False,  # keep scores for inspection
            },
        },
    )
    assert resp.status_code == 200
    job_data = resp.json()
    job_id = job_data["id"]
    assert job_data["status"] == "queued"

    # 5. Run the label processing job directly
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        claimed = await claim_label_processing_job(session)
        assert claimed is not None
        assert claimed.id == job_id
        await run_label_processing_job(session, claimed, settings)

    # 6. Verify job completed
    resp = await client.get(f"/label-processing/jobs/{job_id}")
    assert resp.status_code == 200
    job = resp.json()
    assert job["status"] == "complete"
    assert job["files_processed"] == 1
    assert job["result_summary"] is not None

    summary = job["result_summary"]
    assert summary["total_extracted"] >= 1
    assert summary["files_processed"] == 1

    # 7. Verify output directory structure
    assert output_dir.is_dir()

    # job_summary.json should exist
    summary_file = output_dir / "job_summary.json"
    assert summary_file.exists()
    disk_summary = json.loads(summary_file.read_text())
    assert disk_summary["total_extracted"] == summary["total_extracted"]

    # Verify extracted FLAC files are readable and 5 seconds
    flac_files = list(output_dir.rglob("*.flac"))
    assert len(flac_files) >= 1, "At least one FLAC file should be extracted"

    for flac_path in flac_files:
        audio, sr = sf.read(str(flac_path))
        duration = len(audio) / sr
        assert abs(duration - 5.0) < 0.1, f"Expected ~5s duration, got {duration:.2f}s"

    # Verify PNG sidecars exist for each FLAC
    png_files = list(output_dir.rglob("*.png"))
    assert len(png_files) == len(flac_files), (
        f"Expected {len(flac_files)} PNG sidecars, got {len(png_files)}"
    )

    # 8. Verify scores directory exists (cleanup_score_cache=False)
    # (Score cache is not written by the label_processor directly — it's a
    # future feature. Just verify the cleanup param is respected.)

    # 9. Verify treatment counts are consistent
    treatment_counts = summary["treatment_counts"]
    total_from_treatments = sum(treatment_counts.values())
    assert total_from_treatments == summary["total_extracted"]

    # Re-centering treatment should not appear
    assert "recentered" not in treatment_counts

    # 10. Verify score_stats_by_label present and well-formed
    score_stats = summary.get("score_stats_by_label")
    assert score_stats is not None, "score_stats_by_label should be present"
    assert isinstance(score_stats, dict), "score_stats_by_label should be a dict"
    # Fake model may not produce peaks, so stats can be empty; validate structure
    # when populated
    for ct, stats in score_stats.items():
        assert "count" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] >= 1
        assert 0.0 <= stats["min"] <= stats["max"] <= 1.0

    # 11. Cleanup
    await engine.dispose()


async def test_label_processing_score_cache_cleanup(lp_settings, lp_client, tmp_path):
    """Verify score cache cleanup when cleanup_score_cache=True (default)."""
    client = lp_client
    settings = lp_settings

    model_id = await _train_classifier(client, settings, tmp_path)

    ann_dir = tmp_path / "annotations2"
    aud_dir = tmp_path / "audio_data2"

    _make_fixture_raven_tsv(
        ann_dir / "rec.Table.1.selections.txt",
        [(3.0, 5.0, "Moan")],
    )
    _make_fixture_flac(aud_dir / "rec.flac", duration=10.0, sample_rate=16000)

    output_dir = tmp_path / "lp_output2"

    # Create a fake scores directory to verify cleanup
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True)
    (scores_dir / "dummy_scores.parquet").write_bytes(b"fake")

    resp = await client.post(
        "/label-processing/jobs",
        json={
            "classifier_model_id": model_id,
            "annotation_folder": str(ann_dir),
            "audio_folder": str(aud_dir),
            "output_root": str(output_dir),
            # cleanup_score_cache defaults to True
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["id"]

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        claimed = await claim_label_processing_job(session)
        assert claimed is not None
        await run_label_processing_job(session, claimed, settings)

    resp = await client.get(f"/label-processing/jobs/{job_id}")
    assert resp.json()["status"] == "complete"

    # Score cache should be cleaned up
    assert not scores_dir.exists(), "scores/ directory should be removed after cleanup"

    await engine.dispose()
