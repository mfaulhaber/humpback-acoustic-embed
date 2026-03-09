"""Unit tests for classifier worker extraction behavior."""

import json

from humpback.models.classifier import DetectionJob
from humpback.workers.classifier_worker import run_extraction_job


async def test_hydrophone_extraction_uses_local_hls_client_with_default_cache_path(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """Hydrophone extraction should use LocalHLSClient(settings.s3_cache_path)."""
    tsv_path = tmp_path / "detections.tsv"
    tsv_path.write_text(
        "filename\tstart_sec\tend_sec\tavg_confidence\tpeak_confidence\thumpback\tship\tbackground\n"
        "20250615T080000Z.wav\t0.0\t5.0\t0.9\t0.95\t1\t\t\n"
    )

    settings.s3_cache_path = str(tmp_path / "cache-root")
    capture: dict[str, object] = {}

    class DummyLocalHLSClient:
        def __init__(self, cache_path: str):
            capture["cache_path"] = cache_path

    class DummyCachingS3Client:
        def __init__(self, _cache_path: str):
            raise AssertionError("CachingS3Client should not be used for extraction")

    def _fake_extract_hydrophone_labeled_samples(
        _tsv_path,
        _hydrophone_id,
        _positive_output_path,
        _negative_output_path,
        client,
        *_args,
        **_kwargs,
    ):
        capture["client"] = client
        return {"n_humpback": 1, "n_ship": 0, "n_background": 0, "n_skipped": 0}

    monkeypatch.setattr(
        "humpback.classifier.s3_stream.LocalHLSClient",
        DummyLocalHLSClient,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.CachingS3Client",
        DummyCachingS3Client,
    )
    monkeypatch.setattr(
        "humpback.classifier.extractor.extract_hydrophone_labeled_samples",
        _fake_extract_hydrophone_labeled_samples,
    )

    job = DetectionJob(
        status="complete",
        extract_status="running",
        classifier_model_id="missing-model-is-allowed",
        hydrophone_id="rpi_orcasound_lab",
        hydrophone_name="Orcasound Lab",
        start_timestamp=1000.0,
        end_timestamp=2000.0,
        output_tsv_path=str(tsv_path),
        extract_config=json.dumps(
            {
                "positive_output_path": str(tmp_path / "pos"),
                "negative_output_path": str(tmp_path / "neg"),
            }
        ),
        local_cache_path=None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    await run_extraction_job(session, job, settings)
    await session.refresh(job)

    assert capture["cache_path"] == settings.s3_cache_path
    assert isinstance(capture["client"], DummyLocalHLSClient)
    assert job.extract_status == "complete"
    assert job.extract_summary is not None
