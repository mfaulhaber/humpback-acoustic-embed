"""Tests for ``generate_manifest`` model-version enforcement."""

from __future__ import annotations

import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy.orm import Session

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    write_detection_row_store,
)
from humpback.database import Base  # noqa: F401

# Ensure models register on Base.metadata before table creation.
import humpback.models.classifier  # noqa: F401, E402
import humpback.models.detection_embedding_job  # noqa: F401, E402
import humpback.models.hyperparameter  # noqa: F401, E402
import humpback.models.vocalization  # noqa: F401, E402
import humpback.models.labeling  # noqa: F401, E402
from humpback.models.classifier import (  # noqa: E402
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)
from humpback.models.labeling import VocalizationLabel  # noqa: E402
from humpback.services.hyperparameter_service.manifest import generate_manifest  # noqa: E402
from humpback.storage import (  # noqa: E402
    detection_dir,
    detection_embeddings_path,
    detection_row_store_path,
)


@pytest.fixture
def sync_db_url(tmp_path):
    db_path = tmp_path / "m.db"
    sync_url = f"sqlite:///{db_path}"
    eng = sa_create_engine(sync_url)
    Base.metadata.create_all(eng)
    eng.dispose()
    return sync_url


def _write_row_store(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_row_store(path, rows)


def _write_embeddings(emb_path: Path, row_ids: list[str], dim: int = 8) -> None:
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "row_id": row_ids,
            "embedding": [[0.1] * dim for _ in row_ids],
            "confidence": [0.9] * len(row_ids),
        },
        schema=schema,
    )
    pq.write_table(table, str(emb_path))


def _add_classifier_and_detection_job(
    sync_db_url: str,
    *,
    cm_id: str,
    dj_id: str,
    model_version: str,
    row_store_path: Path,
) -> None:
    eng = sa_create_engine(sync_db_url)
    with Session(eng) as session:
        session.add(
            ClassifierModel(
                id=cm_id,
                name="cm",
                model_path="/tmp/m.joblib",
                model_version=model_version,
                vector_dim=8,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        session.flush()
        session.add(
            DetectionJob(
                id=dj_id,
                status="complete",
                classifier_model_id=cm_id,
                audio_folder="/tmp/audio",
                confidence_threshold=0.5,
                hop_seconds=1.0,
                high_threshold=0.7,
                low_threshold=0.45,
                has_positive_labels=True,
                output_row_store_path=str(row_store_path),
            )
        )
        session.commit()
    eng.dispose()


def _add_training_job(
    sync_db_url: str,
    tj_id: str,
    model_version: str,
    *,
    source_mode: str = "detection_manifest",
) -> None:
    eng = sa_create_engine(sync_db_url)
    with Session(eng) as session:
        session.add(
            ClassifierTrainingJob(
                id=tj_id,
                status="complete",
                name="tj",
                model_version=model_version,
                window_size_seconds=5.0,
                target_sample_rate=32000,
                source_mode=source_mode,
            )
        )
        session.commit()
    eng.dispose()


def _add_vocalization_label(
    sync_db_url: str, dj_id: str, row_id: str, label: str
) -> None:
    eng = sa_create_engine(sync_db_url)
    with Session(eng) as session:
        session.add(
            VocalizationLabel(
                id=str(uuid.uuid4()),
                detection_job_id=dj_id,
                row_id=row_id,
                label=label,
            )
        )
        session.commit()
    eng.dispose()


def _make_row(row_id: str, start_utc: float, label: str) -> dict[str, str]:
    r = {f: "" for f in ROW_STORE_FIELDNAMES}
    r["row_id"] = row_id
    r["start_utc"] = str(start_utc)
    r["end_utc"] = str(start_utc + 5)
    r[label] = "1"
    return r


def _patch_settings(monkeypatch, storage_root):
    from humpback.config import Settings

    s = Settings()
    s.storage_root = storage_root  # type: ignore[assignment]
    monkeypatch.setattr(
        Settings,
        "from_repo_env",
        classmethod(lambda cls: s),
    )


def test_rejects_missing_embedding_model_version(tmp_path, sync_db_url, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="embedding_model_version is required"):
        generate_manifest(
            training_job_ids=["tj-1"],
            db_url=sync_db_url,
            embedding_model_version=None,
        )


def test_rejects_legacy_embedding_set_training_job(tmp_path, sync_db_url, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    tj_id = str(uuid.uuid4())
    _add_training_job(
        sync_db_url,
        tj_id,
        "tf2_some_model",
        source_mode="embedding_sets",
    )

    with pytest.raises(ValueError, match="retired embedding-set sources"):
        generate_manifest(
            training_job_ids=[tj_id],
            db_url=sync_db_url,
            embedding_model_version="perch_v2",
        )


def test_rejects_missing_detection_embeddings(tmp_path, sync_db_url, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    dj_id = str(uuid.uuid4())
    cm_id = str(uuid.uuid4())
    rs_path = detection_row_store_path(tmp_path, dj_id)
    _write_row_store(rs_path, [_make_row("r1", 1000.0, "humpback")])
    _add_classifier_and_detection_job(
        sync_db_url,
        cm_id=cm_id,
        dj_id=dj_id,
        model_version="perch_v2",
        row_store_path=rs_path,
    )

    with pytest.raises(FileNotFoundError, match="Detection embeddings not found"):
        generate_manifest(
            detection_job_ids=[dj_id],
            db_url=sync_db_url,
            embedding_model_version="perch_v2",
        )


def test_promotes_legacy_detection_embeddings(tmp_path, sync_db_url, monkeypatch):
    """Legacy-path embeddings are copied to the model-versioned path when the
    source classifier model version matches."""
    _patch_settings(monkeypatch, tmp_path)
    dj_id = str(uuid.uuid4())
    cm_id = str(uuid.uuid4())
    rs_path = detection_row_store_path(tmp_path, dj_id)
    rows = [
        _make_row("r-pos", 1000.0, "humpback"),
        _make_row("r-neg", 2000.0, "background"),
    ]
    _write_row_store(rs_path, rows)
    _add_classifier_and_detection_job(
        sync_db_url,
        cm_id=cm_id,
        dj_id=dj_id,
        model_version="surfperch-tensorflow2",
        row_store_path=rs_path,
    )

    legacy_path = detection_dir(tmp_path, dj_id) / "detection_embeddings.parquet"
    _write_embeddings(legacy_path, ["r-pos", "r-neg"])

    manifest = generate_manifest(
        detection_job_ids=[dj_id],
        db_url=sync_db_url,
        embedding_model_version="surfperch-tensorflow2",
    )
    assert len(manifest["examples"]) == 2

    promoted_path = detection_embeddings_path(tmp_path, dj_id, "surfperch-tensorflow2")
    assert promoted_path.exists()


def test_legacy_promotion_skipped_on_model_mismatch(tmp_path, sync_db_url, monkeypatch):
    """Legacy-path embeddings are NOT promoted when the source classifier model
    version differs from the requested embedding model version."""
    _patch_settings(monkeypatch, tmp_path)
    dj_id = str(uuid.uuid4())
    cm_id = str(uuid.uuid4())
    rs_path = detection_row_store_path(tmp_path, dj_id)
    _write_row_store(rs_path, [_make_row("r1", 1000.0, "humpback")])
    _add_classifier_and_detection_job(
        sync_db_url,
        cm_id=cm_id,
        dj_id=dj_id,
        model_version="surfperch-tensorflow2",
        row_store_path=rs_path,
    )

    legacy_path = detection_dir(tmp_path, dj_id) / "detection_embeddings.parquet"
    _write_embeddings(legacy_path, ["r1"])

    with pytest.raises(FileNotFoundError, match="Detection embeddings not found"):
        generate_manifest(
            detection_job_ids=[dj_id],
            db_url=sync_db_url,
            embedding_model_version="perch_v2",
        )


def test_perch_v2_binary_only_labels_skip_vocalization_labels(
    tmp_path, sync_db_url, monkeypatch
):
    _patch_settings(monkeypatch, tmp_path)
    dj_id = str(uuid.uuid4())
    cm_id = str(uuid.uuid4())
    rs_path = detection_row_store_path(tmp_path, dj_id)
    rows = [
        _make_row("r-pos", 1000.0, "humpback"),
        _make_row("r-neg", 2000.0, "background"),
    ]
    _write_row_store(rs_path, rows)
    _add_classifier_and_detection_job(
        sync_db_url,
        cm_id=cm_id,
        dj_id=dj_id,
        model_version="perch_v2",
        row_store_path=rs_path,
    )

    emb_path = detection_embeddings_path(tmp_path, dj_id, "perch_v2")
    _write_embeddings(emb_path, ["r-pos", "r-neg"])

    # Spurious vocalization label — perch_v2 path must IGNORE it.
    _add_vocalization_label(sync_db_url, dj_id, "r-pos", "(Negative)")

    manifest = generate_manifest(
        detection_job_ids=[dj_id],
        db_url=sync_db_url,
        embedding_model_version="perch_v2",
    )

    examples_by_row = {ex["row_id"]: ex["label"] for ex in manifest["examples"]}
    assert examples_by_row == {"r-pos": 1, "r-neg": 0}
    assert manifest["metadata"]["embedding_model_version"] == "perch_v2"


def test_tf2_path_still_uses_vocalization_labels(tmp_path, sync_db_url, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    dj_id = str(uuid.uuid4())
    cm_id = str(uuid.uuid4())
    rs_path = detection_row_store_path(tmp_path, dj_id)
    _write_row_store(rs_path, [_make_row("r-pos", 1000.0, "humpback")])
    _add_classifier_and_detection_job(
        sync_db_url,
        cm_id=cm_id,
        dj_id=dj_id,
        model_version="tf2_model",
        row_store_path=rs_path,
    )

    emb_path = detection_embeddings_path(tmp_path, dj_id, "tf2_model")
    _write_embeddings(emb_path, ["r-pos"])

    # Vocalization negative on binary-positive row → conflict (vocalization
    # path engaged for tf2_model).
    _add_vocalization_label(sync_db_url, dj_id, "r-pos", "(Negative)")

    manifest = generate_manifest(
        detection_job_ids=[dj_id],
        db_url=sync_db_url,
        embedding_model_version="tf2_model",
    )
    assert not manifest["examples"]
    assert (
        manifest["metadata"]["detection_job_summaries"][dj_id]["skipped_conflicts"] == 1
    )


def test_manifest_metadata_includes_embedding_model_version(
    tmp_path, sync_db_url, monkeypatch
):
    _patch_settings(monkeypatch, tmp_path)
    tj_id = str(uuid.uuid4())
    _add_training_job(sync_db_url, tj_id, "perch_v2")

    manifest = generate_manifest(
        training_job_ids=[tj_id],
        db_url=sync_db_url,
        embedding_model_version="perch_v2",
    )
    assert manifest["metadata"]["embedding_model_version"] == "perch_v2"
