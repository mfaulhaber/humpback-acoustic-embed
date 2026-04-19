"""Unit tests for classifier training service — detection-manifest source mode.

Covers the ``create_training_job_from_detection_manifest`` service function:
mixed-source schema rejection, missing embeddings, missing labels, and
successful submission.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.classifier.detection_rows import write_detection_row_store
from humpback.models.classifier import (
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)
from humpback.models.model_registry import ModelConfig
from humpback.schemas.classifier import ClassifierTrainingJobCreate
from humpback.services.classifier_service.training import (
    create_training_job_from_detection_manifest,
)
from humpback.storage import detection_embeddings_path, detection_row_store_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_model_config(session_add, *, name="perch_v2"):
    mc = ModelConfig(
        name=name,
        display_name=f"{name} (TFLite)",
        path=f"models/{name}.tflite",
        model_type="tflite",
        input_format="waveform",
        vector_dim=1536,
        is_default=False,
    )
    session_add(mc)
    return mc


def _seed_classifier_model(session_add, *, model_version="perch_v2"):
    cm = ClassifierModel(
        name="test-classifier",
        model_path="/tmp/fake.joblib",
        model_version=model_version,
        vector_dim=1536,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session_add(cm)
    return cm


def _seed_detection_job(session_add, *, classifier_model_id, status="complete"):
    dj = DetectionJob(
        status=status,
        classifier_model_id=classifier_model_id,
        audio_folder="/tmp/fake",
        confidence_threshold=0.5,
        detection_mode="windowed",
    )
    session_add(dj)
    return dj


def _write_row_store(storage_root, dj_id, rows):
    rs_path = detection_row_store_path(storage_root, dj_id)
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_row_store(rs_path, rows)


def _write_embeddings(storage_root, dj_id, model_version, row_ids):
    emb_path = detection_embeddings_path(storage_root, dj_id, model_version)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(
                [[0.1] * 1536 for _ in row_ids], type=pa.list_(pa.float32())
            ),
        }
    )
    pq.write_table(table, str(emb_path))


# ---------------------------------------------------------------------------
# Schema-level validation (no DB needed)
# ---------------------------------------------------------------------------


class TestClassifierTrainingJobCreateSchema:
    """Pydantic schema validates source-mode exclusivity."""

    def test_mixed_sources_rejected(self):
        with pytest.raises(ValueError, match="Cannot mix"):
            ClassifierTrainingJobCreate(
                name="mixed",
                positive_embedding_set_ids=["es-1"],
                negative_embedding_set_ids=["es-2"],
                detection_job_ids=["dj-1"],
                embedding_model_version="perch_v2",
            )

    def test_empty_sources_rejected(self):
        with pytest.raises(ValueError, match="Must provide"):
            ClassifierTrainingJobCreate(name="empty")

    def test_detection_jobs_require_model_version(self):
        with pytest.raises(ValueError, match="embedding_model_version is required"):
            ClassifierTrainingJobCreate(
                name="no-version",
                detection_job_ids=["dj-1"],
            )

    def test_embedding_sets_require_both_pos_and_neg(self):
        with pytest.raises(ValueError, match="negative embedding set"):
            ClassifierTrainingJobCreate(
                name="pos-only",
                positive_embedding_set_ids=["es-1"],
            )

    def test_valid_detection_job_input(self):
        req = ClassifierTrainingJobCreate(
            name="valid",
            detection_job_ids=["dj-1"],
            embedding_model_version="perch_v2",
        )
        assert req.detection_job_ids == ["dj-1"]
        assert req.embedding_model_version == "perch_v2"

    def test_valid_embedding_set_input(self):
        req = ClassifierTrainingJobCreate(
            name="valid",
            positive_embedding_set_ids=["es-1"],
            negative_embedding_set_ids=["es-2"],
        )
        assert req.positive_embedding_set_ids == ["es-1"]


# ---------------------------------------------------------------------------
# Service-level validation (needs DB session)
# ---------------------------------------------------------------------------


class TestCreateTrainingJobFromDetectionManifest:
    """Tests for the service function directly."""

    @pytest.fixture(autouse=True)
    def _init_storage(self, settings):
        self.storage_root = settings.storage_root

    async def test_unregistered_model_version_rejected(self, session):
        with pytest.raises(ValueError, match="not registered"):
            await create_training_job_from_detection_manifest(
                session,
                name="test",
                detection_job_ids=["nonexistent"],
                embedding_model_version="unknown_model",
                storage_root=self.storage_root,
            )

    async def test_missing_detection_job_rejected(self, session):
        _seed_model_config(session.add)
        await session.flush()

        with pytest.raises(ValueError, match="not found"):
            await create_training_job_from_detection_manifest(
                session,
                name="test",
                detection_job_ids=["nonexistent-dj"],
                embedding_model_version="perch_v2",
                storage_root=self.storage_root,
            )

    async def test_missing_embeddings_rejected(self, session):
        _seed_model_config(session.add)
        cm = _seed_classifier_model(session.add)
        await session.flush()
        dj = _seed_detection_job(session.add, classifier_model_id=cm.id)
        await session.flush()

        # Write row store but no embeddings parquet.
        _write_row_store(
            self.storage_root,
            dj.id,
            [
                {
                    "start_utc": "1000.0",
                    "end_utc": "1005.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "n_windows": "1",
                    "humpback": "1",
                }
            ],
        )

        with pytest.raises(ValueError, match="no embeddings.*re-embed"):
            await create_training_job_from_detection_manifest(
                session,
                name="test",
                detection_job_ids=[dj.id],
                embedding_model_version="perch_v2",
                storage_root=self.storage_root,
            )

    async def test_missing_labels_rejected(self, session):
        """All rows unlabeled -> no positive or negative labels -> rejection."""
        _seed_model_config(session.add)
        cm = _seed_classifier_model(session.add)
        await session.flush()
        dj = _seed_detection_job(session.add, classifier_model_id=cm.id)
        await session.flush()

        # Row store with no binary labels set.
        _write_row_store(
            self.storage_root,
            dj.id,
            [
                {
                    "start_utc": "1000.0",
                    "end_utc": "1005.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "n_windows": "1",
                }
            ],
        )
        _write_embeddings(self.storage_root, dj.id, "perch_v2", ["r1"])

        with pytest.raises(ValueError, match="at least one positive"):
            await create_training_job_from_detection_manifest(
                session,
                name="test",
                detection_job_ids=[dj.id],
                embedding_model_version="perch_v2",
                storage_root=self.storage_root,
            )

    async def test_no_negative_labels_rejected(self, session):
        """All rows labeled positive but none negative -> rejection."""
        _seed_model_config(session.add)
        cm = _seed_classifier_model(session.add)
        await session.flush()
        dj = _seed_detection_job(session.add, classifier_model_id=cm.id)
        await session.flush()

        _write_row_store(
            self.storage_root,
            dj.id,
            [
                {
                    "start_utc": "1000.0",
                    "end_utc": "1005.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "n_windows": "1",
                    "humpback": "1",
                }
            ],
        )
        _write_embeddings(self.storage_root, dj.id, "perch_v2", ["r1"])

        with pytest.raises(ValueError, match="at least one positive.*neg=0"):
            await create_training_job_from_detection_manifest(
                session,
                name="test",
                detection_job_ids=[dj.id],
                embedding_model_version="perch_v2",
                storage_root=self.storage_root,
            )

    async def test_successful_detection_manifest_submission(self, session):
        _seed_model_config(session.add)
        cm = _seed_classifier_model(session.add)
        await session.flush()
        dj = _seed_detection_job(session.add, classifier_model_id=cm.id)
        await session.flush()

        _write_row_store(
            self.storage_root,
            dj.id,
            [
                {
                    "start_utc": "1000.0",
                    "end_utc": "1005.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "n_windows": "1",
                    "humpback": "1",
                },
                {
                    "start_utc": "1005.0",
                    "end_utc": "1010.0",
                    "avg_confidence": "0.3",
                    "peak_confidence": "0.4",
                    "n_windows": "1",
                    "background": "1",
                },
            ],
        )
        _write_embeddings(self.storage_root, dj.id, "perch_v2", ["r1", "r2"])

        job = await create_training_job_from_detection_manifest(
            session,
            name="perch-v2-from-detections",
            detection_job_ids=[dj.id],
            embedding_model_version="perch_v2",
            storage_root=self.storage_root,
        )

        assert isinstance(job, ClassifierTrainingJob)
        assert job.source_mode == "detection_manifest"
        assert job.model_version == "perch_v2"
        assert job.status == "queued"
        assert job.source_detection_job_ids is not None
        assert json.loads(job.source_detection_job_ids) == [dj.id]
        assert job.window_size_seconds == 5.0
        assert job.target_sample_rate == 32000

    async def test_parameters_passed_through(self, session):
        _seed_model_config(session.add)
        cm = _seed_classifier_model(session.add)
        await session.flush()
        dj = _seed_detection_job(session.add, classifier_model_id=cm.id)
        await session.flush()

        _write_row_store(
            self.storage_root,
            dj.id,
            [
                {
                    "start_utc": "1000.0",
                    "end_utc": "1005.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "n_windows": "1",
                    "humpback": "1",
                },
                {
                    "start_utc": "1005.0",
                    "end_utc": "1010.0",
                    "avg_confidence": "0.3",
                    "peak_confidence": "0.4",
                    "n_windows": "1",
                    "ship": "1",
                },
            ],
        )
        _write_embeddings(self.storage_root, dj.id, "perch_v2", ["r1", "r2"])

        params = {"classifier_type": "mlp", "C": 0.5}
        job = await create_training_job_from_detection_manifest(
            session,
            name="with-params",
            detection_job_ids=[dj.id],
            embedding_model_version="perch_v2",
            storage_root=self.storage_root,
            parameters=params,
        )
        assert job.parameters is not None
        assert json.loads(job.parameters) == params


# ---------------------------------------------------------------------------
# Training data summary — detection_manifest source mode
# ---------------------------------------------------------------------------


class TestGetTrainingDataSummaryDetectionManifest:
    """Tests for get_training_data_summary with detection_manifest models."""

    async def test_detection_manifest_with_per_job_counts(self, session):
        from humpback.services.classifier_service.training import (
            get_training_data_summary,
        )

        cm_source = _seed_classifier_model(session.add)
        await session.flush()

        dj1 = DetectionJob(
            status="complete",
            classifier_model_id=cm_source.id,
            hydrophone_name="Orcasound Lab",
            start_timestamp=1635638400.0,
            end_timestamp=1635724800.0,
        )
        dj2 = DetectionJob(
            status="complete",
            classifier_model_id=cm_source.id,
            hydrophone_name="Bush Point",
            start_timestamp=1637366400.0,
            end_timestamp=1637452800.0,
        )
        session.add_all([dj1, dj2])
        await session.flush()

        training_summary = json.dumps(
            {
                "n_positive": 500,
                "n_negative": 300,
                "detection_job_ids": [dj1.id, dj2.id],
                "training_data_source": {
                    "per_job_counts": [
                        {
                            "detection_job_id": dj1.id,
                            "positive_count": 300,
                            "negative_count": 200,
                        },
                        {
                            "detection_job_id": dj2.id,
                            "positive_count": 200,
                            "negative_count": 100,
                        },
                    ],
                },
            }
        )

        tj = ClassifierTrainingJob(
            name="det-manifest-job",
            status="complete",
            source_mode="detection_manifest",
            positive_embedding_set_ids="[]",
            negative_embedding_set_ids="[]",
            model_version="perch_v2",
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(tj)
        await session.flush()

        cm = ClassifierModel(
            name="det-manifest-model",
            model_path="/tmp/fake.joblib",
            model_version="perch_v2",
            vector_dim=1536,
            window_size_seconds=5.0,
            target_sample_rate=32000,
            training_summary=training_summary,
            training_job_id=tj.id,
            training_source_mode="detection_manifest",
        )
        session.add(cm)
        await session.flush()

        result = await get_training_data_summary(session, cm.id)

        assert result is not None
        assert result["total_positive"] == 500
        assert result["total_negative"] == 300
        assert result["positive_sources"] == []
        assert result["negative_sources"] == []
        assert len(result["detection_sources"]) == 2

        src1 = result["detection_sources"][0]
        assert src1["detection_job_id"] == dj1.id
        assert src1["hydrophone_name"] == "Orcasound Lab"
        assert src1["start_timestamp"] == 1635638400.0
        assert src1["positive_count"] == 300
        assert src1["negative_count"] == 200

        src2 = result["detection_sources"][1]
        assert src2["hydrophone_name"] == "Bush Point"

    async def test_detection_manifest_without_per_job_counts(self, session):
        from humpback.services.classifier_service.training import (
            get_training_data_summary,
        )

        cm_source = _seed_classifier_model(session.add)
        await session.flush()

        dj = DetectionJob(
            status="complete",
            classifier_model_id=cm_source.id,
            hydrophone_name="Orcasound Lab",
            start_timestamp=1635638400.0,
            end_timestamp=1635724800.0,
        )
        session.add(dj)
        await session.flush()

        training_summary = json.dumps(
            {
                "n_positive": 100,
                "n_negative": 80,
                "detection_job_ids": [dj.id],
                "training_data_source": {
                    "manifest_path": "/tmp/fake/manifest.json",
                },
            }
        )

        tj = ClassifierTrainingJob(
            name="det-job",
            status="complete",
            source_mode="detection_manifest",
            positive_embedding_set_ids="[]",
            negative_embedding_set_ids="[]",
            model_version="perch_v2",
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(tj)
        await session.flush()

        cm = ClassifierModel(
            name="det-model",
            model_path="/tmp/fake.joblib",
            model_version="perch_v2",
            vector_dim=1536,
            window_size_seconds=5.0,
            target_sample_rate=32000,
            training_summary=training_summary,
            training_job_id=tj.id,
            training_source_mode="detection_manifest",
        )
        session.add(cm)
        await session.flush()

        result = await get_training_data_summary(session, cm.id)

        assert result is not None
        assert len(result["detection_sources"]) == 1
        src = result["detection_sources"][0]
        assert src["hydrophone_name"] == "Orcasound Lab"
        assert src["positive_count"] is None
        assert src["negative_count"] is None
