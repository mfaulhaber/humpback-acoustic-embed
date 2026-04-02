"""Unit tests for vocalization inference pipeline."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from humpback.classifier.vocalization_inference import (
    load_vocalization_model,
    read_predictions,
    run_inference,
    score_embeddings,
)
from humpback.classifier.vocalization_trainer import (
    save_model_artifacts,
    train_multilabel_classifiers,
)


def _train_and_save(tmp_path, dim=16, n_per_type=20):
    """Helper: train model on synthetic data and save artifacts."""
    rng = np.random.RandomState(42)
    types = ["whup", "moan"]
    centers = {"whup": rng.randn(dim) * 2, "moan": rng.randn(dim) * 2 + 4.0}

    embeddings = []
    label_sets: list[set[str]] = []
    for t in types:
        for _ in range(n_per_type):
            embeddings.append(centers[t] + rng.randn(dim) * 0.3)
            label_sets.append({t})
    for _ in range(10):
        embeddings.append(rng.randn(dim) * 5)
        label_sets.append(set())

    X = np.array(embeddings, dtype=np.float32)
    pipelines, thresholds, metrics = train_multilabel_classifiers(X, label_sets)

    model_dir = tmp_path / "model"
    save_model_artifacts(model_dir, pipelines, thresholds, metrics)
    return model_dir, X


class TestLoadModel:
    def test_load_roundtrip(self, tmp_path):
        model_dir, _ = _train_and_save(tmp_path)
        pipelines, vocabulary, thresholds = load_vocalization_model(model_dir)

        assert sorted(vocabulary) == ["moan", "whup"]
        assert "whup" in pipelines
        assert "moan" in pipelines
        assert 0.0 < thresholds["whup"] < 1.0

    def test_load_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_vocalization_model(tmp_path / "nonexistent")


class TestScoreEmbeddings:
    def test_score_shape(self, tmp_path):
        model_dir, X = _train_and_save(tmp_path)
        pipelines, vocabulary, _ = load_vocalization_model(model_dir)

        scores = score_embeddings(pipelines, vocabulary, X[:5])
        for type_name in vocabulary:
            assert scores[type_name].shape == (5,)
            assert np.all(scores[type_name] >= 0)
            assert np.all(scores[type_name] <= 1)


class TestRunInference:
    def test_basic_inference_with_filenames(self, tmp_path):
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        filenames = [f"file_{i}.wav" for i in range(len(X))]
        start_secs = [float(i * 5) for i in range(len(X))]
        end_secs = [float(i * 5 + 5) for i in range(len(X))]

        result = run_inference(
            model_dir,
            X,
            output_path,
            filenames=filenames,
            start_secs=start_secs,
            end_secs=end_secs,
        )

        assert result["total_windows"] == len(X)
        assert "whup" in result["per_type_counts"]
        assert "moan" in result["per_type_counts"]
        assert result["vocabulary"] == ["moan", "whup"]

        table = pq.read_table(str(output_path))
        assert table.num_rows == len(X)
        assert "filename" in table.column_names
        assert "whup" in table.column_names

    def test_basic_inference_with_row_ids(self, tmp_path):
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        row_ids = [f"row-{i}" for i in range(len(X))]

        result = run_inference(model_dir, X, output_path, row_ids=row_ids)

        assert result["total_windows"] == len(X)
        table = pq.read_table(str(output_path))
        assert table.num_rows == len(X)
        assert "row_id" in table.column_names
        assert "filename" not in table.column_names
        assert table.column("row_id").to_pylist() == row_ids

    def test_inference_empty(self, tmp_path):
        model_dir, _ = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        result = run_inference(
            model_dir,
            np.empty((0, 16), dtype=np.float32),
            output_path,
        )

        assert result["total_windows"] == 0
        assert output_path.exists()


class TestReadPredictions:
    def test_read_with_thresholds(self, tmp_path):
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        filenames = [f"file_{i}.wav" for i in range(len(X))]
        start_secs = [float(i * 5) for i in range(len(X))]
        end_secs = [float(i * 5 + 5) for i in range(len(X))]

        run_inference(
            model_dir,
            X,
            output_path,
            filenames=filenames,
            start_secs=start_secs,
            end_secs=end_secs,
        )

        _, vocabulary, thresholds = load_vocalization_model(model_dir)
        rows = read_predictions(output_path, vocabulary, thresholds)

        assert len(rows) == len(X)
        for row in rows:
            assert "filename" in row
            assert "scores" in row
            assert "tags" in row
            assert isinstance(row["tags"], list)

    def test_threshold_overrides(self, tmp_path):
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        filenames = [f"file_{i}.wav" for i in range(len(X))]
        start_secs = [float(i * 5) for i in range(len(X))]
        end_secs = [float(i * 5 + 5) for i in range(len(X))]

        run_inference(
            model_dir,
            X,
            output_path,
            filenames=filenames,
            start_secs=start_secs,
            end_secs=end_secs,
        )

        _, vocabulary, thresholds = load_vocalization_model(model_dir)

        # With very low threshold, everything should be tagged
        rows_low = read_predictions(
            output_path,
            vocabulary,
            thresholds,
            threshold_overrides={"whup": 0.01, "moan": 0.01},
        )
        tagged_low = sum(len(r["tags"]) for r in rows_low)

        # With very high threshold, fewer should be tagged
        rows_high = read_predictions(
            output_path,
            vocabulary,
            thresholds,
            threshold_overrides={"whup": 0.99, "moan": 0.99},
        )
        tagged_high = sum(len(r["tags"]) for r in rows_high)

        assert tagged_low > tagged_high

    def test_read_nonexistent(self, tmp_path):
        rows = read_predictions(tmp_path / "nope.parquet", ["whup"], {"whup": 0.5})
        assert rows == []


class TestConfidenceThreading:
    """Tests for detection confidence flowing through the inference pipeline."""

    def test_run_inference_with_confidences(self, tmp_path):
        """run_inference writes confidence column when provided."""
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        n = len(X)
        filenames = [f"file_{i}.wav" for i in range(n)]
        start_secs = [float(i * 5) for i in range(n)]
        end_secs = [float(i * 5 + 5) for i in range(n)]
        confidences = [0.9 - i * 0.01 for i in range(n)]

        run_inference(
            model_dir,
            X,
            output_path,
            filenames=filenames,
            start_secs=start_secs,
            end_secs=end_secs,
            confidences=confidences,
        )

        table = pq.read_table(str(output_path))
        assert "confidence" in table.column_names
        stored = table.column("confidence").to_pylist()
        assert len(stored) == n
        assert stored[0] == pytest.approx(0.9, abs=1e-4)

    def test_run_inference_without_confidences(self, tmp_path):
        """run_inference omits confidence column when not provided."""
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        filenames = [f"file_{i}.wav" for i in range(len(X))]
        start_secs = [float(i * 5) for i in range(len(X))]
        end_secs = [float(i * 5 + 5) for i in range(len(X))]

        run_inference(
            model_dir,
            X,
            output_path,
            filenames=filenames,
            start_secs=start_secs,
            end_secs=end_secs,
        )

        table = pq.read_table(str(output_path))
        assert "confidence" not in table.column_names

    def test_read_predictions_with_confidence(self, tmp_path):
        """read_predictions includes confidence when present in parquet."""
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        n = len(X)
        filenames = [f"file_{i}.wav" for i in range(n)]
        start_secs = [float(i * 5) for i in range(n)]
        end_secs = [float(i * 5 + 5) for i in range(n)]
        confidences = [0.8, 0.6] + [0.5] * (n - 2)

        run_inference(
            model_dir,
            X,
            output_path,
            filenames=filenames,
            start_secs=start_secs,
            end_secs=end_secs,
            confidences=confidences,
        )

        _, vocabulary, thresholds = load_vocalization_model(model_dir)
        rows = read_predictions(output_path, vocabulary, thresholds)

        assert rows[0]["confidence"] == pytest.approx(0.8, abs=1e-4)
        assert rows[1]["confidence"] == pytest.approx(0.6, abs=1e-4)

    def test_read_predictions_without_confidence(self, tmp_path):
        """read_predictions omits confidence when not in parquet."""
        model_dir, X = _train_and_save(tmp_path)
        output_path = tmp_path / "output" / "predictions.parquet"

        filenames = [f"file_{i}.wav" for i in range(len(X))]
        start_secs = [float(i * 5) for i in range(len(X))]
        end_secs = [float(i * 5 + 5) for i in range(len(X))]

        run_inference(
            model_dir,
            X,
            output_path,
            filenames=filenames,
            start_secs=start_secs,
            end_secs=end_secs,
        )

        _, vocabulary, thresholds = load_vocalization_model(model_dir)
        rows = read_predictions(output_path, vocabulary, thresholds)

        assert "confidence" not in rows[0]
