import json
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
async def embedding_set_id(app_settings):
    """Create a minimal EmbeddingSet row for clustering API tests."""
    from sqlalchemy import insert

    from humpback.database import create_engine, create_session_factory
    from humpback.models.audio import AudioFile
    from humpback.models.processing import EmbeddingSet

    audio_id = str(uuid.uuid4())
    es_id = str(uuid.uuid4())
    checksum = uuid.uuid4().hex
    parquet_path = Path(app_settings.storage_root) / "fixtures" / f"{es_id}.parquet"

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        await session.execute(
            insert(AudioFile).values(
                id=audio_id,
                filename="fixture.wav",
                folder_path="fixtures",
                checksum_sha256=checksum,
            )
        )
        await session.execute(
            insert(EmbeddingSet).values(
                id=es_id,
                audio_file_id=audio_id,
                encoding_signature=f"sig-{es_id}",
                model_version="fixture_model",
                window_size_seconds=5.0,
                target_sample_rate=32000,
                vector_dim=1280,
                parquet_path=str(parquet_path),
            )
        )
        await session.commit()
    await engine.dispose()
    return es_id


async def test_create_clustering_job_with_empty_list(client, embedding_set_id):
    """Creating a clustering job with empty embedding_set_ids should fail validation."""
    resp = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": []},
    )
    assert resp.status_code == 422


async def test_create_clustering_job_invalid_ids(client, embedding_set_id):
    """Creating a clustering job with non-existent IDs should fail."""
    resp = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": ["fake-id-1"]},
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


async def test_get_clustering_job(client, embedding_set_id):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job_id


async def test_get_clustering_job_not_found(client, embedding_set_id):
    resp = await client.get("/clustering/jobs/nonexistent")
    assert resp.status_code == 404


async def test_list_clusters_empty(client, embedding_set_id):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/clusters")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_visualization_not_found(client, embedding_set_id):
    resp = await client.get("/clustering/jobs/nonexistent/visualization")
    assert resp.status_code == 404


async def test_visualization_not_complete(client, embedding_set_id):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    # Job is queued, not complete
    resp = await client.get(f"/clustering/jobs/{job_id}/visualization")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"].lower()


async def test_visualization_success(client, embedding_set_id, app_settings):
    """Test visualization endpoint with a manually placed umap_coords.parquet."""
    # Create a job
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]

    # Manually mark job as complete in DB (via the API we can't, so use storage hack)
    # We need to manually update the job status. Use the internal service.
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy import update
    from humpback.models.clustering import ClusteringJob

    engine = create_async_engine(app_settings.database_url)
    async_session = async_sessionmaker(engine)
    async with async_session() as session:
        await session.execute(
            update(ClusteringJob)
            .where(ClusteringJob.id == job_id)
            .values(status="complete")
        )
        await session.commit()
    await engine.dispose()

    # Create umap_coords.parquet in the cluster directory
    cluster_path = Path(app_settings.storage_root) / "clusters" / job_id
    cluster_path.mkdir(parents=True, exist_ok=True)

    n = 5
    table = pa.table(
        {
            "x": pa.array(np.random.randn(n).astype(np.float32), type=pa.float32()),
            "y": pa.array(np.random.randn(n).astype(np.float32), type=pa.float32()),
            "cluster_label": pa.array([0, 0, 1, 1, -1], type=pa.int32()),
            "embedding_set_id": pa.array(
                ["es1", "es1", "es2", "es2", "es1"], type=pa.string()
            ),
            "embedding_row_index": pa.array([0, 1, 0, 1, 2], type=pa.int32()),
        }
    )
    pq.write_table(table, str(cluster_path / "umap_coords.parquet"))

    resp = await client.get(f"/clustering/jobs/{job_id}/visualization")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["x"]) == n
    assert len(data["y"]) == n
    assert len(data["cluster_label"]) == n
    assert len(data["embedding_set_id"]) == n
    assert len(data["embedding_row_index"]) == n
    assert data["cluster_label"] == [0, 0, 1, 1, -1]


async def test_visualization_no_umap_file(client, embedding_set_id, app_settings):
    """Test 404 when job is complete but umap_coords.parquet doesn't exist."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]

    # Mark job as complete
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import update
    from humpback.models.clustering import ClusteringJob

    engine = create_async_engine(app_settings.database_url)
    async_session = async_sessionmaker(engine)
    async with async_session() as session:
        await session.execute(
            update(ClusteringJob)
            .where(ClusteringJob.id == job_id)
            .values(status="complete")
        )
        await session.commit()
    await engine.dispose()

    resp = await client.get(f"/clustering/jobs/{job_id}/visualization")
    assert resp.status_code == 404
    assert "not available" in resp.json()["detail"].lower()


async def _mark_job_complete_with_metrics(app_settings, job_id, metrics=None):
    """Helper to mark a job complete and optionally set metrics_json."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import update
    from humpback.models.clustering import ClusteringJob

    engine = create_async_engine(app_settings.database_url)
    async_session = async_sessionmaker(engine)
    values = {"status": "complete"}
    if metrics is not None:
        values["metrics_json"] = json.dumps(metrics)
    async with async_session() as session:
        await session.execute(
            update(ClusteringJob).where(ClusteringJob.id == job_id).values(**values)
        )
        await session.commit()
    await engine.dispose()


async def test_metrics_not_found(client, embedding_set_id):
    """Metrics for nonexistent job returns 404."""
    resp = await client.get("/clustering/jobs/nonexistent/metrics")
    assert resp.status_code == 404


async def test_metrics_not_complete(client, embedding_set_id):
    """Metrics for a queued job returns 400."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/metrics")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"].lower()


async def test_metrics_empty(client, embedding_set_id, app_settings):
    """Metrics for a complete job with no metrics_json returns empty dict."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    resp = await client.get(f"/clustering/jobs/{job_id}/metrics")
    assert resp.status_code == 200
    assert resp.json() == {}


async def test_metrics_with_data(client, embedding_set_id, app_settings):
    """Metrics endpoint returns stored metrics."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    metrics = {"silhouette_score": 0.75, "n_clusters": 3}
    await _mark_job_complete_with_metrics(app_settings, job_id, metrics)

    resp = await client.get(f"/clustering/jobs/{job_id}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["silhouette_score"] == 0.75
    assert data["n_clusters"] == 3


async def test_metrics_in_job_response(client, embedding_set_id, app_settings):
    """Job detail response includes metrics field."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    metrics = {"silhouette_score": 0.5}
    await _mark_job_complete_with_metrics(app_settings, job_id, metrics)

    resp = await client.get(f"/clustering/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"] is not None
    assert data["metrics"]["silhouette_score"] == 0.5


async def test_parameter_sweep_not_found(client, embedding_set_id):
    """Parameter sweep for nonexistent job returns 404."""
    resp = await client.get("/clustering/jobs/nonexistent/parameter-sweep")
    assert resp.status_code == 404


async def test_parameter_sweep_not_complete(client, embedding_set_id):
    """Parameter sweep for a queued job returns 400."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/parameter-sweep")
    assert resp.status_code == 400


async def test_parameter_sweep_success(client, embedding_set_id, app_settings):
    """Parameter sweep endpoint returns stored sweep data."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    # Write sweep file
    cluster_path = Path(app_settings.storage_root) / "clusters" / job_id
    cluster_path.mkdir(parents=True, exist_ok=True)
    sweep_data = [
        {
            "min_cluster_size": 2,
            "silhouette_score": 0.6,
            "n_clusters": 5,
            "noise_fraction": 0.1,
        },
        {
            "min_cluster_size": 3,
            "silhouette_score": 0.7,
            "n_clusters": 3,
            "noise_fraction": 0.05,
        },
    ]
    (cluster_path / "parameter_sweep.json").write_text(json.dumps(sweep_data))

    resp = await client.get(f"/clustering/jobs/{job_id}/parameter-sweep")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["min_cluster_size"] == 2
    assert data[1]["silhouette_score"] == 0.7


async def test_parameter_sweep_no_file(client, embedding_set_id, app_settings):
    """Parameter sweep 404 when file doesn't exist."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    resp = await client.get(f"/clustering/jobs/{job_id}/parameter-sweep")
    assert resp.status_code == 404


# --- Stability endpoint tests ---


async def test_stability_not_found(client, embedding_set_id):
    """Stability for nonexistent job returns 404."""
    resp = await client.get("/clustering/jobs/nonexistent/stability")
    assert resp.status_code == 404


async def test_stability_not_complete(client, embedding_set_id):
    """Stability for a queued job returns 400."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/stability")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"].lower()


async def test_stability_no_file(client, embedding_set_id, app_settings):
    """Stability 404 when file doesn't exist for a complete job."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    resp = await client.get(f"/clustering/jobs/{job_id}/stability")
    assert resp.status_code == 404


async def test_stability_success(client, embedding_set_id, app_settings):
    """Stability endpoint returns stored stability data."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    # Write stability file
    cluster_path = Path(app_settings.storage_root) / "clusters" / job_id
    cluster_path.mkdir(parents=True, exist_ok=True)
    stability_data = {
        "n_runs": 3,
        "seeds": [42, 123, 456],
        "pairwise_label_agreement": {
            "mean_pairwise_ari": 0.9,
            "std_pairwise_ari": 0.02,
            "min_pairwise_ari": 0.85,
            "max_pairwise_ari": 0.95,
        },
        "aggregate_metrics": {
            "n_clusters_mean": 5.0,
            "n_clusters_std": 0.5,
            "n_clusters_min": 4.0,
            "n_clusters_max": 6.0,
        },
        "per_run": [
            {
                "run_index": 0,
                "seed": 42,
                "n_clusters": 5,
                "noise_fraction": 0.0,
                "silhouette_score": 0.5,
                "adjusted_rand_index": None,
                "normalized_mutual_info": None,
                "fragmentation_index": None,
            },
        ],
    }
    (cluster_path / "stability_summary.json").write_text(json.dumps(stability_data))

    resp = await client.get(f"/clustering/jobs/{job_id}/stability")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_runs"] == 3
    assert data["pairwise_label_agreement"]["mean_pairwise_ari"] == 0.9
    assert len(data["per_run"]) == 1


# --- Classifier endpoint tests ---


async def test_classifier_not_found(client, embedding_set_id):
    """Classifier for nonexistent job returns 404."""
    resp = await client.get("/clustering/jobs/nonexistent/classifier")
    assert resp.status_code == 404


async def test_classifier_not_complete(client, embedding_set_id):
    """Classifier for a queued job returns 400."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/classifier")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"].lower()


async def test_classifier_no_file(client, embedding_set_id, app_settings):
    """Classifier 404 when file doesn't exist for a complete job."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    resp = await client.get(f"/clustering/jobs/{job_id}/classifier")
    assert resp.status_code == 404


async def test_classifier_success(client, embedding_set_id, app_settings):
    """Classifier endpoint returns stored classifier report."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    # Write classifier report file
    cluster_path = Path(app_settings.storage_root) / "clusters" / job_id
    cluster_path.mkdir(parents=True, exist_ok=True)
    classifier_data = {
        "n_samples": 100,
        "n_categories": 3,
        "n_folds": 5,
        "categories_excluded": [],
        "overall_accuracy": 0.85,
        "per_class": {
            "Grunt": {
                "precision": 0.9,
                "recall": 0.85,
                "f1_score": 0.87,
                "support": 40,
            },
        },
        "macro_avg": {
            "precision": 0.84,
            "recall": 0.83,
            "f1_score": 0.83,
            "support": 100,
        },
        "weighted_avg": {
            "precision": 0.85,
            "recall": 0.85,
            "f1_score": 0.85,
            "support": 100,
        },
        "confusion_matrix": {"Grunt": {"Grunt": 34, "Buzz": 6}},
    }
    (cluster_path / "classifier_report.json").write_text(json.dumps(classifier_data))

    resp = await client.get(f"/clustering/jobs/{job_id}/classifier")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] == 100
    assert data["overall_accuracy"] == 0.85
    assert "Grunt" in data["per_class"]


async def test_label_queue_no_file(client, embedding_set_id, app_settings):
    """Label queue 404 when file doesn't exist for a complete job."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    resp = await client.get(f"/clustering/jobs/{job_id}/label-queue")
    assert resp.status_code == 404


async def test_label_queue_success(client, embedding_set_id, app_settings):
    """Label queue endpoint returns stored queue data."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    # Write label queue file
    cluster_path = Path(app_settings.storage_root) / "clusters" / job_id
    cluster_path.mkdir(parents=True, exist_ok=True)
    queue_data = [
        {
            "rank": 1,
            "global_index": 5,
            "embedding_set_id": "es-1",
            "embedding_row_index": 3,
            "current_category": None,
            "predicted_category": None,
            "entropy": None,
            "margin": None,
            "max_prob": None,
            "fragmentation_boost": 0.0,
            "priority": 1.0,
        },
        {
            "rank": 2,
            "global_index": 10,
            "embedding_set_id": "es-2",
            "embedding_row_index": 1,
            "current_category": "Grunt",
            "predicted_category": "Grunt",
            "entropy": 0.5,
            "margin": 0.3,
            "max_prob": 0.7,
            "fragmentation_boost": 0.1,
            "priority": 0.8,
        },
    ]
    (cluster_path / "label_queue.json").write_text(json.dumps(queue_data))

    resp = await client.get(f"/clustering/jobs/{job_id}/label-queue")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["rank"] == 1
    assert data[0]["priority"] == 1.0
    assert data[1]["current_category"] == "Grunt"


# --- Refinement endpoint tests ---


async def test_create_job_refined_from_invalid_id(client, embedding_set_id):
    """Creating a job with a non-existent refined_from_job_id should fail."""
    resp = await client.post(
        "/clustering/jobs",
        json={
            "embedding_set_ids": [embedding_set_id],
            "refined_from_job_id": "nonexistent-job-id",
        },
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


async def test_create_job_refined_from_incomplete(client, embedding_set_id):
    """Creating a job with a non-complete source job should fail."""
    # Create a queued job to use as source
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    source_id = create.json()["id"]

    resp = await client.post(
        "/clustering/jobs",
        json={
            "embedding_set_ids": [embedding_set_id],
            "refined_from_job_id": source_id,
        },
    )
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"].lower()


async def test_create_job_refined_from_no_parquet(
    client, embedding_set_id, app_settings
):
    """Creating a job from a complete source without refined parquet should fail."""
    # Create and mark complete (no refined_embeddings.parquet)
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    source_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, source_id)

    resp = await client.post(
        "/clustering/jobs",
        json={
            "embedding_set_ids": [embedding_set_id],
            "refined_from_job_id": source_id,
        },
    )
    assert resp.status_code == 400
    assert "no refined embeddings" in resp.json()["detail"].lower()


async def test_create_job_refined_from_valid(client, embedding_set_id, app_settings):
    """Creating a job from a complete source with refined parquet should succeed."""
    # Create and mark complete
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    source_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, source_id)

    # Write a refined_embeddings.parquet
    cluster_path = Path(app_settings.storage_root) / "clusters" / source_id
    cluster_path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "embedding_set_id": pa.array(["es1", "es1"], type=pa.string()),
            "embedding_row_index": pa.array([0, 1], type=pa.int32()),
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
    )
    pq.write_table(table, str(cluster_path / "refined_embeddings.parquet"))

    resp = await client.post(
        "/clustering/jobs",
        json={
            "embedding_set_ids": [embedding_set_id],
            "refined_from_job_id": source_id,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["refined_from_job_id"] == source_id
    assert data["status"] == "queued"


async def test_refinement_not_found(client, embedding_set_id):
    """Refinement for nonexistent job returns 404."""
    resp = await client.get("/clustering/jobs/nonexistent/refinement")
    assert resp.status_code == 404


async def test_refinement_not_complete(client, embedding_set_id):
    """Refinement for a queued job returns 400."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/refinement")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"].lower()


async def test_refinement_no_file(client, embedding_set_id, app_settings):
    """Refinement 404 when file doesn't exist for a complete job."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    resp = await client.get(f"/clustering/jobs/{job_id}/refinement")
    assert resp.status_code == 404


async def test_refinement_success(client, embedding_set_id, app_settings):
    """Refinement endpoint returns stored refinement report."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": [embedding_set_id]},
    )
    job_id = create.json()["id"]
    await _mark_job_complete_with_metrics(app_settings, job_id)

    # Write refinement report file
    cluster_path = Path(app_settings.storage_root) / "clusters" / job_id
    cluster_path.mkdir(parents=True, exist_ok=True)
    refinement_data = {
        "training_params": {
            "output_dim": 128,
            "hidden_dim": 512,
            "n_epochs": 50,
            "lr": 0.001,
            "margin": 1.0,
            "batch_size": 256,
            "mining_strategy": "semi-hard",
        },
        "n_labeled_samples": 100,
        "n_categories": 3,
        "n_total_samples": 120,
        "categories_used": ["Grunt", "Buzz", "Upsweep"],
        "loss_history": [0.95, 0.72, 0.51],
        "final_loss": 0.51,
        "comparison": [
            {
                "metric": "Silhouette Score",
                "key": "silhouette_score",
                "base": 0.32,
                "refined": 0.51,
                "delta": 0.19,
                "improved": True,
            },
        ],
        "base_summary": {"silhouette_score": 0.32},
        "refined_summary": {"silhouette_score": 0.51},
    }
    (cluster_path / "refinement_report.json").write_text(json.dumps(refinement_data))

    resp = await client.get(f"/clustering/jobs/{job_id}/refinement")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_labeled_samples"] == 100
    assert data["n_categories"] == 3
    assert len(data["comparison"]) == 1
    assert data["comparison"][0]["improved"] is True
    assert data["final_loss"] == 0.51
