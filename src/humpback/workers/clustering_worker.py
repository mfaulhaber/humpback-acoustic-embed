"""Clustering worker: load embeddings → reduce → cluster → persist."""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.clustering.pipeline import compute_cluster_sizes, run_clustering_pipeline
from humpback.config import Settings
from humpback.models.clustering import Cluster, ClusterAssignment, ClusteringJob
from humpback.models.processing import EmbeddingSet
from humpback.processing.embeddings import read_embeddings
from humpback.storage import cluster_dir, ensure_dir
from humpback.workers.queue import complete_clustering_job, fail_clustering_job

logger = logging.getLogger(__name__)


async def run_clustering_job(
    session: AsyncSession,
    job: ClusteringJob,
    settings: Settings,
) -> None:
    """Execute a clustering job end-to-end."""
    try:
        es_ids = json.loads(job.embedding_set_ids)
        params = json.loads(job.parameters) if job.parameters else None

        # Load all embedding sets
        all_embeddings = []
        all_es_ids = []
        all_row_indices = []

        for es_id in es_ids:
            result = await session.execute(
                select(EmbeddingSet).where(EmbeddingSet.id == es_id)
            )
            es = result.scalar_one_or_none()
            if es is None:
                raise ValueError(f"Embedding set {es_id} not found")

            indices, embeddings = await asyncio.to_thread(
                read_embeddings, Path(es.parquet_path)
            )
            for i, idx in enumerate(indices):
                all_embeddings.append(embeddings[i])
                all_es_ids.append(es_id)
                all_row_indices.append(int(idx))

        if not all_embeddings:
            raise ValueError("No embeddings found")

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Run clustering in thread
        labels, reduced = await asyncio.to_thread(
            run_clustering_pipeline, embeddings_array, params
        )

        # Compute cluster sizes
        sizes = compute_cluster_sizes(labels)

        # Write UMAP coordinates if dimensionality reduction was performed
        output_dir = ensure_dir(cluster_dir(settings.storage_root, job.id))
        if reduced is not None:
            umap_table = pa.table({
                "x": pa.array(reduced[:, 0], type=pa.float32()),
                "y": pa.array(reduced[:, 1], type=pa.float32()),
                "cluster_label": pa.array([int(l) for l in labels], type=pa.int32()),
                "embedding_set_id": pa.array(all_es_ids, type=pa.string()),
                "embedding_row_index": pa.array(all_row_indices, type=pa.int32()),
            })
            pq.write_table(umap_table, str(output_dir / "umap_coords.parquet"))

        # Persist clusters and assignments

        cluster_map: dict[int, Cluster] = {}
        for label, size in sizes.items():
            c = Cluster(
                clustering_job_id=job.id,
                cluster_label=int(label),
                size=size,
            )
            session.add(c)
            cluster_map[label] = c

        await session.flush()

        # Create assignments
        assignments_data = []
        for i, label in enumerate(labels):
            label_int = int(label)
            cluster = cluster_map[label_int]
            assignment = ClusterAssignment(
                cluster_id=cluster.id,
                embedding_set_id=all_es_ids[i],
                embedding_row_index=all_row_indices[i],
            )
            session.add(assignment)
            assignments_data.append({
                "cluster_id": cluster.id,
                "cluster_label": label_int,
                "embedding_set_id": all_es_ids[i],
                "embedding_row_index": all_row_indices[i],
            })

        # Write output files
        clusters_json = {
            "job_id": job.id,
            "n_clusters": len(cluster_map),
            "clusters": [
                {"label": int(label), "size": size, "id": c.id}
                for label, (size, c) in zip(
                    sizes.keys(),
                    zip(sizes.values(), [cluster_map[l] for l in sizes.keys()]),
                )
            ],
        }
        (output_dir / "clusters.json").write_text(json.dumps(clusters_json, indent=2))

        # Write assignments parquet
        if assignments_data:
            table = pa.table({
                "cluster_id": [a["cluster_id"] for a in assignments_data],
                "cluster_label": [a["cluster_label"] for a in assignments_data],
                "embedding_set_id": [a["embedding_set_id"] for a in assignments_data],
                "embedding_row_index": [a["embedding_row_index"] for a in assignments_data],
            })
            pq.write_table(table, str(output_dir / "assignments.parquet"))

        await complete_clustering_job(session, job.id)

    except Exception as e:
        logger.exception(f"Clustering job {job.id} failed")
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_clustering_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark clustering job as failed")
