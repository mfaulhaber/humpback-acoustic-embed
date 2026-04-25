"""Clustering worker: load embeddings -> reduce -> cluster -> persist."""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.clustering.metrics import (
    compute_cluster_metrics,
    compute_detailed_category_metrics,
    compute_fragmentation_report,
    extract_category_from_folder_path,
    run_parameter_sweep,
)
from humpback.clustering.pipeline import (
    ClusteringResult,
    compute_cluster_sizes,
    run_clustering_pipeline,
)
from humpback.config import Settings
from humpback.models.audio import AudioFile
from humpback.models.clustering import Cluster, ClusterAssignment, ClusteringJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.models.processing import EmbeddingSet
from humpback.processing.embeddings import read_embeddings
from humpback.storage import cluster_dir, detection_embeddings_path, ensure_dir
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
        det_ids = json.loads(job.detection_job_ids) if job.detection_job_ids else []
        params = json.loads(job.parameters) if job.parameters else None

        all_embeddings: list[np.ndarray] = []
        all_es_ids: list[str] = []
        all_row_indices: list[int] = []
        es_folder_paths: dict[str, str] = {}

        if det_ids:
            for dj_id in det_ids:
                emb_result = await session.execute(
                    select(DetectionEmbeddingJob)
                    .where(
                        DetectionEmbeddingJob.detection_job_id == dj_id,
                        DetectionEmbeddingJob.status == "complete",
                    )
                    .order_by(DetectionEmbeddingJob.created_at.desc())
                    .limit(1)
                )
                emb_job = emb_result.scalar_one_or_none()
                if emb_job is None:
                    raise ValueError(
                        f"No completed embedding job for detection job {dj_id}"
                    )

                parquet_path = detection_embeddings_path(
                    settings.storage_root, dj_id, emb_job.model_version
                )
                if not parquet_path.exists():
                    raise ValueError(
                        f"Embeddings parquet not found for detection job {dj_id}"
                    )

                det_table = await asyncio.to_thread(pq.read_table, str(parquet_path))
                row_ids = det_table.column("row_id").to_pylist()
                emb_col = det_table.column("embedding").to_pylist()
                for i, rid in enumerate(row_ids):
                    all_embeddings.append(np.array(emb_col[i], dtype=np.float32))
                    all_es_ids.append(dj_id)
                    all_row_indices.append(i)

                logger.info(
                    "Loaded %d detection embeddings for job %s",
                    len(row_ids),
                    dj_id,
                )
        else:
            for es_id in es_ids:
                result = await session.execute(
                    select(EmbeddingSet).where(EmbeddingSet.id == es_id)
                )
                es = result.scalar_one_or_none()
                if es is None:
                    raise ValueError(f"Embedding set {es_id} not found")

                if es_id not in es_folder_paths:
                    audio_result = await session.execute(
                        select(AudioFile).where(AudioFile.id == es.audio_file_id)
                    )
                    audio_file = audio_result.scalar_one_or_none()
                    es_folder_paths[es_id] = (
                        audio_file.folder_path if audio_file else ""
                    )

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
        category_labels: list[str | None] = []

        # If re-clustering from refined embeddings, replace the array
        if job.refined_from_job_id:
            source_dir = cluster_dir(settings.storage_root, job.refined_from_job_id)
            refined_path = source_dir / "refined_embeddings.parquet"
            if not refined_path.exists():
                raise ValueError(
                    f"Refined embeddings not found for source job {job.refined_from_job_id}"
                )
            refined_table = pq.read_table(str(refined_path))
            refined_es_ids = refined_table.column("embedding_set_id").to_pylist()
            refined_row_indices = refined_table.column(
                "embedding_row_index"
            ).to_pylist()
            refined_vectors = refined_table.column("embedding").to_pylist()
            embeddings_array = np.array(refined_vectors, dtype=np.float32)
            all_es_ids = refined_es_ids
            all_row_indices = refined_row_indices
            logger.info(
                "Loaded %d refined embeddings from job %s",
                len(embeddings_array),
                job.refined_from_job_id,
            )

        # Run clustering in thread
        clustering_result: ClusteringResult = await asyncio.to_thread(
            run_clustering_pipeline, embeddings_array, params
        )
        labels = clustering_result.labels
        reduced = clustering_result.reduced_embeddings

        # Compute cluster sizes
        sizes = compute_cluster_sizes(labels)

        # Write UMAP coordinates if dimensionality reduction was performed
        output_dir = ensure_dir(cluster_dir(settings.storage_root, job.id))
        if reduced is not None:
            umap_table = pa.table(
                {
                    "x": pa.array(reduced[:, 0], type=pa.float32()),
                    "y": pa.array(reduced[:, 1], type=pa.float32()),
                    "cluster_label": pa.array(
                        [int(lb) for lb in labels], type=pa.int32()
                    ),
                    "embedding_set_id": pa.array(all_es_ids, type=pa.string()),
                    "embedding_row_index": pa.array(all_row_indices, type=pa.int32()),
                }
            )
            pq.write_table(umap_table, str(output_dir / "umap_coords.parquet"))

        # --- Compute evaluation metrics ---
        metrics: dict = {}

        # Internal metrics (silhouette, davies-bouldin, calinski-harabasz)
        try:
            internal = await asyncio.to_thread(
                compute_cluster_metrics, clustering_result.cluster_input, labels
            )
            metrics.update(internal)
        except Exception:
            logger.exception("Failed to compute internal cluster metrics")

        # Category-based supervised metrics (detailed)
        try:
            if det_ids:
                category_labels = await _resolve_vocalization_labels(
                    session, settings, all_es_ids, all_row_indices
                )
            else:
                category_labels = [
                    extract_category_from_folder_path(es_folder_paths.get(es_id, ""))
                    for es_id in all_es_ids
                ]
            if any(c is not None for c in category_labels):
                cat_metrics = compute_detailed_category_metrics(labels, category_labels)
                metrics.update(cat_metrics)
        except Exception:
            logger.exception("Failed to compute category metrics")

        # Parameter sweep (with category labels for ARI/NMI)
        try:
            sweep = await asyncio.to_thread(
                run_parameter_sweep,
                clustering_result.cluster_input,
                params,
                category_labels,
            )
            sweep_path = output_dir / "parameter_sweep.json"
            sweep_path.write_text(json.dumps(sweep, indent=2))
        except Exception:
            logger.exception("Failed to run parameter sweep")

        # Fragmentation report
        frag_report = None
        try:
            frag_report = compute_fragmentation_report(labels, category_labels, job.id)
            if frag_report is not None:
                report_path = output_dir / "report.json"
                report_path.write_text(json.dumps(frag_report, indent=2))
        except Exception:
            logger.exception("Failed to compute fragmentation report")

        # Classifier baseline (opt-in via run_classifier parameter)
        try:
            if (params or {}).get("run_classifier", False) and any(
                c is not None for c in category_labels
            ):
                from humpback.clustering.classifier import run_classifier_baseline

                classifier_result = await asyncio.to_thread(
                    run_classifier_baseline,
                    embeddings_array,
                    category_labels,
                    frag_report,
                    all_es_ids,
                    all_row_indices,
                )
                if classifier_result is not None:
                    (output_dir / "classifier_report.json").write_text(
                        json.dumps(classifier_result["classifier_report"], indent=2)
                    )
                    (output_dir / "label_queue.json").write_text(
                        json.dumps(classifier_result["label_queue"], indent=2)
                    )
        except Exception:
            logger.exception("Failed to run classifier baseline")

        # Stability evaluation (opt-in via stability_runs parameter)
        try:
            stability_runs = (params or {}).get("stability_runs", 0)
            if isinstance(stability_runs, int) and stability_runs >= 2:
                from humpback.clustering.stability import run_stability_evaluation

                stability_result = await asyncio.to_thread(
                    run_stability_evaluation,
                    embeddings_array,
                    params,
                    category_labels,
                    stability_runs,
                )
                stability_path = output_dir / "stability_summary.json"
                stability_path.write_text(json.dumps(stability_result, indent=2))
        except Exception:
            logger.exception("Failed to run stability evaluation")

        # Metric learning refinement (opt-in via enable_metric_learning parameter)
        try:
            if (params or {}).get("enable_metric_learning", False) and any(
                c is not None for c in category_labels
            ):
                from humpback.clustering.metric_learning import (
                    run_metric_learning_refinement,
                )

                refinement_result = await asyncio.to_thread(
                    run_metric_learning_refinement,
                    embeddings_array,
                    category_labels,
                    frag_report,
                    params,
                )
                if refinement_result is not None:
                    # Pop and persist refined embeddings before JSON serialization
                    refined_emb = refinement_result.pop("_refined_embeddings", None)
                    (output_dir / "refinement_report.json").write_text(
                        json.dumps(refinement_result, indent=2)
                    )
                    if refined_emb is not None:
                        refined_table = pa.table(
                            {
                                "embedding_set_id": pa.array(
                                    all_es_ids, type=pa.string()
                                ),
                                "embedding_row_index": pa.array(
                                    all_row_indices, type=pa.int32()
                                ),
                                "embedding": [
                                    refined_emb[i].tolist()
                                    for i in range(len(refined_emb))
                                ],
                            }
                        )
                        pq.write_table(
                            refined_table,
                            str(output_dir / "refined_embeddings.parquet"),
                        )
                        logger.info(
                            "Saved refined embeddings (%d rows) to %s",
                            len(refined_emb),
                            output_dir / "refined_embeddings.parquet",
                        )
        except Exception:
            logger.exception("Failed to run metric learning refinement")

        # Persist metrics via explicit UPDATE (job object is detached from this session)
        if metrics:
            await session.execute(
                update(ClusteringJob)
                .where(ClusteringJob.id == job.id)
                .values(metrics_json=json.dumps(metrics))
            )

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
            assignments_data.append(
                {
                    "cluster_id": cluster.id,
                    "cluster_label": label_int,
                    "embedding_set_id": all_es_ids[i],
                    "embedding_row_index": all_row_indices[i],
                }
            )

        # Write output files
        clusters_json = {
            "job_id": job.id,
            "n_clusters": len(cluster_map),
            "clusters": [
                {"label": int(label), "size": size, "id": c.id}
                for label, (size, c) in zip(
                    sizes.keys(),
                    zip(sizes.values(), [cluster_map[lb] for lb in sizes.keys()]),
                )
            ],
        }
        (output_dir / "clusters.json").write_text(json.dumps(clusters_json, indent=2))

        # Write assignments parquet
        if assignments_data:
            table = pa.table(
                {
                    "cluster_id": [a["cluster_id"] for a in assignments_data],
                    "cluster_label": [a["cluster_label"] for a in assignments_data],
                    "embedding_set_id": [
                        a["embedding_set_id"] for a in assignments_data
                    ],
                    "embedding_row_index": [
                        a["embedding_row_index"] for a in assignments_data
                    ],
                }
            )
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


async def _resolve_vocalization_labels(
    session: AsyncSession,
    settings: Settings,
    all_es_ids: list[str],
    all_row_indices: list[int],
) -> list[str | None]:
    """Resolve vocalization inference labels for detection-based clustering points."""
    from humpback.models.vocalization import (
        VocalizationClassifierModel,
        VocalizationInferenceJob,
    )

    active_result = await session.execute(
        select(VocalizationClassifierModel).where(
            VocalizationClassifierModel.is_active.is_(True)
        )
    )
    active_model = active_result.scalar_one_or_none()
    if active_model is None:
        return [None] * len(all_es_ids)

    vocabulary: list[str] = json.loads(active_model.vocabulary_snapshot)
    thresholds: dict[str, float] = json.loads(active_model.per_class_thresholds)

    voc_label_map: dict[tuple[str, int], str] = {}

    unique_dj_ids = list(set(all_es_ids))
    for dj_id in unique_dj_ids:
        emb_result = await session.execute(
            select(DetectionEmbeddingJob)
            .where(
                DetectionEmbeddingJob.detection_job_id == dj_id,
                DetectionEmbeddingJob.status == "complete",
            )
            .order_by(DetectionEmbeddingJob.created_at.desc())
            .limit(1)
        )
        emb_job = emb_result.scalar_one_or_none()
        if emb_job is None:
            continue

        emb_parquet = detection_embeddings_path(
            settings.storage_root, dj_id, emb_job.model_version
        )
        if not emb_parquet.exists():
            continue

        emb_table = pq.read_table(str(emb_parquet), columns=["row_id"])
        ordered_row_ids = emb_table.column("row_id").to_pylist()

        inf_result = await session.execute(
            select(VocalizationInferenceJob).where(
                VocalizationInferenceJob.source_type == "detection_job",
                VocalizationInferenceJob.source_id == dj_id,
                VocalizationInferenceJob.vocalization_model_id == active_model.id,
                VocalizationInferenceJob.status == "complete",
            )
        )
        inf_job = inf_result.scalar_one_or_none()
        if inf_job is None or not inf_job.output_path:
            continue

        pred_path = Path(inf_job.output_path)
        if not pred_path.exists():
            continue

        pred_table = pq.read_table(str(pred_path))
        pred_cols = set(pred_table.column_names)
        pred_row_ids = (
            pred_table.column("row_id").to_pylist() if "row_id" in pred_cols else []
        )

        row_id_to_label: dict[str, str] = {}
        for pi in range(pred_table.num_rows):
            rid = pred_row_ids[pi] if pred_row_ids else str(pi)
            best_type = ""
            best_score = -1.0
            for type_name in vocabulary:
                if type_name not in pred_cols:
                    continue
                score = float(pred_table.column(type_name)[pi].as_py())
                t = thresholds.get(type_name, 0.5)
                if score >= t and score > best_score:
                    best_score = score
                    best_type = type_name
            if best_type:
                row_id_to_label[rid] = best_type

        for idx, rid in enumerate(ordered_row_ids):
            label = row_id_to_label.get(rid)
            if label:
                voc_label_map[(dj_id, idx)] = label

    return [
        voc_label_map.get((all_es_ids[i], all_row_indices[i]))
        for i in range(len(all_es_ids))
    ]
