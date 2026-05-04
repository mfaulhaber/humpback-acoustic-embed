"""Loader for CRNN region-based HMM jobs (ADR-057)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pyarrow.parquet as pq

from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.sequence_models.exemplars import WindowMeta
from humpback.sequence_models.overlay import OverlayMetadata
from humpback.storage import (
    continuous_embedding_parquet_path,
    hmm_sequence_pca_model_path,
)

if TYPE_CHECKING:
    from humpback.sequence_models.loaders import OverlayInputs


class CrnnRegionLoader:
    """Reads CRNN region-based parquets and builds :class:`OverlayInputs`.

    Sequences are grouped by ``region_id`` (string UUID) and ordered by
    each region's minimum ``start_timestamp`` ascending — matching the
    timeline navigation order the frontend uses for CRNN HMM jobs. Each
    :class:`WindowMeta` carries ``extras["tier"]`` so the frontend can
    badge exemplars with the chunk's tier classification.

    ``decoded_artifact_path`` is the explicit path to the decoded sequence
    parquet (post-ADR-061: ``decoded.parquet``). Masked-transformer jobs
    pass their per-k decoded path here.
    """

    def __init__(self, decoded_artifact_path: str) -> None:
        self.decoded_artifact_path = decoded_artifact_path

    def load(
        self,
        storage_root: Path,
        hmm_job: HMMSequenceJob,
        cej: ContinuousEmbeddingJob,
    ) -> "OverlayInputs":
        from humpback.sequence_models.loaders import OverlayInputs, read_decoded_parquet

        pca_model = joblib.load(hmm_sequence_pca_model_path(storage_root, hmm_job.id))
        emb_table = pq.read_table(
            continuous_embedding_parquet_path(storage_root, cej.id)
        )
        states_table = read_decoded_parquet(self.decoded_artifact_path)

        emb_region_vals = emb_table.column("region_id").to_pylist()
        emb_starts = emb_table.column("start_timestamp").to_pylist()

        min_start_per_region: dict[str, float] = {}
        for region_id, start in zip(emb_region_vals, emb_starts):
            existing = min_start_per_region.get(region_id)
            if existing is None or start < existing:
                min_start_per_region[region_id] = float(start)
        ordered_regions = sorted(
            min_start_per_region.keys(),
            key=lambda r: (min_start_per_region[r], r),
        )

        states_dict = states_table.to_pydict()
        states_by_region: dict[str, dict[int, dict[str, Any]]] = {}
        audio_file_by_region_chunk: dict[tuple[str, int], int | None] = {}
        tier_by_region_chunk: dict[tuple[str, int], str] = {}
        for i in range(states_table.num_rows):
            rid = states_dict["region_id"][i]
            cidx = states_dict["chunk_index_in_region"][i]
            states_by_region.setdefault(rid, {})[cidx] = {
                "viterbi_state": states_dict["label"][i],
                "max_state_probability": states_dict["max_state_probability"][i],
            }
            audio_file_by_region_chunk[(rid, cidx)] = states_dict["audio_file_id"][i]
            tier_by_region_chunk[(rid, cidx)] = states_dict["tier"][i]

        raw_sequences: list[np.ndarray] = []
        viterbi_states: list[np.ndarray] = []
        max_probs: list[np.ndarray] = []
        all_sequence_ids: list[str] = []
        all_positions: list[int] = []
        all_starts: list[float] = []
        all_ends: list[float] = []
        window_metas: list[WindowMeta] = []

        for region_id in ordered_regions:
            indices = [i for i, v in enumerate(emb_region_vals) if v == region_id]
            sub = emb_table.take(indices).sort_by("chunk_index_in_region")
            embeddings = np.array(sub.column("embedding").to_pylist(), dtype=np.float32)
            raw_sequences.append(embeddings)

            region_states = states_by_region.get(region_id, {})
            n_rows = sub.num_rows
            v_states = np.zeros(n_rows, dtype=np.int16)
            m_probs = np.zeros(n_rows, dtype=np.float32)
            for row_i in range(n_rows):
                cidx = sub.column("chunk_index_in_region")[row_i].as_py()
                st = region_states.get(cidx, {})
                prob = st.get("max_state_probability", 0.0)
                v_states[row_i] = st.get("viterbi_state", 0)
                m_probs[row_i] = prob
                start = sub.column("start_timestamp")[row_i].as_py()
                end = sub.column("end_timestamp")[row_i].as_py()
                tier = tier_by_region_chunk.get((region_id, cidx), "")
                all_sequence_ids.append(region_id)
                all_positions.append(cidx)
                all_starts.append(start)
                all_ends.append(end)
                window_metas.append(
                    WindowMeta(
                        sequence_id=region_id,
                        position_in_sequence=cidx,
                        audio_file_id=audio_file_by_region_chunk.get((region_id, cidx)),
                        start_timestamp=start,
                        end_timestamp=end,
                        max_state_probability=prob,
                        extras={"tier": tier},
                    )
                )

            viterbi_states.append(v_states)
            max_probs.append(m_probs)

        meta = OverlayMetadata(
            sequence_ids=all_sequence_ids,
            positions_in_sequence=all_positions,
            start_timestamps=all_starts,
            end_timestamps=all_ends,
        )
        return OverlayInputs(
            pca_model=pca_model,
            raw_sequences=raw_sequences,
            viterbi_states=viterbi_states,
            max_probs=max_probs,
            metadata=meta,
            window_metas=window_metas,
        )
