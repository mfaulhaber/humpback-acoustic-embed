"""Loader for SurfPerch event-padded HMM jobs (ADR-056)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.sequence_models.exemplars import WindowMeta
from humpback.sequence_models.overlay import OverlayMetadata
from humpback.storage import (
    continuous_embedding_parquet_path,
    hmm_sequence_pca_model_path,
)

if TYPE_CHECKING:
    from humpback.sequence_models.loaders import (
        LabelDistributionInputs,
        OverlayInputs,
    )


class SurfPerchLoader:
    """Reads SurfPerch parquets and builds the generic :class:`OverlayInputs`.

    Sequences are grouped by ``merged_span_id`` (sorted ascending), which
    matches the historical ordering ``_load_overlay_inputs()`` produced.
    The unified identifier pair stringifies the int span id so downstream
    code is source-agnostic. ``extras`` is empty for SurfPerch rows.

    ``decoded_artifact_path`` is the explicit path to the decoded sequence
    parquet (post-ADR-061: ``decoded.parquet``). At v1, only the HMM path
    exercises this loader; the masked-transformer track does not yet have
    a SurfPerch source variant.
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

        emb_span_vals = emb_table.column("merged_span_id").to_pylist()
        unique_spans = sorted(set(emb_span_vals))

        states_dict = states_table.to_pydict()
        states_by_span: dict[int, dict[int, dict[str, Any]]] = {}
        audio_file_by_span_win: dict[tuple[int, int], int] = {}
        for i in range(states_table.num_rows):
            sid = states_dict["merged_span_id"][i]
            widx = states_dict["window_index_in_span"][i]
            states_by_span.setdefault(sid, {})[widx] = {
                "viterbi_state": states_dict["label"][i],
                "max_state_probability": states_dict["max_state_probability"][i],
            }
            audio_file_by_span_win[(sid, widx)] = states_dict["audio_file_id"][i]

        raw_sequences: list[np.ndarray] = []
        viterbi_states: list[np.ndarray] = []
        max_probs: list[np.ndarray] = []
        all_sequence_ids: list[str] = []
        all_positions: list[int] = []
        all_starts: list[float] = []
        all_ends: list[float] = []
        window_metas: list[WindowMeta] = []

        for span_id in unique_spans:
            indices = [i for i, v in enumerate(emb_span_vals) if v == span_id]
            sub = emb_table.take(indices).sort_by("window_index_in_span")
            embeddings = np.array(sub.column("embedding").to_pylist(), dtype=np.float32)
            raw_sequences.append(embeddings)

            span_states = states_by_span.get(span_id, {})
            n_rows = sub.num_rows
            v_states = np.zeros(n_rows, dtype=np.int16)
            m_probs = np.zeros(n_rows, dtype=np.float32)
            for row_i in range(n_rows):
                widx = sub.column("window_index_in_span")[row_i].as_py()
                st = span_states.get(widx, {})
                prob = st.get("max_state_probability", 0.0)
                v_states[row_i] = st.get("viterbi_state", 0)
                m_probs[row_i] = prob
                start = sub.column("start_timestamp")[row_i].as_py()
                end = sub.column("end_timestamp")[row_i].as_py()
                sequence_id = str(span_id)
                all_sequence_ids.append(sequence_id)
                all_positions.append(widx)
                all_starts.append(start)
                all_ends.append(end)
                window_metas.append(
                    WindowMeta(
                        sequence_id=sequence_id,
                        position_in_sequence=widx,
                        audio_file_id=audio_file_by_span_win.get((span_id, widx), 0),
                        start_timestamp=start,
                        end_timestamp=end,
                        max_state_probability=prob,
                        extras={},
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

    async def load_label_distribution_inputs(
        self,
        session: AsyncSession,
        storage_root: Path,
        hmm_job: HMMSequenceJob,
        cej: ContinuousEmbeddingJob,
    ) -> "LabelDistributionInputs":
        from humpback.models.call_parsing import (
            EventSegmentationJob,
            RegionDetectionJob,
        )
        from humpback.sequence_models.loaders import (
            LabelDistributionInputs,
            read_decoded_parquet,
        )

        seg_job = await session.get(EventSegmentationJob, cej.event_segmentation_job_id)
        if seg_job is None:
            raise ValueError(
                f"EventSegmentationJob not found: {cej.event_segmentation_job_id}"
            )
        rdj = await session.get(RegionDetectionJob, seg_job.region_detection_job_id)
        if rdj is None:
            raise ValueError(
                f"RegionDetectionJob not found: {seg_job.region_detection_job_id}"
            )

        states_table = read_decoded_parquet(self.decoded_artifact_path)
        state_rows: list[dict[str, Any]] = []
        for i in range(states_table.num_rows):
            state_rows.append(
                {
                    "start_timestamp": float(
                        states_table.column("start_timestamp")[i].as_py()
                    ),
                    "end_timestamp": float(
                        states_table.column("end_timestamp")[i].as_py()
                    ),
                    "viterbi_state": states_table.column("label")[i].as_py(),
                }
            )

        return LabelDistributionInputs(
            hydrophone_id=rdj.hydrophone_id,
            state_rows=state_rows,
            tier_per_row=None,
        )
