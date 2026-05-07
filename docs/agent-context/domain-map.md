# Domain Map

Use this map before planning or implementing a task. Pick one primary domain
from the changed paths or task keywords, then add neighbor domains only for
direct dependencies.

## Path Routing

| Path or file | Primary domain | Common neighbors |
|---|---|---|
| `alembic/`, `alembic.ini` | `core-platform` | domain owning the changed table |
| `src/humpback/database.py` | `core-platform` | domain owning the changed model |
| `src/humpback/config.py` | `core-platform` | affected feature domain |
| `src/humpback/storage.py` | `core-platform` | domain owning the artifact root |
| `src/humpback/models/` | `core-platform` | domain named by model file |
| `src/humpback/schemas/` | `core-platform` | API/domain named by schema file |
| `src/humpback/workers/queue.py`, `runner.py` | `core-platform` | worker/domain named by claim/run function |
| `src/humpback/processing/` | `signal-timeline` | `ingest-detection`, `call-parsing`, `sequence-models` |
| `src/humpback/api/routers/timeline.py` | `signal-timeline` | `frontend-shell`, source feature domain |
| `frontend/src/components/timeline/` | `signal-timeline` | `frontend-shell`, consuming workspace domain |
| `src/humpback/classifier/` | `ingest-detection` | `signal-timeline`, `vocalization-clustering`, `call-parsing` |
| `src/humpback/api/routers/classifier/` | `ingest-detection` | `frontend-shell` |
| `src/humpback/services/classifier_service/` | `ingest-detection` | `core-platform` |
| `src/humpback/workers/classifier_worker/` | `ingest-detection` | `core-platform`, `signal-timeline` |
| `frontend/src/components/classifier/` | `ingest-detection` | `frontend-shell`, `signal-timeline` |
| `src/humpback/services/detection_embedding_service.py` | `ingest-detection` | `sequence-models`, `core-platform` |
| `src/humpback/api/routers/labeling.py` | `vocalization-clustering` | `ingest-detection`, `frontend-shell` |
| `src/humpback/api/routers/vocalization.py` | `vocalization-clustering` | `frontend-shell`, `core-platform` |
| `src/humpback/clustering/` | `vocalization-clustering` | `ingest-detection` |
| `src/humpback/services/vocalization_service.py` | `vocalization-clustering` | `core-platform` |
| `src/humpback/services/training_dataset.py` | `vocalization-clustering` | `ingest-detection` |
| `frontend/src/components/vocalization/` | `vocalization-clustering` | `frontend-shell`, `signal-timeline` |
| `src/humpback/call_parsing/` | `call-parsing` | `signal-timeline`, `vocalization-clustering`, `sequence-models` |
| `src/humpback/api/routers/call_parsing.py` | `call-parsing` | `frontend-shell`, `core-platform` |
| `src/humpback/services/call_parsing.py` | `call-parsing` | `core-platform` |
| `src/humpback/workers/*segmentation*`, `*classification*`, `region_detection_worker.py`, `window_classification_worker.py` | `call-parsing` | `signal-timeline`, `core-platform` |
| `frontend/src/components/call-parsing/` | `call-parsing` | `frontend-shell`, `signal-timeline` |
| `src/humpback/sequence_models/` | `sequence-models` | `call-parsing`, `signal-timeline` |
| `src/humpback/services/continuous_embedding_service.py` | `sequence-models` | `core-platform`, `call-parsing` |
| `src/humpback/workers/continuous_embedding_worker.py` | `sequence-models` | `core-platform`, `call-parsing`, `signal-timeline` |
| `src/humpback/api/routers/sequence_models.py` | `sequence-models` | `frontend-shell` |
| `frontend/src/components/sequence-models/` | `sequence-models` | `frontend-shell`, `signal-timeline` |
| `frontend/src/components/layout/`, `shared/`, `ui/` | `frontend-shell` | consuming feature domain |
| `frontend/src/api/`, `frontend/src/hooks/queries/` | `frontend-shell` | API/domain named by hook or client function |
| `docs/workflows/`, `AGENTS.md`, `CLAUDE.md` | `core-platform` | affected domain capsule |

## Keyword Routing

| Keyword | Primary domain | Common neighbors |
|---|---|---|
| Alembic, migration, schema, SQLAlchemy, settings, queue, worker polling | `core-platform` | feature domain |
| spectrogram, PCEN, playback, tile, timeline cache, UTC timeline | `signal-timeline` | consuming workspace domain |
| hydrophone, HLS, NOAA, extraction, detection job, classifier model, hyperparameter | `ingest-detection` | `signal-timeline` |
| labels, vocalization type, negative label, training dataset, clustering, UMAP | `vocalization-clustering` | `ingest-detection` |
| region detection, segmentation, classification, corrections, feedback training, window classify | `call-parsing` | `signal-timeline`, `vocalization-clustering` |
| continuous embedding, encoding signature, CRNN chunks, SurfPerch event windows | `sequence-models` | `call-parsing`, `signal-timeline` |
| navigation, side nav, breadcrumbs, shared component, React query hook | `frontend-shell` | feature domain |

## Neighbor Loading Rule

Load a neighbor domain when the task changes a boundary contract, not merely
because code imports from it. Examples:

- Sequence Models job creation that validates Pass 2 jobs should load
  `sequence-models` and `call-parsing`.
- Timeline overlay rendering inside Classify Review should load
  `signal-timeline` and `call-parsing`.
- A detection-embedding idempotency fix should load `ingest-detection` and
  `core-platform`.
