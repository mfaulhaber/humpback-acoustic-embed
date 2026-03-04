"""Shared model cache for workers — avoids loading duplicate model instances."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.processing.inference import EmbeddingModel, FakeTF2Model, FakeTFLiteModel
from humpback.services.model_registry_service import get_model_by_name

logger = logging.getLogger(__name__)

_model_cache: dict[str, EmbeddingModel] = {}


async def get_model_by_version(
    session: AsyncSession,
    model_version: str,
    settings: Settings,
) -> tuple[EmbeddingModel, str]:
    """Load model by version name, using registry and cache.

    Returns (model, input_format).
    """
    model_config = await get_model_by_name(session, model_version)
    input_format = model_config.input_format if model_config else "spectrogram"
    model_type = model_config.model_type if model_config else "tflite"

    if model_version in _model_cache:
        logger.info(
            "Using cached model for %s (type=%s)",
            model_version, type(_model_cache[model_version]).__name__,
        )
        return _model_cache[model_version], input_format

    if not settings.use_real_model:
        vector_dim = model_config.vector_dim if model_config else settings.vector_dim
        if input_format == "waveform":
            model = FakeTF2Model(vector_dim)
        else:
            model = FakeTFLiteModel(vector_dim)
        _model_cache[model_version] = model
        return model, input_format

    # Real model
    if model_config:
        if model_type == "tf2_saved_model":
            from humpback.processing.inference import TF2SavedModel

            logger.info("Loading TF2SavedModel: path=%s, dim=%d", model_config.path, model_config.vector_dim)
            model = TF2SavedModel(model_config.path, model_config.vector_dim, force_cpu=settings.tf_force_cpu)
        else:
            from humpback.processing.inference import TFLiteModel

            logger.info("Loading TFLiteModel: path=%s, dim=%d", model_config.path, model_config.vector_dim)
            model = TFLiteModel(model_config.path, model_config.vector_dim)
    else:
        from humpback.processing.inference import TFLiteModel

        model = TFLiteModel(settings.model_path, settings.vector_dim)

    _model_cache[model_version] = model
    return model, input_format
