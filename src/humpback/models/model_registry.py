from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class ModelConfig(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "model_configs"

    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    vector_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=1280)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_type: Mapped[str] = mapped_column(
        String, nullable=False, default="tflite", server_default="tflite"
    )
    input_format: Mapped[str] = mapped_column(
        String, nullable=False, default="spectrogram", server_default="spectrogram"
    )


# Backward-compatible alias
TFLiteModelConfig = ModelConfig
