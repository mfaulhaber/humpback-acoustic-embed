"""Seed the perch_v2 ModelConfig row idempotently.

Inserts a ``model_configs`` row describing the locally-trained Perch v2
TFLite model (waveform input, 1536-d). Idempotent on re-run.

Revision ID: 051
Revises: 050
Create Date: 2026-04-17
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

revision = "051"
down_revision = "050"
branch_labels = None
depends_on = None

_PERCH_V2_NAME = "perch_v2"


def upgrade() -> None:
    bind = op.get_bind()

    existing = bind.execute(
        sa.text("SELECT id FROM model_configs WHERE name = :n"),
        {"n": _PERCH_V2_NAME},
    ).fetchone()
    if existing is not None:
        return

    now = datetime.now(timezone.utc).isoformat()
    bind.execute(
        sa.text(
            "INSERT INTO model_configs "
            "(id, name, display_name, path, vector_dim, description, is_default, "
            " model_type, input_format, created_at, updated_at) "
            "VALUES (:id, :name, :display_name, :path, :vector_dim, :description, "
            "        :is_default, :model_type, :input_format, :created_at, :updated_at)"
        ),
        {
            "id": str(uuid.uuid4()),
            "name": _PERCH_V2_NAME,
            "display_name": "Perch v2 (TFLite)",
            "path": "models/perch_v2.tflite",
            "vector_dim": 1536,
            "description": None,
            "is_default": False,
            "model_type": "tflite",
            "input_format": "waveform",
            "created_at": now,
            "updated_at": now,
        },
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text("DELETE FROM model_configs WHERE name = :n"),
        {"n": _PERCH_V2_NAME},
    )
