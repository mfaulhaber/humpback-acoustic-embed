"""Static checks for the masked-transformer contrastive sampler migration."""

from __future__ import annotations

from pathlib import Path


def test_contrastive_sampler_migration_defaults_and_downgrade() -> None:
    path = (
        Path(__file__).parents[2]
        / "alembic"
        / "versions"
        / "071_masked_transformer_contrastive_sampler.py"
    )
    source = path.read_text(encoding="utf-8")

    assert 'revision = "071"' in source
    assert 'down_revision = "070"' in source
    assert 'op.batch_alter_table("masked_transformer_jobs")' in source
    assert '"contrastive_sampler_enabled"' in source
    assert '"contrastive_labels_per_batch"' in source
    assert '"contrastive_events_per_label"' in source
    assert '"contrastive_max_unlabeled_fraction"' in source
    assert '"contrastive_region_balance"' in source
    assert "server_default=sa.true()" in source
    assert 'server_default="4"' in source
    assert 'server_default="0.25"' in source
    assert 'batch.drop_column("contrastive_region_balance")' in source
    assert 'batch.drop_column("contrastive_sampler_enabled")' in source
