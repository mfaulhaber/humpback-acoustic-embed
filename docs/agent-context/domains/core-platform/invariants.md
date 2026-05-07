# Core Platform Invariants

- Use `uv` for Python package operations and Python tools.
- Use Alembic for schema changes; SQLite migrations use `op.batch_alter_table()`.
- Back up the production database before migrations, schema changes, data
  backfills, manual SQL, or destructive deletes.
- SQL worker claims rely on atomic status updates because SQLite has no true
  row-level locks.
- Worker status transitions remain `queued -> running -> complete`,
  `queued -> running -> failed`, or `queued -> canceled`.
- Stale job recovery must not duplicate canonical derived artifacts.
- Storage helpers should produce paths under `storage_root`; destructive
  cleanup must verify path containment.
- Global instructions stay small. Put domain-specific rules in the matching
  `docs/agent-context/domains/*/` capsule or detailed `docs/reference/` file.
