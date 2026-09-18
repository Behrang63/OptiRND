# Alembic Migrations — OptiRND

Schema migrations replace the old `Base.metadata.create_all()`-only approach.
`init_db()` still creates tables for fresh in-memory/test databases, but any
**file-backed** database must now be versioned through Alembic.

## Database URL resolution (highest priority first)

1. CLI: `alembic -x db_url=sqlite:///path/to.db upgrade head`
2. Environment: `OPTIRND_DB_URL=... alembic upgrade head`
3. Default: `sqlite:///pajoheshyar.db`

## Common commands

```bash
# Apply all pending migrations
alembic upgrade head

# Check current revision
alembic current

# After changing an ORM model in core/database.py - autogenerate a migration,
# REVIEW IT, then apply:
alembic revision --autogenerate -m "describe the change"
alembic upgrade head

# Roll back one revision
alembic downgrade -1

# Full teardown (dev only)
alembic downgrade base
```

## Conventions

- SQLite batch mode (`render_as_batch=True`) is enabled in `env.py` — required
  for ALTER operations on SQLite.
- The migration baseline (`7f3a9c1d5e2b`) mirrors the Phase 1 ORM schema
  exactly; verify with `alembic check` or an empty autogenerate diff.
- Never edit an applied migration; always add a new revision.
