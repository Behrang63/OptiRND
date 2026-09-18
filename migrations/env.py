"""
Alembic migration environment for OptiRND.

Wires the SQLAlchemy metadata declared in core/database.py into Alembic.
Database URL resolution order:
    1. -x db_url=... on the command line
    2. OPTIRND_DB_URL environment variable
    3. fallback to the application default (sqlite:///pajoheshyar.db)
"""
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from core.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve target DB URL: -x db_url=... > env var > app default.
x_args = context.get_x_argument(as_dictionary=True)
DB_URL = (
    x_args.get("db_url")
    or os.environ.get("OPTIRND_DB_URL")
    or "sqlite:///pajoheshyar.db"
)
config.set_main_option("sqlalchemy.url", DB_URL)

# Single source of truth for schema: the ORM models themselves.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL scripts without a live DB connection."""
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,   # required for ALTER support on SQLite
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,   # required for ALTER support on SQLite
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
