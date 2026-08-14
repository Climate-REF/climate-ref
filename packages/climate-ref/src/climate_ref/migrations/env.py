from alembic import context, op
from loguru import logger
from sqlalchemy import Connection, inspect

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Base
from climate_ref.models.execution import ExecutionOutput
from climate_ref.models.metric_value import MetricValue
from climate_ref.models.mixins import DimensionMixin
from climate_ref_core.logging import capture_logging
from climate_ref_core.pycmec.controlled_vocabulary import CV

try:
    import alembic_postgresql_enum  # noqa
except ImportError:
    logger.warning("alembic_postgresql_enum not installed, skipping enum migration support")

# Setup logging
capture_logging()
logger.debug("Running alembic env")

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
ref_config: Config = config.attributes.get("ref_config") or Config.default()

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


# Custom migration functions that are run on every migration


def _add_dimension_columns(connection: Connection, table: str, Cls: type[DimensionMixin]) -> None:
    """
    Add any missing columns in the current CV to the database

    This must be run in online mode

    Parameters
    ----------
    connection
        Open connection to the database
    """
    inspector = inspect(connection)

    # Check if table already exists
    # Skip if it doesn't
    tables = inspector.get_table_names()
    if table not in tables:
        logger.warning(f"No table named {table!r} found")
        return

    # Extract the current columns in the DB
    existing_columns = [c["name"] for c in inspector.get_columns(table)]

    cv = CV.load(ref_config.paths.dimensions_cv_resource)

    for dimension in cv.dimensions:
        if dimension.name not in existing_columns:
            logger.info(f"Adding missing value dimension: {dimension.name!r}")
            op.add_column(table, Cls.build_dimension_column(dimension))


def include_object(object_, name: str, type_, reflected, compare_to) -> bool:
    """
    Object-level check to include or exclude objects from the migration

    Excludes columns that are marked with `skip_autogenerate` in the info dict

    Based on  https://alembic.sqlalchemy.org/en/latest/autogenerate.html#omitting-based-on-object
    """
    if object_.info.get("skip_autogenerate", False):
        return False
    else:
        return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_name="sqlite",
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = config.attributes.get("connection", None)

    # A database opened here is ours to close.
    db = None
    if connectable is None:
        db = Database.from_config(ref_config, run_migrations=False)
        connectable = db._engine

    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=True,
                include_object=include_object,
            )

            with context.begin_transaction():
                context.run_migrations()

                # Set up the Operations context
                # This is needed to alter the tables
                with op.Operations.context(context.get_context()):  # type: ignore
                    _add_dimension_columns(connection, "metric_value", MetricValue)
                    _add_dimension_columns(connection, "execution_output", ExecutionOutput)
    finally:
        if db is not None:
            db.close()


if context.is_offline_mode():
    logger.warning("Running in offline mode")
    run_migrations_offline()
else:
    run_migrations_online()
