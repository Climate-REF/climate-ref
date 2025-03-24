import importlib.resources

from alembic import context, op
from loguru import logger
from sqlalchemy import Connection, inspect

from cmip_ref.cli import capture_logging
from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.models import Base, MetricValue
from cmip_ref_core.pycmec.controlled_vocabulary import CV

# Setup logging
capture_logging()
logger.debug("Running alembic env")

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
ref_config = Config.default()

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


# Custom migration functions that are run on every migration


def _add_metric_value_columns(connection: Connection) -> None:
    """
    Add any missing columns in the current CV to the database

    This must be run in online mode

    Parameters
    ----------
    connection
        Open connection to the database
    """
    metric_table = "metric_value"

    inspector = inspect(connection)

    # Check if table already exists
    # Skip if it doesn't
    tables = inspector.get_table_names()
    if metric_table not in tables:
        logger.warning(f"No table named {metric_table!r} found")
        return

    # Extract the current columns in the DB
    existing_columns = [c["name"] for c in inspector.get_columns(metric_table)]

    # TODO: Load CV from configration
    cv_file = str(importlib.resources.files("cmip_ref_core.pycmec") / "cv_cmip_ar7ft.yaml")
    cv = CV.load_from_file(cv_file)

    for dimension in cv.dimensions:
        if dimension.name not in existing_columns:
            logger.info(f"Adding missing metric value dimension: {dimension.name!r}")

            op.add_column(metric_table, MetricValue.build_dimension_column(dimension))


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
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        db = Database.from_config(ref_config, run_migrations=False)
        connectable = db._engine

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)

        with context.begin_transaction():
            context.run_migrations()

            # Set up the Operations context
            # This is needed to alter the tables
            with op.Operations.context(context.get_context()):  # type: ignore
                _add_metric_value_columns(connection)


if context.is_offline_mode():
    logger.warning("Running in offline mode")
    run_migrations_offline()
else:
    run_migrations_online()
