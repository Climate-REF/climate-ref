"""Database adapter layer

This module provides a database adapter layer that abstracts the database connection and migrations.
This allows us to easily switch between different database backends,
and to run migrations when the database is loaded.

The `Database` class is the main entry point for interacting with the database.
It provides a session object that can be used to interact with the database and run queries.
"""

import enum
import importlib.resources
import shutil
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib import parse as urlparse

import alembic.command
import sqlalchemy
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from loguru import logger
from sqlalchemy.orm import Session

from climate_ref.models import MetricValue, Table
from climate_ref.models.execution import ExecutionOutput
from climate_ref_core.pycmec.controlled_vocabulary import CV

if TYPE_CHECKING:
    from climate_ref.config import Config

_REMOVED_REVISIONS = [
    "ea2aa1134cb3",
    "4b95a617184e",
    "4a447fbf6d65",
    "c1818a18d87f",
    "6634396f139a",
    "1f5969a92b85",
    "c5de99c14533",
    "e1cdda7dcf1d",
    "904f2f2db24a",
    "6bc6ad5fc5e1",
    "4fc26a7d2d28",
    "4ac252ba38ed",
]
"""
List of revisions that have been deleted

If a user's database contains these revisions then they need to delete their database and start again.
"""


def _get_database_revision(connection: sqlalchemy.engine.Connection) -> str | None:
    context = MigrationContext.configure(connection)
    current_rev = context.get_current_revision()
    return current_rev


def _create_backup(db_path: Path, max_backups: int) -> Path:
    """
    Create a backup of the database file

    Parameters
    ----------
    db_path
        Path to the database file
    max_backups
        Maximum number of backups to keep

    Returns
    -------
    :
        Path to the backup file
    """
    if not db_path.exists():
        logger.warning(f"Database file {db_path} does not exist, skipping backup")
        return db_path

    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}_{timestamp}{db_path.suffix}"

    logger.info(f"Creating backup of database at {backup_path}")
    shutil.copy2(db_path, backup_path)

    # Clean up old backups
    backups = sorted(backup_dir.glob(f"{db_path.stem}_*{db_path.suffix}"), reverse=True)
    for old_backup in backups[max_backups:]:
        logger.info(f"Removing old backup {old_backup}")
        old_backup.unlink()

    return backup_path


def validate_database_url(database_url: str) -> str:
    """
    Validate a database URL

    We support sqlite databases, and we create the directory if it doesn't exist.
    We may aim to support PostgreSQL databases, but this is currently experimental and untested.

    Parameters
    ----------
    database_url
        The database URL to validate

        See [climate_ref.config.DbConfig.database_url][climate_ref.config.DbConfig.database_url]
        for more information on the format of the URL.

    Raises
    ------
    ValueError
        If the database scheme is not supported

    Returns
    -------
    :
        The validated database URL
    """
    split_url = urlparse.urlsplit(database_url)
    path = split_url.path[1:]

    if split_url.scheme == "sqlite":
        if path == ":memory:":
            logger.warning("Using an in-memory database")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    elif split_url.scheme == "postgresql":
        # We don't need to do anything special for PostgreSQL
        logger.warning("PostgreSQL support is currently experimental and untested")
    else:
        raise ValueError(f"Unsupported database scheme: {split_url.scheme}")

    return database_url


class ModelState(enum.Enum):
    """
    State of a model instance
    """

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class Database:
    """
    Manage the database connection and migrations

    The database migrations are optionally run after the connection to the database is established.
    """

    def __init__(self, url: str) -> None:
        logger.info(f"Connecting to database at {url}")
        self.url = url
        self._engine = sqlalchemy.create_engine(self.url)
        # TODO: Set autobegin=False
        self.session = Session(self._engine)

    def __enter__(self) -> "Database":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """
        Close the database connection

        This closes the session and disposes of the engine, releasing all connections.
        """
        try:
            self.session.close()
        finally:
            self._engine.dispose()

    def alembic_config(self, config: "Config") -> AlembicConfig:
        """
        Get the Alembic configuration object for the database

        This includes an open connection with the database engine and the REF configuration.

        Returns
        -------
        :
            The Alembic configuration object that can be used with alembic commands
        """
        alembic_config_filename = importlib.resources.files("climate_ref") / "alembic.ini"
        if not alembic_config_filename.is_file():  # pragma: no cover
            raise FileNotFoundError(f"{alembic_config_filename} not found")

        alembic_config = AlembicConfig(str(alembic_config_filename))
        alembic_config.attributes["connection"] = self._engine
        alembic_config.attributes["ref_config"] = config

        return alembic_config

    def migrate(self, config: "Config") -> None:
        """
        Migrate the database to the latest revision

        Parameters
        ----------
        config
            REF Configuration

            This is passed to alembic
        """
        # Check if the database revision is one of the removed revisions
        # If it is, then we need to delete the database and start again
        with self._engine.connect() as connection:
            current_rev = _get_database_revision(connection)
            logger.debug(f"Current database revision: {current_rev}")
            if current_rev in _REMOVED_REVISIONS:
                raise ValueError(
                    f"Database revision {current_rev!r} has been removed in "
                    f"https://github.com/Climate-REF/climate-ref/pull/271. "
                    "Please delete your database and start again."
                )

        # Create backup before running migrations
        split_url = urlparse.urlsplit(self.url)
        if split_url.scheme == "sqlite" and split_url.path != ":memory:":
            db_path = Path(split_url.path[1:])
            _create_backup(db_path, config.db.max_backups)

        alembic.command.upgrade(self.alembic_config(config), "heads")

    @staticmethod
    def from_config(config: "Config", run_migrations: bool = True) -> "Database":
        """
        Create a Database instance from a Config instance

        The `REF_DATABASE_URL` environment variable will take preference,
         and override the database URL specified in the config.

        Parameters
        ----------
        config
            The Config instance that includes information about where the database is located

        Returns
        -------
        :
            A new Database instance
        """
        database_url: str = config.db.database_url

        database_url = validate_database_url(database_url)

        cv = CV.load_from_file(config.paths.dimensions_cv)
        db = Database(database_url)

        if run_migrations:
            # Run any outstanding migrations
            # This also adds any diagnostic value columns to the DB if they don't exist
            db.migrate(config)
        # Register the CV dimensions with the MetricValue model
        # This will add new columns to the db if the CVs have changed
        MetricValue.register_cv_dimensions(cv)

        # Register the CV dimensions with the ExecutionOutput model
        # This enables dimension-based filtering of outputs
        ExecutionOutput.register_cv_dimensions(cv)

        return db

    def update_or_create(
        self, model: type[Table], defaults: dict[str, Any] | None = None, **kwargs: Any
    ) -> tuple[Table, ModelState | None]:
        """
        Update an existing instance or create a new one

        This doesn't commit the transaction,
        so you will need to call `session.commit()` after this method
        or use a transaction context manager.

        Parameters
        ----------
        model
            The model to update or create
        defaults
            Default values to use when creating a new instance, or values to update on existing instance
        kwargs
            The filter parameters to use when querying for an instance

        Returns
        -------
        :
            A tuple containing the instance and a state enum indicating if the instance was created or updated
        """
        instance = self.session.query(model).filter_by(**kwargs).first()
        state: ModelState | None = None
        if instance:
            # Update existing instance with defaults
            if defaults:
                for key, value in defaults.items():
                    if getattr(instance, key) != value:
                        logger.debug(f"Updating {model.__name__} {key} to {value}")
                        setattr(instance, key, value)
                        state = ModelState.UPDATED
            return instance, state
        else:
            # Create new instance
            params = {**kwargs, **(defaults or {})}
            instance = model(**params)
            self.session.add(instance)
            return instance, ModelState.CREATED

    def get_or_create(
        self, model: type[Table], defaults: dict[str, Any] | None = None, **kwargs: Any
    ) -> tuple[Table, ModelState | None]:
        """
        Get or create an instance of a model

        This doesn't commit the transaction,
        so you will need to call `session.commit()` after this method
        or use a transaction context manager.

        Parameters
        ----------
        model
            The model to get or create
        defaults
            Default values to use when creating a new instance
        kwargs
            The filter parameters to use when querying for an instance

        Returns
        -------
        :
            A tuple containing the instance and enum indicating if the instance was created
        """
        instance = self.session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance, None
        else:
            params = {**kwargs, **(defaults or {})}
            instance = model(**params)
            self.session.add(instance)
            return instance, ModelState.CREATED
