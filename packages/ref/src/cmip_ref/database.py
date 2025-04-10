"""Database adapter layer

This module provides a database adapter layer that abstracts the database connection and migrations.
This allows us to easily switch between different database backends,
and to run migrations when the database is loaded.

The `Database` class is the main entry point for interacting with the database.
It provides a session object that can be used to interact with the database and run queries.
"""

import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib import parse as urlparse

import alembic.command
import sqlalchemy
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from loguru import logger
from sqlalchemy.orm import Session

from cmip_ref.models import MetricValue, Table
from cmip_ref_core.pycmec.controlled_vocabulary import CV

if TYPE_CHECKING:
    from cmip_ref.config import Config


def validate_database_url(database_url: str) -> str:
    """
    Validate a database URL

    We support sqlite databases, and we create the directory if it doesn't exist.
    We may aim to support PostgreSQL databases, but this is currently experimental and untested.

    Parameters
    ----------
    database_url
        The database URL to validate

        See [cmip_ref.config.DbConfig.database_url][cmip_ref.config.DbConfig.database_url]
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


class Database:
    """
    Manage the database connection and migrations

    The database migrations are optionally run after the connection to the database is established.
    """

    def __init__(self, url: str) -> None:
        logger.info(f"Connecting to database at {url}")
        self.url = url
        self._engine = sqlalchemy.create_engine(self.url)
        self.session = Session(self._engine)

    def migrate(self, config: "Config") -> None:
        """
        Migrate the database to the latest revision

        Parameters
        ----------
        config
            REF Configuration

            This is passed to alembic
        """
        alembic_config_filename = importlib.resources.files("cmip_ref") / "alembic.ini"
        if not alembic_config_filename.is_file():  # pragma: no cover
            raise FileNotFoundError(f"{alembic_config_filename} not found")
        alembic_config = AlembicConfig(str(alembic_config_filename))
        alembic_config.attributes["connection"] = self._engine
        alembic_config.attributes["ref_config"] = config

        script = ScriptDirectory.from_config(alembic_config)
        head = script.get_current_head()

        # Run migrations
        alembic.command.upgrade(alembic_config, head or "heads")

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
        run_migrations
            If true, run the migrations when the database is loaded

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
            # This also adds any metric value columns to the DB if they don't exist
            db.migrate(config)
        # Register the CV dimensions with the MetricValue model
        # This will add new columns to the db if the CVs have changed
        MetricValue.register_cv_dimensions(cv)

        return db

    def get_or_create(
        self, model: type[Table], defaults: dict[str, Any] | None = None, **kwargs: Any
    ) -> tuple[Table, bool]:
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
            A tuple containing the instance and a boolean indicating if the instance was created
        """
        instance = self.session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance, False
        else:
            params = {**kwargs, **(defaults or {})}
            instance = model(**params)
            self.session.add(instance)
            return instance, True
