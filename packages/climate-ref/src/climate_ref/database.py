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
import pandas as pd
import sqlalchemy
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
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


def _get_sqlite_path(database_url: str) -> Path | None:
    """
    Extract the file path from a SQLite database URL.

    Returns ``None`` for in-memory databases, URI-form URLs (``sqlite:///file:...``),
    or non-SQLite URLs.
    """
    split_url = urlparse.urlsplit(database_url)
    if split_url.scheme != "sqlite":
        return None
    path = urlparse.unquote(split_url.path[1:])
    if not path or path == ":memory:":
        return None
    if path.startswith("file:"):
        # URI-form SQLite URL — the path is opaque (may contain query params
        # of its own for mode=ro etc.) and we shouldn't try to interpret it.
        return None
    return Path(path)


def _make_readonly_sqlite_url(database_url: str) -> tuple[str, dict[str, Any]]:
    """
    Rewrite a file-based SQLite URL to read-only URI form.

    Returns the rewritten URL and connect_args to pass to SQLAlchemy.
    For non-SQLite URLs, or URLs that can't be rewritten, the URL is returned unchanged
    with empty connect_args.
    """
    split_url = urlparse.urlsplit(database_url)
    if split_url.scheme != "sqlite":
        logger.warning("Read-only mode is only supported for SQLite databases; ignoring read-only flag")
        return database_url, {}

    # Preserve the original URL encoding — the rewritten URL also needs to be
    # parseable as a URI, so percent-encoded characters (e.g. spaces as ``%20``)
    # must not be decoded back into raw characters here.
    encoded_path = split_url.path[1:]
    if not encoded_path or encoded_path == ":memory:":
        return database_url, {}

    if encoded_path.startswith("file:"):
        # Already URI form — caller is responsible for any ro/immutable flags.
        return database_url, {"uri": True}

    return f"sqlite:///file:{encoded_path}?mode=ro&immutable=1&uri=true", {"uri": True}


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

    if split_url.scheme == "sqlite":
        # URI-form SQLite URLs (``sqlite:///file:...``) are passed through
        # verbatim — the caller has supplied an explicit URI, possibly for a
        # read-only on-disk file, and we should neither treat it as in-memory
        # nor try to mkdir its (opaque) parent directory.
        if split_url.path[1:].startswith("file:"):
            logger.debug("Using URI-form SQLite URL; skipping parent directory creation")
        else:
            sqlite_path = _get_sqlite_path(database_url)
            if sqlite_path is None:
                logger.warning("Using an in-memory database")
            else:
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    elif split_url.scheme == "postgresql":
        # We don't need to do anything special for PostgreSQL
        logger.warning("PostgreSQL support is currently experimental and untested")
    else:
        raise ValueError(f"Unsupported database scheme: {split_url.scheme}")

    return database_url


def _values_differ(current: Any, new: Any) -> bool:
    """
    Safely compare two values for inequality, handling ``pd.NA`` and ``np.nan``.

    Direct ``!=`` comparison with ``pd.NA`` raises ``TypeError`` because
    ``bool(pd.NA)`` is ambiguous.  This helper avoids that by checking
    for NA on both sides first.
    """
    try:
        current_is_na = pd.isna(current)
        new_is_na = pd.isna(new)
    except (TypeError, ValueError):
        current_is_na = False
        new_is_na = False

    if current_is_na and new_is_na:
        return False
    if current_is_na or new_is_na:
        return True

    try:
        return bool(current != new)
    except (TypeError, ValueError):
        # Fallback for types whose __ne__ returns an ambiguous result
        return True


class ModelState(enum.Enum):
    """
    State of a model instance
    """

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class MigrationState(enum.Enum):
    """
    State of the database schema relative to the expected Alembic head.
    """

    UP_TO_DATE = "up_to_date"
    BEHIND = "behind"
    UNMANAGED = "unmanaged"
    REMOVED = "removed"


class Database:
    """
    Manage the database connection and migrations

    The database migrations are optionally run after the connection to the database is established.
    """

    def __init__(self, url: str, *, connect_args: dict[str, Any] | None = None) -> None:
        logger.info(f"Connecting to database at {url}")
        self.url = url
        engine_kwargs: dict[str, Any] = {}
        if connect_args:
            engine_kwargs["connect_args"] = connect_args
        self._engine = sqlalchemy.create_engine(self.url, **engine_kwargs)
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

    def migration_status(self, config: "Config") -> dict[str, Any]:
        """
        Report the current migration state of the database.

        Returns a dict with ``current`` (the current revision or ``None``),
        ``head`` (the latest available revision), and ``state`` (a
        :class:`MigrationState`).

        This is the canonical way for consumers of the library to check whether
        the database schema matches what the installed ``climate_ref`` expects.
        Prefer it over re-deriving Alembic plumbing in downstream code.

        Parameters
        ----------
        config
            REF Configuration, used to build the Alembic config.

        Returns
        -------
        :
            A dict with keys ``current``, ``head``, and ``state``.
        """
        alembic_cfg = self.alembic_config(config)
        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()

        with self._engine.connect() as connection:
            current_rev = _get_database_revision(connection)

        if current_rev in _REMOVED_REVISIONS:
            state = MigrationState.REMOVED
        elif current_rev is None:
            state = MigrationState.UNMANAGED
        elif current_rev == head_rev:
            state = MigrationState.UP_TO_DATE
        else:
            state = MigrationState.BEHIND

        return {"current": current_rev, "head": head_rev, "state": state}

    def migrate(self, config: "Config", skip_backup: bool = False) -> None:
        """
        Migrate the database to the latest revision

        Parameters
        ----------
        config
            REF Configuration

            This is passed to alembic
        skip_backup
            If True, skip creating a backup before running migrations.
            Useful for read-only commands that don't modify the database.
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

        # Create backup before running migrations (unless skipped)
        db_path = _get_sqlite_path(self.url)
        if not skip_backup and db_path is not None:
            _create_backup(db_path, config.db.max_backups)

        alembic.command.upgrade(self.alembic_config(config), "heads")

    @staticmethod
    def from_config(
        config: "Config",
        run_migrations: bool = True,
        skip_backup: bool = False,
        *,
        read_only: bool = False,
    ) -> "Database":
        """
        Create a Database instance from a Config instance

        The `REF_DATABASE_URL` environment variable will take preference,
         and override the database URL specified in the config.

        Parameters
        ----------
        config
            The Config instance that includes information about where the database is located
        run_migrations
            If True, run any outstanding database migrations.
            Forced to False when ``read_only=True``.
        skip_backup
            If True, skip creating a backup before running migrations.
            Useful for read-only commands that don't modify the database.
        read_only
            If True, open the database in read-only mode and skip migrations.

            SQLite URLs are rewritten to URI form with ``mode=ro&immutable=1``.
            For other backends, callers must configure the connecting role as
            read-only themselves.

        Returns
        -------
        :
            A new Database instance
        """
        database_url: str = config.db.database_url
        connect_args: dict[str, Any] = {}

        if read_only:
            database_url, connect_args = _make_readonly_sqlite_url(database_url)
            run_migrations = False

        database_url = validate_database_url(database_url)

        cv = CV.load_from_file(config.paths.dimensions_cv)
        db = Database(database_url, connect_args=connect_args or None)

        if run_migrations:
            # Run any outstanding migrations
            # This also adds any diagnostic value columns to the DB if they don't exist
            db.migrate(config, skip_backup=skip_backup)
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
                    if _values_differ(getattr(instance, key), value):
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
