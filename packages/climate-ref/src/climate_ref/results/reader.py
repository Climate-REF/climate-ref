"""The entry point for reading REF results."""

import functools
from pathlib import Path

from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.results.artifacts import ArtifactsReader
from climate_ref.results.datasets import DatasetsReader
from climate_ref.results.diagnostics import DiagnosticsReader
from climate_ref.results.executions import ExecutionsReader
from climate_ref.results.resources import ResourcesReader
from climate_ref.results.values import ValuesReader


class Reader:
    """
    Entry point for reading REF results into pandas.

    Build it from an open [Database][climate_ref.database.Database].
    Open the database with ``read_only=True`` so that reading never changes it.
    Each kind of result has its own reader, available as a property:

    * [values][climate_ref.results.reader.Reader.values]: metric values, as single numbers or series.
    * [executions][climate_ref.results.reader.Reader.executions]: groups, executions and output files.
    * [datasets][climate_ref.results.reader.Reader.datasets]: the datasets that have been ingested.
    * [diagnostics][climate_ref.results.reader.Reader.diagnostics]: diagnostics and how many groups succeeded.
    * [resources][climate_ref.results.reader.Reader.resources]: CPU and memory use of executions.
    * [artifacts][climate_ref.results.reader.Reader.artifacts]: paths of output files (needs ``results``).

    Parameters
    ----------
    database
        The open REF database.
    results
        The folder that holds the results, e.g. ``config.paths.results``. Only ``reader.artifacts`` needs it.
    session
        A database session that you manage yourself, such as one from
        [Database.session_scope][climate_ref.database.Database.session_scope].
        Most users leave it out.
        If you set it, do not use the reader after that session ends.

    Examples
    --------
    See which models have results for a diagnostic, then compare them:

    ```python
    from climate_ref.config import Config
    from climate_ref.database import Database
    from climate_ref.results import MetricValueFilter, Reader

    config = Config.default()

    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db, results=config.paths.results)
        collection = reader.values.scalar_values(MetricValueFilter(diagnostic_slug="enso_tel"))
        print(collection.facets_dict()["source_id"])
        df = collection.to_pandas()
        groups = reader.executions.groups().to_pandas()
    ```
    """

    def __init__(
        self, database: Database, results: Path | None = None, *, session: Session | None = None
    ) -> None:
        self._db = database
        self._results = results
        self._session = session

    @property
    def session(self) -> Session:
        """The database session the reads use."""
        return self._session if self._session is not None else self._db.session

    @functools.cached_property
    def values(self) -> ValuesReader:
        """Reader for scalar and series metric values."""
        return ValuesReader(self._db, session=self._session)

    @functools.cached_property
    def executions(self) -> ExecutionsReader:
        """Reader for execution groups, executions and their outputs."""
        return ExecutionsReader(self._db, session=self._session)

    @functools.cached_property
    def datasets(self) -> DatasetsReader:
        """Reader for ingested datasets."""
        return DatasetsReader(self._db, session=self._session)

    @functools.cached_property
    def diagnostics(self) -> DiagnosticsReader:
        """Reader for diagnostics and their execution counts."""
        return DiagnosticsReader(self._db, session=self._session)

    @functools.cached_property
    def resources(self) -> ResourcesReader:
        """Reader for the CPU and memory use of executions."""
        return ResourcesReader(self._db, session=self._session)

    @functools.cached_property
    def artifacts(self) -> ArtifactsReader:
        """
        Reader for the paths of execution outputs.

        Raises
        ------
        ValueError
            If no ``results`` root was passed to ``Reader``.
        """
        if self._results is None:
            raise ValueError(
                "reader.artifacts requires a results root; construct "
                "Reader(database, results=config.paths.results)."
            )
        return ArtifactsReader(self._results)
