"""
Data access layer for REF results.

This package is the single source of truth for how result data is filtered and queried.
It serves notebooks / local users (via pandas-friendly, detached value objects) and,
in future, the ref-app API (which can reuse the shared ``Select`` builders directly).

Layers:

* [_query][climate_ref.results._query] -- ``MetricValueFilter`` + ``select_*`` builders
  (return a ``Select``; the source of truth for filter/join logic) + ``count_values`` +
  ``latest_execution_for_group``.
* [frames][climate_ref.results.frames] -- ``*_to_frame`` + ``collect_facets``.
* [outliers][climate_ref.results.outliers] -- source-id-aware IQR outlier detection.
* [values][climate_ref.results.values] -- typed DTOs + collections + the ``Reader`` facade.
* [executions][climate_ref.results.executions] -- execution-group / execution DTOs + collection +
  the ``ExecutionsReader`` facade.
* [artifacts][climate_ref.results.artifacts] -- the ``ArtifactsReader`` facade,
  resolving execution output fragments into filesystem paths under a results root.

The typical notebook entry point::

    from climate_ref.config import Config
    from climate_ref.database import Database
    from climate_ref.results import Reader

    with Database.from_config(Config.default(), read_only=True) as db:
        df = Reader(db).values.scalar_values().to_pandas()

Public surface convention
-------------------------

The top level exports only what a caller *names to make a call*,
so the namespace stays small as domains are added:

* the ``Reader`` entry point,
* filter objects you construct and pass in (``MetricValueFilter``, ``ExecutionGroupFilter``,
  and the future ``DatasetFilter``),
* value objects you pass in (``OutlierPolicy``).

Everything the package *returns* -- DTOs, collections, views -- and the sub-reader classes reached
via ``reader.values`` etc. live in their domain submodule (``climate_ref.results.values``, ...).
Import them from there on the rare occasion you need to name one, e.g.
``from climate_ref.results.values import ScalarValue``.
The ``Select`` builders and other plumbing stay in ``climate_ref.results._query``
and are not part of the public surface.
"""

from climate_ref.results._query import MetricValueFilter
from climate_ref.results.executions import ExecutionGroupFilter
from climate_ref.results.outliers import OutlierPolicy
from climate_ref.results.values import Reader

__all__ = [
    "ExecutionGroupFilter",
    "MetricValueFilter",
    "OutlierPolicy",
    "Reader",
]
