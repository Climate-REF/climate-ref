"""
Data access layer for REF results.

This package is the single source of truth for how result data is filtered and queried. It serves
notebooks / local users (via pandas-friendly, detached value objects) and, in future, the ref-app
API (which can reuse the shared ``Select`` builders directly).

Layers:

* [_query][climate_ref.results._query] -- ``MetricValueFilter`` + ``select_*`` builders
  (return a ``Select``; the source of truth for filter/join logic) + ``count_values`` +
  ``latest_execution_for_group``.
* [frames][climate_ref.results.frames] -- ``*_to_frame`` + ``collect_facets``.
* [outliers][climate_ref.results.outliers] -- source-id-aware IQR outlier detection.
* [values][climate_ref.results.values] -- typed DTOs + collections + the ``Reader`` facade.

The typical notebook entry point::

    from climate_ref.config import Config
    from climate_ref.database import Database
    from climate_ref.results import Reader

    with Database.from_config(Config.default(), read_only=True) as db:
        df = Reader(db).scalar_values().to_pandas()
"""

from climate_ref.results._query import (
    MetricValueFilter,
    count_values,
    latest_execution_for_group,
    select_scalar_values,
    select_series_values,
)
from climate_ref.results.frames import (
    collect_facets,
    scalar_values_to_frame,
    series_values_to_frame,
)
from climate_ref.results.outliers import OutlierPolicy, detect_scalar_outliers
from climate_ref.results.values import (
    Facet,
    Reader,
    ScalarValue,
    ScalarValueCollection,
    SeriesValue,
    SeriesValueCollection,
)

__all__ = [
    "Facet",
    "MetricValueFilter",
    "OutlierPolicy",
    "Reader",
    "ScalarValue",
    "ScalarValueCollection",
    "SeriesValue",
    "SeriesValueCollection",
    "collect_facets",
    "count_values",
    "detect_scalar_outliers",
    "latest_execution_for_group",
    "scalar_values_to_frame",
    "select_scalar_values",
    "select_series_values",
    "series_values_to_frame",
]
