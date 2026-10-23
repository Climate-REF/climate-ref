"""
DataFrame conversion and facet collection for metric values.

These functions decide the columns of the DataFrames that ``to_pandas()`` returns for metric values.
Most users call ``to_pandas()`` on the result of a read and never need them.
They are useful to turn values you already hold, such as a filtered list of them, into the same layout.
They work on the value objects from [climate_ref.results.values][] and need no database.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy import JSON, Select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.functions import FunctionElement

from climate_ref.models.metric_value import MetricValue

if TYPE_CHECKING:
    from climate_ref.results.values import ScalarValue, SeriesValue


def scalar_values_to_frame(values: "Sequence[ScalarValue]", *, detection_ran: bool = False) -> pd.DataFrame:
    """
    Turn scalar values into a table with one row per value.

    The columns are one per label on the values (such as ``source_id`` or ``metric``),
    then ``id``, ``execution_id``, ``execution_group_id``, ``kind`` and ``value``.
    ``kind`` says whether the value is for a model or a reference.

    The columns ``is_outlier`` and ``verification_status`` are added only if ``detection_ran`` is true.
    The columns ``diagnostic_slug`` and ``provider_slug`` are added only if the values carry them,
    which needs ``include_context=True`` on the read.

    ``value`` is not cleaned, so NaN and inf stay as they are.

    Examples
    --------
    >>> from climate_ref.results.frames import scalar_values_to_frame
    >>> from climate_ref.results.values import ScalarValue
    >>> value = ScalarValue(
    ...     id=1,
    ...     execution_id=2,
    ...     execution_group_id=3,
    ...     value=0.5,
    ...     kind="model",
    ...     dimensions={"source_id": "CanESM5", "statistic": "mean"},
    ...     attributes={},
    ... )
    >>> list(scalar_values_to_frame([value]).columns)
    ['source_id', 'statistic', 'id', 'execution_id', 'execution_group_id', 'kind', 'value']
    """
    records = []
    for v in values:
        rec: dict[str, Any] = dict(v.dimensions)
        rec.update(
            id=v.id,
            execution_id=v.execution_id,
            execution_group_id=v.execution_group_id,
            kind=v.kind,
            value=v.value,
        )
        if detection_ran:
            rec.update(is_outlier=v.is_outlier, verification_status=v.verification_status)
        if v.diagnostic_slug is not None:
            rec.update(diagnostic_slug=v.diagnostic_slug, provider_slug=v.provider_slug)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def series_values_to_frame(values: "Sequence[SeriesValue]", *, explode: bool = True) -> pd.DataFrame:
    """
    Turn series values, such as time series, into a table.

    With ``explode=True`` (the default) there is one row per point, with columns ``index`` and ``value``.
    This is the form to plot or group with pandas.
    With ``explode=False`` there is one row per series, and ``values`` and ``index`` hold lists.

    The other columns are one per label on the values (such as ``source_id``),
    then ``id``, ``execution_id``, ``execution_group_id``, ``kind``, ``index_name`` and ``reference_id``.
    The columns ``diagnostic_slug`` and ``provider_slug`` are added only if the values carry them,
    which needs ``include_context=True`` on the read.

    Examples
    --------
    >>> from climate_ref.results.frames import series_values_to_frame
    >>> from climate_ref.results.values import SeriesValue
    >>> series = SeriesValue(
    ...     id=1,
    ...     execution_id=2,
    ...     execution_group_id=3,
    ...     values=[280.1, 280.4],
    ...     index=[2000, 2001],
    ...     index_name="time",
    ...     reference_id=None,
    ...     kind="model",
    ...     dimensions={"source_id": "CanESM5"},
    ...     attributes={},
    ... )
    >>> series_values_to_frame([series])[["index", "value"]]
       index  value
    0   2000  280.1
    1   2001  280.4
    >>> len(series_values_to_frame([series], explode=False))
    1
    """
    records = []
    for v in values:
        base: dict[str, Any] = dict(v.dimensions)
        base.update(
            id=v.id,
            execution_id=v.execution_id,
            execution_group_id=v.execution_group_id,
            kind=v.kind,
            index_name=v.index_name or "index",
            reference_id=v.reference_id,
        )
        if v.diagnostic_slug is not None:
            base.update(diagnostic_slug=v.diagnostic_slug, provider_slug=v.provider_slug)
        if explode:
            idx = v.index
            for i, value in enumerate(v.values):
                rec = dict(base)
                rec.update(value=value, index=idx[i] if idx is not None and i < len(idx) else i)
                records.append(rec)
        else:
            rec = dict(base)
            rec.update(values=list(v.values), index=list(v.index) if v.index is not None else None)
            records.append(rec)
    return pd.DataFrame.from_records(records)


class _DistinctValues(FunctionElement[Any]):
    """JSON array of the distinct values of a column, including any null."""

    inherit_cache = True
    type = JSON()


@compiles(_DistinctValues)
def _compile_distinct_values(element: _DistinctValues, compiler: SQLCompiler, **kw: Any) -> str:
    return f"json_agg(DISTINCT {compiler.process(element.clauses, **kw)})"


@compiles(_DistinctValues, "sqlite")
def _compile_distinct_values_sqlite(element: _DistinctValues, compiler: SQLCompiler, **kw: Any) -> str:
    return f"json_group_array(DISTINCT {compiler.process(element.clauses, **kw)})"


def collect_facets(
    session: Session,
    stmt: Select[Any],
    entity: type[MetricValue],
) -> dict[str, list[str]]:
    """
    Distinct non-null values for each registered CV dimension of a filtered query.

    ``stmt`` selects the matching value rows, before any paging.
    Every dimension is aggregated in one pass, so each row is read once.
    The result maps a dimension to its sorted values, and leaves out dimensions with no value.
    """
    keys = list(entity._cv_dimensions)
    if not keys:
        return {}
    sub = stmt.with_only_columns(*(_DistinctValues(getattr(entity, key)) for key in keys)).order_by(None)
    return facets_from_columns(keys, session.execute(sub).one())


def facets_from_columns(keys: Sequence[str], columns: Sequence[Sequence[Any] | None]) -> dict[str, list[Any]]:
    """
    Sorted distinct non-null values per key, from one sequence of values per key.

    Keys with no value are dropped.

    Examples
    --------
    >>> from climate_ref.results.frames import facets_from_columns
    >>> facets_from_columns(["source_id", "region"], [["b", "a", "b", None], [None]])
    {'source_id': ['a', 'b']}
    """
    facets: dict[str, list[Any]] = {}
    for key, values in zip(keys, columns, strict=True):
        present = sorted({v for v in values or () if v is not None})
        if present:
            facets[key] = present
    return facets
