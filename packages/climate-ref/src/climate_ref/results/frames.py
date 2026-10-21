"""
DataFrame conversion and facet collection for metric values.

These pure helpers are the single source of truth for the column layout of a metric-value frame.
Both the collections' ``to_pandas()`` and any other consumer build their frames here,
so a scalar (or series) frame has identical columns.
The builders take detached DTOs (from [climate_ref.results.values][]) rather than ORM rows,
so they never touch a session and are portable.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy import Select, String, literal, select, union_all
from sqlalchemy.orm import Session

from climate_ref.models.metric_value import MetricValue

if TYPE_CHECKING:
    from climate_ref.results.values import ScalarValue, SeriesValue


def scalar_values_to_frame(values: "Sequence[ScalarValue]", *, detection_ran: bool = False) -> pd.DataFrame:
    """
    Flatten scalar value DTOs to a tidy DataFrame.

    One row per value; one column per CV dimension present, plus ``id``, ``execution_id``,
    ``execution_group_id``, ``kind`` and ``value``.

    ``kind`` is promoted out of the dimension columns.

    Outlier columns (``is_outlier``/``verification_status``) are added only when ``detection_ran`` is set,
    and context columns (``diagnostic_slug``/``provider_slug``) only when populated on the DTOs.

    ``value`` is left raw (NaN/inf preserved).
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
    Flatten series value DTOs to a DataFrame.

    With ``explode=True`` (default) the result is long-form: one row per (series, index point),
    with columns ``value`` and ``index`` in addition to the shared metadata.
    With ``explode=False`` each series is one row with list-valued ``values``/``index`` cells.
    Shared columns are the CV dimensions present plus ``id``, ``execution_id``, ``execution_group_id``,
    ``kind``, ``index_name`` and ``reference_id``.

    Context columns (``diagnostic_slug``/``provider_slug``) are added only when populated on the DTOs.
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


def collect_facets(
    session: Session,
    stmt: Select[Any],
    entity: type[MetricValue],
) -> dict[str, list[str]]:
    """
    Distinct non-null values for each registered CV dimension of a filtered query.

    Runs one statement: a CTE over the (pre-pagination) ``stmt`` that selects only the dimension columns,
    then a ``UNION ALL`` of one ``GROUP BY`` per dimension.
    The wide ``SERIES`` payload is never read, and the filtered rows are scanned once.
    Returns only dimensions that have at least one value.
    """
    keys = list(entity._cv_dimensions)
    if not keys:
        return {}
    rows_cte = stmt.with_only_columns(*(getattr(entity, k) for k in keys)).order_by(None).cte("facet_rows")
    union = union_all(
        *(
            select(literal(k, String).label("key"), rows_cte.c[k].label("value"))
            .where(rows_cte.c[k].is_not(None))
            .group_by(rows_cte.c[k])
            for k in keys
        )
    )
    facets: dict[str, list[str]] = {}
    for key, value in session.execute(union):
        facets.setdefault(key, []).append(value)
    return {k: sorted(facets[k]) for k in keys if k in facets}
