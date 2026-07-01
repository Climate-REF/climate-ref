"""
DataFrame conversion and facet collection for metric values.

These pure helpers replace logic duplicated across the CLI (hand-rolled ``pd.DataFrame([...])``
comprehensions) and the ref-app API (CSV flattening + ``collect_facets_from_query``). They take
rows / a ``Select`` and return plain data, so they can be reused by any consumer.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Select, distinct
from sqlalchemy.orm import Session

from climate_ref.models.metric_value import (
    MetricValue,
    ScalarMetricValue,
    SeriesMetricValue,
)


def scalar_values_to_frame(rows: Sequence[ScalarMetricValue]) -> pd.DataFrame:
    """
    Flatten scalar rows to a tidy DataFrame.

    One row per value; one column per non-null CV dimension, plus ``value``, ``id``,
    ``execution_id`` and ``type``. ``value`` is left raw (NaN/inf preserved); callers that
    serialise to JSON/CSV are responsible for any sanitisation.
    """
    records = []
    for r in rows:
        rec: dict[str, Any] = dict(r.dimensions)
        rec.update(id=r.id, execution_id=r.execution_id, value=r.value, type="scalar")
        records.append(rec)
    return pd.DataFrame.from_records(records)


def series_values_to_frame(rows: Sequence[SeriesMetricValue]) -> pd.DataFrame:
    """
    Flatten series rows to a long-form (tidy) DataFrame.

    One row per (series, index point). Columns: the non-null CV dimensions, plus ``value``,
    ``index``, ``index_name``, ``reference_id``, ``id``, ``execution_id`` and ``type``. The index
    is resolved from the shared axis; when a series has no index the positional integer is used.
    """
    records = []
    for r in rows:
        dims = r.dimensions
        idx = r.index
        name = r.index_name or "index"
        for i, v in enumerate(r.values or []):
            rec: dict[str, Any] = dict(dims)
            rec.update(
                id=r.id,
                execution_id=r.execution_id,
                value=v,
                index=idx[i] if idx is not None and i < len(idx) else i,
                index_name=name,
                reference_id=r.reference_id,
                type="series",
            )
            records.append(rec)
    return pd.DataFrame.from_records(records)


def collect_facets(
    session: Session,
    stmt: Select[Any],
    entity: type[MetricValue],
) -> dict[str, list[str]]:
    """
    Distinct non-null values for each registered CV dimension of a filtered query.

    Runs one ``DISTINCT`` per dimension over the (pre-pagination) ``stmt`` so cost scales with
    cardinality rather than row count. Returns only dimensions that have at least one value.
    """
    facets: dict[str, list[str]] = {}
    for key in entity._cv_dimensions:
        col = getattr(entity, key)
        sub = stmt.with_only_columns(distinct(col)).order_by(None)
        values = [v for (v,) in session.execute(sub) if v is not None]
        if values:
            facets[key] = sorted(values)
    return facets
