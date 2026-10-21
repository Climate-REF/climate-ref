# /// script
# requires-python = ">=3.11"
# dependencies = ["climate-ref", "typer"]
#
# [tool.uv.sources]
# climate-ref = { path = "../packages/climate-ref" }
# ///
"""
Time the old per-dimension DISTINCT facets against the single-statement version.

The database is opened read-only (SQLite ``mode=ro``), so nothing is written.
"""

import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from sqlalchemy import Select, distinct

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models.metric_value import MetricValue, ScalarMetricValue, SeriesMetricValue
from climate_ref.results._query import MetricValueFilter, select_scalar_values, select_series_values
from climate_ref.results.frames import collect_facets
from climate_ref_core.pycmec.controlled_vocabulary import CV

RUNS = 5


def per_dimension_facets(session: Any, stmt: Select[Any], entity: type[MetricValue]) -> dict[str, list[str]]:
    """Collect facets the previous way: one ``DISTINCT`` statement per dimension."""
    facets: dict[str, list[str]] = {}
    for key in entity._cv_dimensions:
        sub = stmt.with_only_columns(distinct(getattr(entity, key))).order_by(None)
        values = [v for (v,) in session.execute(sub) if v is not None]
        if values:
            facets[key] = sorted(values)
    return facets


def median_seconds(fn: Callable[[], object]) -> float:
    """Median wall time of ``fn`` over a few runs."""
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main(
    db: Annotated[Path, typer.Option("--db", help="Path to the SQLite database file (opened read-only).")],
    diagnostic: Annotated[
        str | None, typer.Option("--diagnostic", help="Diagnostic slug to filter on.")
    ] = None,
) -> None:
    """Compare old and new facet collection on a real database."""
    uri = f"sqlite:///file:{db.resolve()}?mode=ro&uri=true"
    database = Database(uri, connect_args={"uri": True})
    cv = CV.load(Config.default().paths.dimensions_cv_resource)
    MetricValue.register_cv_dimensions(cv)

    cases: list[tuple[str, Select[Any], type[MetricValue]]] = [
        ("unfiltered scalar", select_scalar_values(MetricValueFilter()), ScalarMetricValue),
        ("unfiltered series", select_series_values(MetricValueFilter()), SeriesMetricValue),
    ]
    if diagnostic:
        flt = MetricValueFilter(diagnostic_slug=diagnostic)
        cases.append((f"{diagnostic} scalar", select_scalar_values(flt), ScalarMetricValue))
        cases.append((f"{diagnostic} series", select_series_values(flt), SeriesMetricValue))

    session = database.session
    for name, stmt, entity in cases:
        old = per_dimension_facets(session, stmt, entity)
        new = collect_facets(session, stmt, entity)
        t_old = median_seconds(lambda: per_dimension_facets(session, stmt, entity))
        t_new = median_seconds(lambda: collect_facets(session, stmt, entity))
        match = "same" if old == new else "DIFFERENT"
        typer.echo(f"{name}: old {t_old:.3f}s, new {t_new:.3f}s ({t_old / t_new:.1f}x), results {match}")


if __name__ == "__main__":
    typer.run(main)
