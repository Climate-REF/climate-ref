"""
Fixtures for the reader performance tests, which run against a large synthetic database.

These only run when ``REF_PERF_SCALE`` is set (see ``make test-reader-perf``).
The database is built once per shape, scale and schema, and cached under ``REF_PERF_CACHE``.
"""

import json
import os
import re
import sqlite3
import statistics
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import attrs
import pytest

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.results import Reader
from climate_ref.testing.synthetic import cached_synthetic_database, captured_statements, large_table_scans

_SCAN = re.compile(r"^SCAN (\w+)")
_results: list[dict[str, Any]] = []


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Print the timings and table scans of every scenario that ran."""
    if not _results:
        return
    terminalreporter.section("reader performance")
    for r in sorted(_results, key=lambda r: -r["median_seconds"]):
        scans = ", ".join(r["large_table_scans"]) or "-"
        ms = r["median_seconds"] * 1000
        terminalreporter.write_line(
            f"{ms:9.1f} ms  {r['queries']:3d} queries  scans: {scans:40s} {r['scenario']}"
        )
    report = os.environ.get("REF_PERF_REPORT")
    if report:
        Path(report).write_text(json.dumps(_results, indent=2) + "\n")


@pytest.fixture(scope="session")
def perf_database_path() -> Path:
    """The synthetic database for this shape and scale, built on first use."""
    if not os.environ.get("REF_PERF_SCALE"):
        pytest.skip("set REF_PERF_SCALE to run the reader performance tests")
    cache = Path(os.environ.get("REF_PERF_CACHE", Path.home() / ".cache" / "climate-ref" / "perf"))
    return cached_synthetic_database(cache, scale=float(os.environ["REF_PERF_SCALE"]))


@pytest.fixture(scope="session")
def perf_database(perf_database_path: Path) -> Iterator[Database]:
    """A read-only connection to the synthetic database, like the deployed API."""
    config = Config()
    config.db.database_url = f"sqlite:///{perf_database_path}"
    database = Database.from_config(config, read_only=True)
    yield database
    database.close()


@attrs.frozen(kw_only=True)
class Targets:
    """Ids and slugs that the scenarios query, picked from the generated data."""

    provider_slug: str
    diagnostic_slug: str
    """The diagnostic with the most metric values, the worst case for a diagnostic-scoped read."""

    execution_id: int
    """The execution with the most metric values."""

    execution_group_id: int
    dataset_slug: str
    source_id: str


@pytest.fixture(scope="session")
def targets(perf_database_path: Path) -> Targets:
    """Pick the busiest diagnostic and execution, so each scenario reads its worst case."""
    con = sqlite3.connect(f"file:{perf_database_path}?mode=ro", uri=True)
    execution_id, _ = con.execute(
        "SELECT execution_id, count(*) FROM metric_value GROUP BY execution_id ORDER BY 2 DESC LIMIT 1"
    ).fetchone()
    group_id, diagnostic_slug, provider_slug = con.execute(
        "SELECT g.id, d.slug, p.slug FROM execution e JOIN execution_group g ON g.id = e.execution_group_id "
        "JOIN diagnostic d ON d.id = g.diagnostic_id JOIN provider p ON p.id = d.provider_id WHERE e.id = ?",
        (execution_id,),
    ).fetchone()
    (dataset_slug,) = con.execute("SELECT slug FROM dataset ORDER BY id DESC LIMIT 1").fetchone()
    (source_id,) = con.execute(
        "SELECT source_id FROM metric_value WHERE execution_id = ? AND source_id IS NOT NULL LIMIT 1",
        (execution_id,),
    ).fetchone()
    con.close()
    return Targets(
        provider_slug=provider_slug,
        diagnostic_slug=diagnostic_slug,
        execution_id=execution_id,
        execution_group_id=group_id,
        dataset_slug=dataset_slug,
        source_id=source_id,
    )


@attrs.frozen(kw_only=True)
class Measurement:
    """What one scenario cost."""

    seconds: tuple[float, ...]
    queries: int
    large_table_scans: frozenset[str]
    """Large tables read in full by any statement."""

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.seconds)


@pytest.fixture(scope="session")
def measure(perf_database: Database, perf_database_path: Path) -> Callable[..., Measurement]:
    """
    Run a reader call several times through fresh sessions and collect its timings and table scans.

    ``REF_PERF_REPEAT`` sets the number of runs (default 3).
    """
    repeat = int(os.environ.get("REF_PERF_REPEAT", "3"))

    def _measure(scenario: str, call: Callable[[Reader], object]) -> Measurement:
        def _time() -> float:
            with perf_database.session_scope() as session:
                start = time.perf_counter()
                call(Reader(perf_database, session=session))
                return time.perf_counter() - start

        with captured_statements(perf_database) as statements:
            seconds = [_time()]
        seconds += [_time() for _ in range(repeat - 1)]

        measurement = Measurement(
            seconds=tuple(seconds),
            queries=len(statements),
            large_table_scans=large_table_scans(perf_database_path, statements),
        )
        _results.append(
            {
                "scenario": scenario,
                "median_seconds": measurement.median_seconds,
                "seconds": list(measurement.seconds),
                "queries": measurement.queries,
                "large_table_scans": sorted(measurement.large_table_scans),
            }
        )
        return measurement

    return _measure
