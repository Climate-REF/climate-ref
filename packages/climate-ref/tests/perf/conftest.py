"""
Fixtures for the reader performance tests, which run against a large synthetic database.

These only run when ``REF_PERF_SCALE`` is set (see ``make test-reader-perf``).
The database is built once per shape, scale and schema, and cached under ``REF_PERF_CACHE``.
"""

import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import attrs
import pytest
from sqlalchemy import event

import climate_ref
from climate_ref.config import Config
from climate_ref.conftest_plugin import _build_cached_file
from climate_ref.database import Database
from climate_ref.results import Reader

SHAPE_PATH = Path(__file__).parent / "shape.json"
SYNTHETIC_PATH = Path(__file__).parent / "synthetic.py"


def _load_synthetic() -> Any:
    # Kept out of the package so it is not a supported testing surface, and conftests cannot import siblings.
    spec = importlib.util.spec_from_file_location("reader_perf_synthetic", SYNTHETIC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


synthetic = _load_synthetic()

# Tables large enough in production that reading every row is too slow for an interactive read.
LARGE_TABLES = frozenset(
    {
        "metric_value",
        "index_axis",
        "execution",
        "execution_group",
        "execution_output",
        "execution_dataset",
        "dataset",
        "dataset_file",
        "cmip6_dataset",
        "cmip7_dataset",
        "obs4mips_dataset",
    }
)

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


def _cache_key(shape_text: str, scale: float) -> str:
    migrations = Path(climate_ref.__file__).parent / "migrations" / "versions"
    digest = hashlib.sha256()
    digest.update(shape_text.encode())
    digest.update(str(scale).encode())
    digest.update(SYNTHETIC_PATH.read_bytes())
    for p in sorted(migrations.glob("*.py")):
        digest.update(p.read_bytes())
    return digest.hexdigest()[:12]


@pytest.fixture(scope="session")
def synthetic_module() -> Any:
    """The synthetic database generator, for tests of the generator itself."""
    return synthetic


@pytest.fixture(scope="session")
def perf_database_path() -> Path:
    """The synthetic database for this shape and scale, built on first use."""
    if not os.environ.get("REF_PERF_SCALE"):
        pytest.skip("set REF_PERF_SCALE to run the reader performance tests")
    scale = float(os.environ["REF_PERF_SCALE"])
    shape_text = SHAPE_PATH.read_text()
    cache = Path(os.environ.get("REF_PERF_CACHE", Path.home() / ".cache" / "climate-ref" / "perf"))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"reader-perf-{scale:g}-{_cache_key(shape_text, scale)}.db"
    _build_cached_file(
        path,
        lambda tmp: synthetic.build_synthetic_database(
            tmp, synthetic.Shape.from_json(SHAPE_PATH), scale=scale
        ),
    )
    return path


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
    plans: tuple[tuple[str, ...], ...]

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.seconds)

    @property
    def scanned(self) -> frozenset[str]:
        """Everything read in full by any statement, including full scans of one of a table's indexes."""
        # SQLAlchemy aliases a repeated table as ``<table>_<n>``.
        return frozenset(
            re.sub(r"_\d+$", "", match.group(1))
            for plan in self.plans
            for match in map(_SCAN.match, plan)
            if match
        )

    @property
    def large_table_scans(self) -> frozenset[str]:
        """Large tables read in full by any statement."""
        return self.scanned & LARGE_TABLES


@pytest.fixture(scope="session")
def measure(perf_database: Database, perf_database_path: Path) -> Iterator[Callable[..., Measurement]]:
    """
    Run a reader call several times through fresh sessions and collect its timings and query plans.

    ``REF_PERF_REPEAT`` sets the number of runs (default 3).
    """
    repeat = int(os.environ.get("REF_PERF_REPEAT", "3"))
    explain = sqlite3.connect(f"file:{perf_database_path}?mode=ro", uri=True)
    tables = {name for (name,) in explain.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    def _measure(scenario: str, call: Callable[[Reader], object]) -> Measurement:
        captured: list[tuple[str, Any]] = []

        def _capture(  # noqa: PLR0913
            conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
        ) -> None:
            captured.append((statement, parameters))

        def _time() -> float:
            with perf_database.session_scope() as session:
                start = time.perf_counter()
                call(Reader(perf_database, session=session))
                return time.perf_counter() - start

        event.listen(perf_database._engine, "before_cursor_execute", _capture)
        try:
            seconds = [_time()]
        finally:
            event.remove(perf_database._engine, "before_cursor_execute", _capture)
        seconds += [_time() for _ in range(repeat - 1)]

        plans = tuple(
            tuple(row[3] for row in explain.execute(f"EXPLAIN QUERY PLAN {statement}", parameters or ()))
            for statement, parameters in captured
        )
        measurement = Measurement(seconds=tuple(seconds), plans=plans)
        # A named alias would hide a scan from the large-table check, so refuse names it cannot map.
        unknown = {
            n for n in measurement.scanned if n not in tables and n != "CONSTANT" and not n.startswith("anon")
        }
        if unknown:
            raise ValueError(f"cannot map scanned names {sorted(unknown)} to tables")
        _results.append(
            {
                "scenario": scenario,
                "median_seconds": measurement.median_seconds,
                "seconds": list(measurement.seconds),
                "queries": len(captured),
                "large_table_scans": sorted(measurement.large_table_scans),
            }
        )
        return measurement

    yield _measure
    explain.close()
