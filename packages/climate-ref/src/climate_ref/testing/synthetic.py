"""
Build large synthetic REF databases for performance testing the read layer.

The row counts and distributions come from a `Shape`,
which `scripts/profile-database-shape.py` measures from a real database,
so a synthetic database can match production without copying any of its data.

The schema comes from the real migrations, so every index matches a deployed database.
Rows are written with raw `sqlite3` because the ORM is far too slow at this size.
Statistics are gathered with `Database.optimize`, as after an ingest or a solve,
because the query planner behaves differently without them.

Downstream projects such as the API use `cached_synthetic_database` to performance test their own reads,
and `captured_statements` with `large_table_scans` to check those reads use an index.
"""

import datetime
import hashlib
import json
import random
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import attrs
from sqlalchemy import event

import climate_ref
from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models.dataset import Dataset
from climate_ref.testing._files import build_cached_file
from climate_ref_core.source_types import SourceDatasetType

_EPOCH = datetime.datetime(2025, 6, 1)
_SPAN_SECONDS = 400 * 24 * 3600
_PROMOTED_VERSION = 2
_OOM_FRACTION = 0.1

DEFAULT_SHAPE = Path(__file__).parent / "shape.json"
"""Shape profiled from the deployed database."""

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
"""Tables large enough in production that reading every row is too slow for an interactive read."""

_SCAN = re.compile(r"^SCAN (\w+)")
_NAMED_SUBQUERY = re.compile(r"(\w+)\s+AS\s*\(|\)\s*AS\s+(\w+)", re.IGNORECASE)
# SQLAlchemy aliases a repeated table as ``<table>_<n>``.
_ALIAS_SUFFIX = re.compile(r"_\d+$")


@attrs.frozen(kw_only=True)
class Shape:
    """
    Row counts and distributions describing a REF database.

    Counts are for the full-size database and are multiplied by the ``scale`` passed to
    `build_synthetic_database`.
    Cardinalities (providers, diagnostics, dimension values) do not scale.
    """

    source: str
    """Where the numbers came from, e.g. a profiled database or ``"estimate"``."""

    providers: int
    diagnostics: int
    execution_groups: int

    executions_per_group: Mapping[str, float]
    """Fraction of groups with each number of executions, keyed by the count as a string."""

    superseded_group_fraction: float
    """Fraction of groups at an older diagnostic version than the promoted one."""

    dirty_group_fraction: float
    failed_execution_fraction: float
    running_execution_fraction: float
    retracted_execution_fraction: float

    measured_execution_fraction: float
    """Fraction of executions carrying resource measurements."""

    scalar_values: int
    series_values: int
    series_length_mean: int
    index_axes: int
    reference_series_fraction: float

    value_share_by_diagnostic: Sequence[float]
    """Share of all metric values held by each diagnostic, largest first."""

    dimension_cardinality: Mapping[str, int]
    """Distinct values of each populated CV dimension."""

    dimension_fill: Mapping[str, float]
    """Fraction of metric values with each CV dimension set."""

    attributes_bytes_mean: int

    datasets: Mapping[str, int]
    """Dataset count for each ``dataset_type`` (enum name, e.g. ``CMIP6``)."""

    files_per_dataset: float
    datasets_per_execution: float
    outputs_per_execution: float

    @classmethod
    def from_json(cls, path: Path) -> "Shape":
        """Load a shape written by `scripts/profile-database-shape.py`."""
        return cls(**json.loads(path.read_text()))


def _timestamp(fraction: float) -> str:
    return str(_EPOCH + datetime.timedelta(seconds=fraction * _SPAN_SECONDS))


def _insert(
    con: sqlite3.Connection, table: str, columns: Sequence[str], rows: Iterable[tuple[Any, ...]]
) -> None:
    quoted = ", ".join(f'"{c}"' for c in columns)
    sql = f'INSERT INTO "{table}" ({quoted}) VALUES ({", ".join("?" * len(columns))})'  # noqa: S608
    con.executemany(sql, rows)


def _migrated_schema(path: Path) -> None:
    config = Config()
    config.db.database_url = f"sqlite:///{path}"
    Database.from_config(config, run_migrations=True, skip_backup=True).close()


def build_synthetic_database(path: Path, shape: Shape, *, scale: float = 1.0, seed: int = 0) -> None:
    """
    Write a synthetic database matching ``shape`` to ``path``.

    Parameters
    ----------
    path
        Destination SQLite file. It must not exist yet.
    shape
        Row counts and distributions to reproduce.
    scale
        Multiplier for the row counts, so ``0.1`` is a tenth of the profiled database.
    seed
        Seed for the random generator, so a given shape and scale always gives the same database.
    """
    if path.exists():
        raise FileExistsError(path)
    _migrated_schema(path)
    rng = random.Random(seed)  # noqa: S311

    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    # Index maintenance during bulk insert is several times slower than rebuilding afterwards.
    indexes = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
    ).fetchall()
    for name, _ in indexes:
        con.execute(f'DROP INDEX "{name}"')

    with con:
        _Writer(con, shape, scale, rng).write()
        for _, sql in indexes:
            con.execute(sql)
    con.close()

    config = Config()
    config.db.database_url = f"sqlite:///{path}"
    database = Database.from_config(config, run_migrations=False)
    database.optimize()
    database.close()


def cached_synthetic_database(cache_dir: Path, *, scale: float, shape_path: Path = DEFAULT_SHAPE) -> Path:
    """
    Return a synthetic database for this shape and scale, building it on first use.

    The file name includes a hash of the shape, the scale, this generator and the migrations,
    so a change to any of them builds a new database.
    Old databases are not removed.

    Parameters
    ----------
    cache_dir
        Directory holding the built databases. It is created if missing.
    scale
        Multiplier for the row counts, passed to `build_synthetic_database`.
    shape_path
        JSON file written by `scripts/profile-database-shape.py`.

    Returns
    -------
    :
        Path to the SQLite file.
    """
    digest = hashlib.sha256()
    digest.update(shape_path.read_bytes())
    digest.update(str(scale).encode())
    digest.update(Path(__file__).read_bytes())
    for migration in sorted((Path(climate_ref.__file__).parent / "migrations" / "versions").glob("*.py")):
        digest.update(migration.read_bytes())

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"synthetic-{scale:g}-{digest.hexdigest()[:12]}.db"
    build_cached_file(
        path, lambda tmp: build_synthetic_database(tmp, Shape.from_json(shape_path), scale=scale)
    )
    return path


@contextmanager
def captured_statements(database: Database) -> Iterator[list[tuple[str, Any]]]:
    """
    Collect the SQL and bound parameters of every statement run on ``database`` inside the block.

    Pass the result to `large_table_scans`.
    """
    captured: list[tuple[str, Any]] = []

    def _capture(  # noqa: PLR0913
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        captured.append((statement, parameters))

    event.listen(database._engine, "before_cursor_execute", _capture)
    try:
        yield captured
    finally:
        event.remove(database._engine, "before_cursor_execute", _capture)


def large_table_scans(path: Path, statements: Iterable[tuple[str, Any]]) -> frozenset[str]:
    """
    Return the large tables that any of these statements reads in full, from SQLite's query plans.

    A full scan of one of a table's indexes counts as reading the table.

    Parameters
    ----------
    path
        The SQLite database the statements ran against.
    statements
        SQL and bound parameters, as collected by `captured_statements`.

    Raises
    ------
    ValueError
        If a plan scans a name that is neither a table nor a named subquery,
        because an alias would otherwise hide a scan from this check.
    """
    statements = list(statements)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {name for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        scanned = {
            _ALIAS_SUFFIX.sub("", match.group(1))
            for statement, parameters in statements
            for row in con.execute(f"EXPLAIN QUERY PLAN {statement}", parameters or ())
            if (match := _SCAN.match(row[3]))
        }
    finally:
        con.close()

    # A named subquery's own plan lists the tables it reads, so a scan of it adds nothing.
    subqueries = {
        _ALIAS_SUFFIX.sub("", n)
        for statement, _ in statements
        for m in _NAMED_SUBQUERY.findall(statement)
        for n in m
        if n
    }
    unknown = {n for n in scanned - tables - subqueries if n != "CONSTANT" and not n.startswith("anon")}
    if unknown:
        raise ValueError(f"cannot map scanned names {sorted(unknown)} to tables")
    return frozenset(scanned) & LARGE_TABLES


class _Writer:
    def __init__(self, con: sqlite3.Connection, shape: Shape, scale: float, rng: random.Random) -> None:
        self.con = con
        self.shape = shape
        self.rng = rng
        self.n_groups = max(shape.diagnostics, round(shape.execution_groups * scale))
        self.n_scalars = round(shape.scalar_values * scale)
        self.n_series = round(shape.series_values * scale)
        self.datasets = {k: max(1, round(v * scale)) for k, v in shape.datasets.items() if v}

    def write(self) -> None:
        self._diagnostics()
        self._groups_and_executions()
        self._datasets()
        self._execution_links()
        self._values()

    def _diagnostics(self) -> None:
        s = self.shape
        _insert(
            self.con,
            "provider",
            ["id", "slug", "name", "version"],
            ((i, f"provider-{i}", f"Provider {i}", "1.0") for i in range(1, s.providers + 1)),
        )
        _insert(
            self.con,
            "diagnostic",
            ["id", "slug", "name", "provider_id", "enabled", "promoted_version"],
            (
                (i, f"diagnostic-{i}", f"Diagnostic {i}", 1 + (i - 1) % s.providers, True, _PROMOTED_VERSION)
                for i in range(1, s.diagnostics + 1)
            ),
        )

    def _groups_and_executions(self) -> None:
        s, rng = self.shape, self.rng
        counts = [int(k) for k in s.executions_per_group]
        weights = list(s.executions_per_group.values())

        groups: list[tuple[Any, ...]] = []
        executions: list[tuple[Any, ...]] = []
        self.successful_executions: dict[int, list[int]] = {}
        for gid in range(1, self.n_groups + 1):
            diagnostic_id = 1 + (gid - 1) % s.diagnostics
            version = (
                _PROMOTED_VERSION - 1 if rng.random() < s.superseded_group_fraction else _PROMOTED_VERSION
            )
            created = gid / self.n_groups
            groups.append(
                (
                    gid,
                    diagnostic_id,
                    f"key-{gid}",
                    rng.random() < s.dirty_group_fraction,
                    json.dumps({"cmip6": [["source_id", f"source-{gid % 50}"], ["member_id", "r1i1p1f1"]]}),
                    _timestamp(created),
                    _timestamp(created),
                    version,
                )
            )
            for n in range(rng.choices(counts, weights)[0]):
                eid = len(executions) + 1
                roll = rng.random()
                if roll < s.running_execution_fraction:
                    successful = None
                elif roll < s.running_execution_fraction + s.failed_execution_fraction:
                    successful = False
                else:
                    successful = True
                measured = rng.random() < s.measured_execution_fraction
                wall = rng.lognormvariate(4, 1.5) if measured else None
                executions.append(
                    (
                        eid,
                        f"provider/diagnostic/{gid}/{eid}",
                        gid,
                        f"hash-{gid}-{n}",
                        successful,
                        rng.random() < s.retracted_execution_fraction,
                        _timestamp(min(1.0, created + n * 0.01)),
                        _timestamp(min(1.0, created + n * 0.01)),
                        wall,
                        wall * rng.uniform(0.5, 4) if wall else None,
                        int(rng.lognormvariate(21, 1)) if measured else None,
                        "proc_tree" if measured else None,
                        "oom" if successful is False and rng.random() < _OOM_FRACTION else None,
                    )
                )
                if successful and version == _PROMOTED_VERSION:
                    self.successful_executions.setdefault(diagnostic_id, []).append(eid)
        self.n_executions = len(executions)

        _insert(
            self.con,
            "execution_group",
            [
                "id",
                "diagnostic_id",
                "key",
                "dirty",
                "selectors",
                "created_at",
                "updated_at",
                "diagnostic_version",
            ],
            groups,
        )
        _insert(
            self.con,
            "execution",
            [
                "id",
                "output_fragment",
                "execution_group_id",
                "dataset_hash",
                "successful",
                "retracted",
                "created_at",
                "updated_at",
                "wall_seconds",
                "cpu_seconds",
                "peak_memory_bytes",
                "memory_source",
                "failure_reason",
            ],
            executions,
        )

    def _datasets(self) -> None:
        rng = self.rng
        dataset_id = 0
        self.dataset_ids: list[int] = []
        for dataset_type, count in self.datasets.items():
            table = Dataset.__mapper__.polymorphic_map[SourceDatasetType[dataset_type]].class_.__tablename__
            first = dataset_id + 1
            dataset_id += count
            ids = range(first, dataset_id + 1)
            self.dataset_ids.extend(ids)
            _insert(
                self.con,
                "dataset",
                ["id", "slug", "dataset_type", "finalised", "version_key"],
                ((i, f"{dataset_type.lower()}.dataset-{i}.v{i % 3}", dataset_type, True, i % 3) for i in ids),
            )
            columns = [
                r[1] for r in self.con.execute(f'PRAGMA table_info("{table}")') if r[3] and r[1] != "id"
            ]
            _insert(
                self.con,
                table,
                ["id", *columns],
                ((i, *(_facet(c, i) for c in columns)) for i in ids),
            )

        def files() -> Iterator[tuple[Any, ...]]:
            for i in self.dataset_ids:
                for f in range(max(1, round(rng.expovariate(1 / self.shape.files_per_dataset)))):
                    yield i, f"/data/{i}/file_{f}.nc", f"{1850 + f * 10}-01-01", f"{1859 + f * 10}-12-31"

        _insert(self.con, "dataset_file", ["dataset_id", "path", "start_time", "end_time"], files())

    def _execution_links(self) -> None:
        s, rng = self.shape, self.rng

        def links() -> Iterator[tuple[int, int]]:
            for eid in range(1, self.n_executions + 1):
                for did in rng.sample(
                    self.dataset_ids, min(len(self.dataset_ids), max(1, round(s.datasets_per_execution)))
                ):
                    yield eid, did

        def outputs() -> Iterator[tuple[Any, ...]]:
            for eid in range(1, self.n_executions + 1):
                for o in range(
                    round(rng.expovariate(1 / s.outputs_per_execution)) if s.outputs_per_execution else 0
                ):
                    yield (
                        eid,
                        "Plot" if o % 3 else "Data",
                        f"output_{o}.png",
                        f"output-{o}",
                        f"Output {o}",
                        f"Description of output {o}",
                    )

        _insert(self.con, "execution_dataset", ["execution_id", "dataset_id"], links())
        _insert(
            self.con,
            "execution_output",
            ["execution_id", "output_type", "filename", "short_name", "long_name", "description"],
            outputs(),
        )

    def _values(self) -> None:
        s, rng = self.shape, self.rng
        dims = list(s.dimension_cardinality)
        diagnostics = [d for d in range(1, s.diagnostics + 1) if d in self.successful_executions]
        shares = list(s.value_share_by_diagnostic)[: len(diagnostics)] or [1.0]
        attributes = json.dumps({"note": "x" * max(0, s.attributes_bytes_mean - 12)})

        def execution_ids(n: int) -> Iterator[int]:
            # Fill each diagnostic's executions in order so a diagnostic's values are contiguous by id,
            # like values written execution by execution.
            for diagnostic_id, share in zip(diagnostics, shares):
                pool = self.successful_executions[diagnostic_id]
                quota = round(n * share / sum(shares))
                for k in range(quota):
                    yield pool[k * len(pool) // max(quota, 1)]

        labels = [(s.dimension_fill[d], _dimension_values(d, s.dimension_cardinality[d])) for d in dims]

        def dimension_values() -> list[str | None]:
            return [pool[rng.randrange(len(pool))] if rng.random() < fill else None for fill, pool in labels]

        created = _timestamp(0.5)
        scalar_rows = (
            (eid, "SCALAR", rng.gauss(10, 3), attributes, created, created, *dimension_values())
            for eid in execution_ids(self.n_scalars)
        )
        columns = ["execution_id", "type", "value", "attributes", "created_at", "updated_at", *dims]
        _insert(self.con, "metric_value", columns, scalar_rows)

        n_axes = max(1, round(s.index_axes))
        _insert(
            self.con,
            "index_axis",
            ["id", "hash", "name", "values", "length"],
            (
                (i, f"axis-{i}", "time", json.dumps(list(range(s.series_length_mean))), s.series_length_mean)
                for i in range(1, n_axes + 1)
            ),
        )
        # Serialising a fresh payload per row dominates build time, so rows reuse a pool of payloads.
        payloads = [
            json.dumps([round(rng.gauss(280, 10), 6) for _ in range(s.series_length_mean)])
            for _ in range(256)
        ]
        series_rows = (
            (
                eid,
                "SERIES",
                payloads[rng.randrange(len(payloads))],
                1 + rng.randrange(n_axes),
                f"ref-{rng.randrange(1000)}" if rng.random() < s.reference_series_fraction else None,
                attributes,
                created,
                created,
                *dimension_values(),
            )
            for eid in execution_ids(self.n_series)
        )
        columns = [
            "execution_id",
            "type",
            "values",
            "index_id",
            "reference_id",
            "attributes",
            "created_at",
            "updated_at",
            *dims,
        ]
        _insert(self.con, "metric_value", columns, series_rows)


def _dimension_values(dimension: str, cardinality: int) -> list[str]:
    if dimension == "kind":
        # The only values the CV allows.
        return ["model", "reference"]
    if dimension == "source_id":
        # Shares the names given to datasets, so a model's values and datasets line up.
        return [f"source-{k}" for k in range(cardinality)]
    return [f"{dimension}-{k}" for k in range(cardinality)]


def _facet(column: str, i: int) -> Any:
    if column == "vertical_levels":
        return 1
    if column == "source_id":
        return f"source-{i % 50}"
    if column == "member_id":
        return f"r{1 + i % 5}i1p1f1"
    return f"{column}-{i % 20}"
