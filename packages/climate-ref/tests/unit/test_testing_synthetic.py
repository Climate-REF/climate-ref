import sqlite3
from pathlib import Path

import pytest

from climate_ref.database import Database
from climate_ref.results import MetricValueFilter, Reader
from climate_ref.testing.synthetic import (
    DEFAULT_SHAPE,
    Shape,
    build_synthetic_database,
    cached_synthetic_database,
    large_table_scans,
)

SCALE = 0.001


@pytest.fixture
def shape():
    return Shape.from_json(DEFAULT_SHAPE)


@pytest.fixture
def synthetic_db(tmp_path: Path, shape) -> Path:
    path = tmp_path / "synthetic.db"
    build_synthetic_database(path, shape, scale=SCALE)
    return path


def test_row_counts_follow_the_shape(synthetic_db, shape):
    con = sqlite3.connect(synthetic_db)
    counts = dict(con.execute("SELECT type, count(*) FROM metric_value GROUP BY type").fetchall())
    (groups,) = con.execute("SELECT count(*) FROM execution_group").fetchone()
    (diagnostics,) = con.execute("SELECT count(*) FROM diagnostic").fetchone()
    indexes = {name for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    con.close()

    assert counts["SCALAR"] == pytest.approx(shape.scalar_values * SCALE, rel=0.05)
    assert counts["SERIES"] == pytest.approx(shape.series_values * SCALE, rel=0.05)
    assert groups == max(shape.diagnostics, round(shape.execution_groups * SCALE))
    assert diagnostics == shape.diagnostics
    assert "ix_metric_value_execution_id_type" in indexes


def test_reader_reads_the_database(synthetic_db, config):
    config.db.database_url = f"sqlite:///{synthetic_db}"
    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)
        values = reader.values.scalar_values(MetricValueFilter(include_retracted=True), limit=10)
        stats = reader.executions.statistics()

    assert len(values) == 10
    assert sum(s.total for s in stats) > 0


def test_refuses_to_overwrite(synthetic_db, shape):
    with pytest.raises(FileExistsError):
        build_synthetic_database(synthetic_db, shape, scale=SCALE)


def test_cached_database_is_built_once(tmp_path: Path):
    path = cached_synthetic_database(tmp_path, scale=SCALE)
    built = path.stat().st_mtime_ns

    assert cached_synthetic_database(tmp_path, scale=SCALE) == path
    assert path.stat().st_mtime_ns == built
    assert cached_synthetic_database(tmp_path, scale=SCALE * 2) != path


def test_large_table_scans_accepts_multiline_cte(synthetic_db):
    # SQLAlchemy puts a line break between ``AS`` and the CTE body.
    sql = (
        "WITH busiest AS\n(SELECT execution_id, count(*) AS n FROM metric_value"
        " WHERE id = 1 GROUP BY execution_id)"
        "\nSELECT * FROM busiest JOIN execution ON execution.id = busiest.execution_id"
    )
    assert large_table_scans(synthetic_db, [(sql, ())]) == frozenset()
