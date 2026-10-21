import sqlite3
from pathlib import Path

import pytest

from climate_ref.database import Database
from climate_ref.results import MetricValueFilter, Reader
from climate_ref.synthetic import Shape, build_synthetic_database

SHAPE = Shape.from_json(Path(__file__).parents[1] / "perf" / "shape.json")


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.db"
    build_synthetic_database(path, SHAPE, scale=0.001)
    return path


def test_row_counts_follow_the_shape(synthetic_db):
    con = sqlite3.connect(synthetic_db)
    counts = dict(con.execute("SELECT type, count(*) FROM metric_value GROUP BY type").fetchall())
    (groups,) = con.execute("SELECT count(*) FROM execution_group").fetchone()
    (diagnostics,) = con.execute("SELECT count(*) FROM diagnostic").fetchone()
    indexes = {name for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    con.close()

    assert counts["SCALAR"] == pytest.approx(SHAPE.scalar_values * 0.001, rel=0.05)
    assert counts["SERIES"] == pytest.approx(SHAPE.series_values * 0.001, rel=0.05)
    assert groups == max(SHAPE.diagnostics, round(SHAPE.execution_groups * 0.001))
    assert diagnostics == SHAPE.diagnostics
    assert "ix_metric_value_execution_id_type" in indexes


def test_reader_reads_the_database(synthetic_db, config):
    config.db.database_url = f"sqlite:///{synthetic_db}"
    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)
        values = reader.values.scalar_values(MetricValueFilter(include_retracted=True), limit=10)
        stats = reader.executions.statistics()

    assert len(values) == 10
    assert sum(s.total for s in stats) > 0


def test_refuses_to_overwrite(synthetic_db):
    with pytest.raises(FileExistsError):
        build_synthetic_database(synthetic_db, SHAPE, scale=0.001)
