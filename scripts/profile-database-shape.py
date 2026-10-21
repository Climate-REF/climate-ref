"""
Measure the shape of a REF SQLite database for the synthetic reader performance databases.

Prints JSON that `climate_ref.synthetic.Shape.from_json` loads.
The output holds only counts and distributions, never row contents, so it is safe to commit.

This uses only the standard library so it runs inside any deployed image, e.g.

    kubectl exec -i <pod running climate-ref> -- \
        python - <path to climate_ref.db> < scripts/profile-database-shape.py \
        > packages/climate-ref/tests/perf/shape.json

The database is opened read-only. Expect it to read every table once.
"""

import json
import sqlite3
import sys

_NOT_DIMENSIONS = {
    "id",
    "execution_id",
    "value",
    "attributes",
    "created_at",
    "updated_at",
    "values",
    "type",
    "index_id",
    "reference_id",
}


def main(path: str) -> None:
    """Print the shape of the database at ``path``."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def one(sql: str) -> object:
        return con.execute(sql).fetchone()[0]

    groups = one("SELECT count(*) FROM execution_group") or 1
    executions = one("SELECT count(*) FROM execution") or 1
    values = dict(con.execute("SELECT type, count(*) FROM metric_value GROUP BY type").fetchall())
    total_values = sum(values.values()) or 1

    per_group = con.execute(
        "SELECT n, count(*) FROM (SELECT count(e.id) n FROM execution_group g "
        "LEFT JOIN execution e ON e.execution_group_id = g.id GROUP BY g.id) GROUP BY n"
    ).fetchall()

    shares = [
        n / total_values
        for (n,) in con.execute(
            "SELECT count(*) FROM metric_value mv JOIN execution e ON e.id = mv.execution_id "
            "JOIN execution_group g ON g.id = e.execution_group_id GROUP BY g.diagnostic_id ORDER BY 1 DESC"
        )
    ]

    dims = [r[1] for r in con.execute("PRAGMA table_info(metric_value)") if r[1] not in _NOT_DIMENSIONS]
    cardinality, fill = {}, {}
    for d in dims:
        distinct, filled = con.execute(
            f'SELECT count(DISTINCT "{d}"), count("{d}") FROM metric_value'  # noqa: S608
        ).fetchone()
        if filled:
            cardinality[d] = distinct
            fill[d] = round(filled / total_values, 4)

    series = values.get("SERIES", 0) or 1
    shape = {
        "source": "profiled",
        "providers": one("SELECT count(*) FROM provider"),
        "diagnostics": one("SELECT count(*) FROM diagnostic"),
        "execution_groups": groups,
        "executions_per_group": {str(n): round(c / groups, 4) for n, c in per_group},
        "superseded_group_fraction": round(
            one(
                "SELECT count(*) FROM execution_group g JOIN diagnostic d ON d.id = g.diagnostic_id "
                "WHERE g.diagnostic_version != d.promoted_version"
            )
            / groups,
            4,
        ),
        "dirty_group_fraction": round(one("SELECT count(*) FROM execution_group WHERE dirty") / groups, 4),
        "failed_execution_fraction": round(
            one("SELECT count(*) FROM execution WHERE successful = 0") / executions, 4
        ),
        "running_execution_fraction": round(
            one("SELECT count(*) FROM execution WHERE successful IS NULL") / executions, 4
        ),
        "retracted_execution_fraction": round(
            one("SELECT count(*) FROM execution WHERE retracted") / executions, 4
        ),
        "measured_execution_fraction": round(
            one("SELECT count(*) FROM execution WHERE wall_seconds IS NOT NULL") / executions, 4
        ),
        "scalar_values": values.get("SCALAR", 0),
        "series_values": values.get("SERIES", 0),
        "series_length_mean": round(
            one("SELECT avg(json_array_length(\"values\")) FROM metric_value WHERE type = 'SERIES'") or 0
        ),
        "index_axes": one("SELECT count(*) FROM index_axis"),
        "reference_series_fraction": round(
            one("SELECT count(*) FROM metric_value WHERE reference_id IS NOT NULL") / series, 4
        ),
        "value_share_by_diagnostic": [round(s, 5) for s in shares],
        "dimension_cardinality": cardinality,
        "dimension_fill": fill,
        "attributes_bytes_mean": round(one("SELECT avg(length(attributes)) FROM metric_value") or 0),
        "datasets": dict(con.execute("SELECT dataset_type, count(*) FROM dataset GROUP BY 1").fetchall()),
        "files_per_dataset": round(
            one("SELECT count(*) FROM dataset_file") / (one("SELECT count(*) FROM dataset") or 1), 2
        ),
        "datasets_per_execution": round(one("SELECT count(*) FROM execution_dataset") / executions, 2),
        "outputs_per_execution": round(one("SELECT count(*) FROM execution_output") / executions, 2),
    }
    json.dump(shape, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
