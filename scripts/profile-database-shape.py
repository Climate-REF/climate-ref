"""
Measure the shape of a REF SQLite database for the synthetic reader performance databases.

Prints JSON that `Shape.from_json` in `packages/climate-ref/tests/perf/synthetic.py` loads.
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

    def frac(sql: str, denominator: int, digits: int = 4) -> float:
        return round(one(sql) / denominator, digits)

    groups = one("SELECT count(*) FROM execution_group") or 1
    executions = one("SELECT count(*) FROM execution") or 1
    failed, running, retracted, measured = con.execute(
        "SELECT count(*) FILTER (WHERE successful = 0), count(*) FILTER (WHERE successful IS NULL), "
        "count(*) FILTER (WHERE retracted), count(wall_seconds) FROM execution"
    ).fetchone()

    # One pass over the largest table, because every extra pass reads gigabytes.
    dims = [r[1] for r in con.execute("PRAGMA table_info(metric_value)") if r[1] not in _NOT_DIMENSIONS]
    columns = ", ".join(
        [
            "count(*) FILTER (WHERE type = 'SCALAR')",
            "count(*) FILTER (WHERE type = 'SERIES')",
            "count(reference_id)",
            "sum(length(attributes))",
            """sum(json_array_length("values")) FILTER (WHERE type = 'SERIES')""",
            *(f'count(DISTINCT "{d}"), count("{d}")' for d in dims),
        ]
    )
    sql = f"SELECT {columns} FROM metric_value"  # noqa: S608
    scalars, series_count, references, attributes_bytes, series_points, *totals = con.execute(sql).fetchone()
    values = {"SCALAR": scalars, "SERIES": series_count}
    total_values = sum(values.values()) or 1
    series = values.get("SERIES", 0) or 1

    cardinality, fill = {}, {}
    for d, distinct, filled in zip(dims, totals[0::2], totals[1::2]):
        if filled:
            cardinality[d] = distinct
            fill[d] = round(filled / total_values, 4)

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

    shape = {
        "source": "profiled",
        "providers": one("SELECT count(*) FROM provider"),
        "diagnostics": one("SELECT count(*) FROM diagnostic"),
        "execution_groups": groups,
        "executions_per_group": {str(n): round(c / groups, 4) for n, c in per_group},
        "superseded_group_fraction": frac(
            "SELECT count(*) FROM execution_group g JOIN diagnostic d ON d.id = g.diagnostic_id "
            "WHERE g.diagnostic_version != d.promoted_version",
            groups,
        ),
        "dirty_group_fraction": frac("SELECT count(*) FROM execution_group WHERE dirty", groups),
        "failed_execution_fraction": round(failed / executions, 4),
        "running_execution_fraction": round(running / executions, 4),
        "retracted_execution_fraction": round(retracted / executions, 4),
        "measured_execution_fraction": round(measured / executions, 4),
        "scalar_values": values.get("SCALAR", 0),
        "series_values": values.get("SERIES", 0),
        "series_length_mean": round((series_points or 0) / series),
        "index_axes": one("SELECT count(*) FROM index_axis"),
        "reference_series_fraction": round(references / series, 4),
        "value_share_by_diagnostic": [round(s, 5) for s in shares],
        "dimension_cardinality": cardinality,
        "dimension_fill": fill,
        "attributes_bytes_mean": round((attributes_bytes or 0) / total_values),
        "datasets": dict(con.execute("SELECT dataset_type, count(*) FROM dataset GROUP BY 1").fetchall()),
        "files_per_dataset": frac(
            "SELECT count(*) FROM dataset_file", one("SELECT count(*) FROM dataset") or 1, 2
        ),
        "datasets_per_execution": frac("SELECT count(*) FROM execution_dataset", executions, 2),
        "outputs_per_execution": frac("SELECT count(*) FROM execution_output", executions, 2),
    }
    json.dump(shape, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
