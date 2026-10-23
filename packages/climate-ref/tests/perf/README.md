# Reader performance tests

These tests time the `Reader` read paths against a synthetic database shaped like production.
Each scenario is one call the API or a notebook makes.
A scenario fails when:

- a query reads every row of a large table, which `EXPLAIN QUERY PLAN` shows at any scale,
- its median time exceeds its budget, which only means something near full scale.

## Running

```bash
make test-reader-perf                   # a tenth of production, quick
make test-reader-perf REF_PERF_SCALE=1  # full size, about 8 GB
```

The database is built on first use and cached in `~/.cache/climate-ref/perf` (`REF_PERF_CACHE`).
A change to `shape.json`, the scale, the generator or the migrations builds a new one.
Old databases are not removed, so clear the cache directory by hand.

Other settings:

- `REF_PERF_REPEAT`: runs per scenario, default 3. The median is checked.
- `REF_PERF_REPORT`: write the timings and table scans to this JSON file, for comparing branches.

## The shape

`shape.json` holds row counts and distributions, never row contents.
`synthetic.py` builds a database from it using the real migrations, so the indexes match production.
Table statistics are gathered with `PRAGMA optimize`, as a deployed database gets them after each ingest or solve.

Refresh it from the deployed database with:

```bash
kubectl exec -i <pod running climate-ref> -- \
    python - <path to climate_ref.db> < scripts/profile-database-shape.py \
    > packages/climate-ref/tests/perf/shape.json
```
