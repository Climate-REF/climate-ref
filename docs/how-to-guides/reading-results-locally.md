# Reading results locally with pandas

The [`using-pre-computed-results`](using-pre-computed-results.py) guide pulls results from the
hosted [REF API](https://api.climate-ref.org).
If instead you have a local REF database because you have run diagnostics yourself,
or you are working against a copy of the database
you can read those results straight into pandas without standing up an API.

This is the job of `climate_ref.results`,
a typed read layer over the REF database.
It returns frozen, ORM-free value objects wrapped in collections that offer `to_pandas()`,
so a DataFrame built inside a `with Database(...)` block keeps working after the session closes.

!!! note

    This guide is not executed when the documentation is built,
    because the documentation build ingests datasets but does not run any diagnostics,
    so there are no execution results to read.
    Every snippet below has been checked against the current source of
    [`climate_ref.results`](../api/climate_ref/results/),
    but you will need your own populated database to run them.

## Opening a database and constructing a `Reader`

Open the configured database read-only and hand it to a `Reader`:

```python
from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.results import Reader

config = Config.default()

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    df = reader.values.scalar_values().to_pandas()

# `df` is a plain DataFrame and remains valid out here, after the session has closed.
```

`read_only=True` opens SQLite in immutable read-only mode and skips migrations,
so the read layer never mutates the database you point it at.

`Reader` is a thin entry point.
It holds the database and exposes one sub-reader per domain as a property:

- `reader.values` — scalar and series metric values,
- `reader.executions` — execution groups and executions,
- `reader.datasets` — ingested datasets,
- `reader.diagnostics` — diagnostics and their status counts,
- `reader.artifacts` — filesystem paths for execution outputs.

`reader.artifacts` needs to know where output files live,
so it is only available when you pass a results root to the constructor:

```python
with Database.from_config(config, read_only=True) as db:
    reader = Reader(db, results=config.paths.results)

    execution = reader.executions.execution(execution_id=42)
    for output in reader.executions.outputs(execution_id=42):
        path = reader.artifacts.output_file(execution.output_fragment, output.filename)
        print(path)
```

Constructing `Reader(db)` without a `results` root and then touching `reader.artifacts`
raises a `ValueError` telling you to supply one.

## Scalar and series values

`reader.values` reads the two shapes of metric value the REF stores:
single scalar numbers, and 1-d series along a shared index.

### Scalar values

`scalar_values()` returns a `ScalarValueCollection`.
Iterate over it for the individual `ScalarValue` objects,
or call `to_pandas()` for a tidy DataFrame with one row per value
and one column per controlled-vocabulary (CV) dimension present:

```python
from climate_ref.results import MetricValueFilter

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    collection = reader.values.scalar_values()

    print(collection.total_count)      # values matching the filter, before pagination
    print(collection.facets_dict())    # {dimension: [distinct values]} over the full set

    df = collection.to_pandas()
```

Pass a `MetricValueFilter` to narrow the query.
Every field is optional; `None`/empty means "do not constrain on this axis".
Dimension filters go through the `dimensions` map,
keyed by any registered CV dimension;
a string is an equality match and a sequence is an `IN` match:

```python
filters = MetricValueFilter(
    diagnostic_slug="global-mean-timeseries",   # exact-match slug
    provider_slug="example",
    dimensions={
        "statistic": "mean",                     # equality
        "source_id": ["ACCESS-ESM1-5", "CanESM5"],  # IN
    },
)

with Database.from_config(config, read_only=True) as db:
    df = Reader(db).values.scalar_values(filters=filters).to_pandas()
```

An unknown dimension key raises `KeyError` rather than silently returning the wrong data.
`diagnostic_slug`/`provider_slug` are exact matches;
`diagnostic_contains`/`provider_contains` are case-insensitive substring matches for search.

#### Outlier detection

Scalar reads can flag outliers using a source-id-aware IQR test,
configured with an `OutlierPolicy`.
Detection is off by default:

```python
from climate_ref.results import OutlierPolicy

with Database.from_config(config, read_only=True) as db:
    collection = Reader(db).values.scalar_values(
        outliers=OutlierPolicy(method="iqr", factor=10.0),
        include_unverified=True,   # keep flagged rows in the result (default drops them)
    )
    print(collection.had_outliers, collection.outlier_count)
    df = collection.to_pandas()    # gains `is_outlier` / `verification_status` columns
```

With `include_unverified=False` (the default) flagged values are removed before pagination
and excluded from `total_count`.
When detection runs, `to_pandas()` adds `is_outlier` and `verification_status` columns.

### Series values

`series_values()` returns a `SeriesValueCollection` of `SeriesValue` objects,
each carrying its `values`, the shared `index`, and the `index_name`:

```python
with Database.from_config(config, read_only=True) as db:
    collection = Reader(db).values.series_values(
        MetricValueFilter(reference_only=False)   # model series only; True for reference/observations
    )

    long_df = collection.to_pandas()                 # one row per (series, index point)
    wide_df = collection.to_pandas(explode=False)    # one row per series; list-valued cells
```

`reference_only` is a series-only axis:
`True` selects observation/reference series, `False` selects model series.

### Pagination

All value reads paginate through `offset`/`limit`,
and `total_count` reports the size of the full match so you can tell there are more rows.
Ordering is deterministic (id-tie-broken), so paging is stable across calls:

```python
with Database.from_config(config, read_only=True) as db:
    page = Reader(db).values.scalar_values(offset=0, limit=100)
```

## Executions

`reader.executions` reads execution groups and the individual executions within them.

`groups()` returns an `ExecutionGroupCollection`;
each `ExecutionGroupView` carries its `latest` execution (or `None` if it has never run):

```python
from climate_ref.results import ExecutionGroupFilter

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)

    groups = reader.executions.groups(
        filters=ExecutionGroupFilter(
            diagnostic_contains=["sea-ice"],   # case-insensitive substring, OR-combined
            successful=True,                    # keep groups whose latest execution succeeded
        )
    )
    df = groups.to_pandas()   # columns mirror the `list-groups` CLI

    for group in groups:
        print(group.id, group.diagnostic_slug, group.successful)
```

Fetch a single group or execution by id
(both return `None` when nothing matches),
or resolve the latest execution for a group directly:

```python
with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)

    group = reader.executions.group(execution_group_id=1)
    execution = reader.executions.execution(execution_id=42)
    latest = reader.executions.latest_execution(execution_group_id=1)
```

`group()`, `groups()`, `latest_execution()`, and `statistics()`
all agree on which execution is "latest" for a group
(ranked by `created_at DESC, id DESC`).

`statistics()` returns per-`(provider, diagnostic)` status counts as `ExecutionStats` objects:

```python
with Database.from_config(config, read_only=True) as db:
    stats = Reader(db).executions.statistics(provider_contains=["esmvaltool"])
    for row in stats:
        print(row.provider, row.diagnostic, row.successful, row.failed, row.running)
```

## Datasets

`reader.datasets` reads the datasets that have been ingested into the database.
`list()` returns a `DatasetCollection`;
`to_pandas()` gives one row per dataset with the base columns plus the source-type facets expanded:

```python
from climate_ref.results import DatasetFilter
from climate_ref_core.source_types import SourceDatasetType

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)

    datasets = reader.datasets.list(
        filters=DatasetFilter(source_type=SourceDatasetType.CMIP6)
    )
    df = datasets.to_pandas()
```

`DatasetFilter.source_type` is required:
a typed listing has to choose a source type, because facet columns are per-type.
This is a deliberate divergence from the other readers,
whose `filters` argument is optional and defaults to "everything".

By default `DatasetFilter.latest_only` is `True`,
so a listing returns exactly one row per dataset — the latest version — rather than every version,
deduplicated with that source type's version key.

`get(slug)` fetches a single dataset by slug, returning the latest version on a tie
(or `None` when no dataset has that slug):

```python
with Database.from_config(config, read_only=True) as db:
    dataset = Reader(db).datasets.get("CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Omon.tos.gn.v20210318")
```

Pass `include_files=True` to `list()` to populate each `DatasetView.files`
with its `DatasetFileView` entries.

## Diagnostics

`reader.diagnostics.list()` returns a `DiagnosticCollection` of `DiagnosticView` objects,
each joined to its provider and carrying execution-group counts and promoted-version status:

```python
from climate_ref.results import DiagnosticFilter

with Database.from_config(config, read_only=True) as db:
    diagnostics = Reader(db).diagnostics.list(
        filters=DiagnosticFilter(provider_contains=["pmp"])
    )
    df = diagnostics.to_pandas()

    for diagnostic in diagnostics:
        print(diagnostic.provider_slug, diagnostic.slug, diagnostic.successful, diagnostic.total)
```

## Artifacts: resolving output paths

`reader.artifacts` turns the fragments an execution stores
(`output_fragment`, the bundle `path`, an output `filename`)
into filesystem `Path`s under the results root.
It is available only when you construct `Reader(db, results=...)`.

```python
with Database.from_config(config, read_only=True) as db:
    reader = Reader(db, results=config.paths.results)

    execution = reader.executions.execution(execution_id=42)
    fragment = execution.output_fragment

    output_dir = reader.artifacts.output_directory(fragment)
    log_file = reader.artifacts.log_file(fragment)
    bundle = reader.artifacts.bundle(fragment, execution.path)   # None when no bundle recorded

    for output in reader.executions.outputs(execution_id=42):
        file_path = reader.artifacts.output_file(fragment, output.filename)
```

Every resolver is containment-guarded:
a fragment that would escape the results root raises `ValueError` rather than returning a path outside it.
The resolver returns paths only — opening and reading the files stays with you
(e.g. via `xarray.open_dataset`).

## Package conventions

`climate_ref.results` keeps its top-level namespace small.
It exports only what you *name to make a call*:

- the `Reader` entry point,
- the filter objects you construct and pass in
  (`MetricValueFilter`, `ExecutionGroupFilter`, `DatasetFilter`, `DiagnosticFilter`),
- the `OutlierPolicy` value object.

Everything the package *returns* — the DTOs (`ScalarValue`, `ExecutionGroupView`, `DatasetView`, …),
their collections, and the per-domain sub-reader classes reached via `reader.values` etc. —
lives in the relevant domain submodule.
Import them from there on the rare occasion you need to name one:

```python
from climate_ref.results.values import ScalarValue
from climate_ref.results.executions import ExecutionGroupView
```

All paginated reads share the same shape:
`offset`/`limit` control the page,
`total_count` reports the full match,
and ordering is deterministic (tie-broken by primary key) so paging is stable.

```
