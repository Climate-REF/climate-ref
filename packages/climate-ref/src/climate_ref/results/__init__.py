"""
Read the results of diagnostic executions from a local REF database.

The REF stores its local results across several database tables:
diagnostics, execution groups, executions, metric values and datasets.
A ``Reader`` joins these tables and skips retracted executions,
so a read needs a filter rather than SQL.

## Reading results

Each property of a ``Reader`` reads one kind of result:

* ``reader.values``: the metric values each execution produced, as single numbers or as series.
* ``reader.executions``: the executions of each diagnostic and whether they succeeded.
* ``reader.diagnostics``: the diagnostics, with how many of their execution groups succeeded or failed.
* ``reader.datasets``: the datasets that have been ingested.
* ``reader.resources``: how much CPU time and memory the executions used.
* ``reader.artifacts``: where the output files of an execution are on disk.

Most reads return a collection.
Loop over it, or call ``to_pandas()`` on it to get a DataFrame.
A few, such as ``reader.resources.profiles``, return a plain tuple of views.

## Filtering

Most reads take a filter, such as ``MetricValueFilter``, to choose which results come back.
By default a filter keeps only each diagnostic's promoted version and leaves out retracted executions.
Readers of single executions also take ``latest_only`` to keep only each group's newest execution.
The filters are documented in [filters][climate_ref.query.filters].

## Pagination

Reads that return many rows are paged with ``offset`` and ``limit``.
By default ``limit`` is ``None``, which returns every matching row.
A collection's ``total_count`` is how many rows match the filter across all pages.
To get the next page, repeat the read with ``offset=page.offset + len(page)``.

## Plain objects instead of ORM objects

A read returns frozen data objects, such as
[ScalarValue][climate_ref.results.values.ScalarValue],
rather than the SQLAlchemy ORM objects that the database uses.
This is on purpose:

* They keep working after the database session is closed.
  An ORM object loads related rows lazily, so reading one of them after the session closes fails.
* Reading a field never runs a hidden query, because everything is loaded during the read.
* They cannot change the database. They are frozen and not attached to a session.
* Their fields can stay the same when the database tables change, so code that uses them does not break.

To work with the ORM objects, use the query builders in [climate_ref.query][climate_ref.query].
Each builder takes the same filter as its reader and returns a SQLAlchemy ``Select``,
which you can narrow further with ``.where()`` before running it in a session.

## Examples

Get the global mean values for two models as a DataFrame:

```python
from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.results import MetricValueFilter, Reader

config = Config.default()
filters = MetricValueFilter(
    provider_slug="example",
    diagnostic_slug="global-mean-timeseries",
    dimensions={"source_id": ["ACCESS-ESM1-5", "CanESM5"]},
)

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    df = reader.values.scalar_values(filters).to_pandas()
```

Find the log file of execution 42.
Reading files needs the results folder as well as the database:

```python
with Database.from_config(config, read_only=True) as db:
    reader = Reader(db, results=config.paths.results)
    execution = reader.executions.execution(execution_id=42)
    log = reader.artifacts.log_file(execution.output_fragment)
```
"""

from climate_ref.query.filters import (
    DatasetFilter,
    DiagnosticFilter,
    ExecutionGroupFilter,
    InputDatasetFilter,
    MetricValueFilter,
    ResourceFilter,
)
from climate_ref.results.outliers import OutlierPolicy
from climate_ref.results.reader import Reader

__all__ = [
    "DatasetFilter",
    "DiagnosticFilter",
    "ExecutionGroupFilter",
    "InputDatasetFilter",
    "MetricValueFilter",
    "OutlierPolicy",
    "Reader",
    "ResourceFilter",
]
