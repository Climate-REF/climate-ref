"""
Read the results of a REF run into pandas.

``Reader`` provides an interface to query and filter the results in a local REF database.
Each read returns a collection that you can loop over or turn into a DataFrame with ``to_pandas()``.
The results are plain Python objects, so you can keep them after the database is closed.

What you can read, as properties of ``Reader``:

* ``reader.values``: metric values, as scalars or series, in [values][climate_ref.results.values].
* ``reader.executions``: execution groups, executions and their output files,
  in [executions][climate_ref.results.executions].
* ``reader.datasets``: the datasets that have been ingested, in [datasets][climate_ref.results.datasets].
* ``reader.diagnostics``: the diagnostics and how many of their runs succeeded or failed,
  in [diagnostics][climate_ref.results.diagnostics].
* ``reader.resources``: CPU and memory use of runs, in [resources][climate_ref.results.resources].
* ``reader.artifacts``: paths of the output files of a run, in [artifacts][climate_ref.results.artifacts].

Narrow a read with the filters in [filters][climate_ref.results.filters].
Flag unusual scalar values with [outliers][climate_ref.results.outliers].
The DataFrame layout is set in [frames][climate_ref.results.frames].
To get database rows instead of reader objects, use the builders in [queries][climate_ref.results.queries].

Examples
--------
Compare two models on one diagnostic, as a table:

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
    groups = reader.executions.groups().to_pandas()
```

Find where the log of a run is. This needs the results folder:

```python
with Database.from_config(config, read_only=True) as db:
    reader = Reader(db, results=config.paths.results)
    execution = reader.executions.execution(execution_id=42)
    log = reader.artifacts.log_file(execution.output_fragment)
```

Public surface convention
-------------------------
The top level exports only what you name to make a call:

* the ``Reader`` entry point,
* the filters you construct and pass in:
  ``MetricValueFilter``, ``ExecutionGroupFilter``, ``DatasetFilter``, ``DiagnosticFilter``,
  ``ResourceFilter`` and ``InputDatasetFilter``,
* the ``OutlierPolicy`` value object.

Everything a reader returns lives in its own submodule.
Import it from there when you need to name it,
e.g. ``from climate_ref.results.values import ScalarValue``.
"""

from climate_ref.results.datasets import DatasetFilter
from climate_ref.results.filters import (
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
