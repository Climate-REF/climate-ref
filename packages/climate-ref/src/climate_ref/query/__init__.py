"""
Filters and the query builders behind every reader.

Each builder takes a filter and returns a SQLAlchemy ``Select``.
It never takes a session or runs the query, so you can filter it further with ``.where()``,
or add columns or loading options, before running it in a session.
The readers in [climate_ref.results][climate_ref.results] run these same builders,
then page the rows and turn them into plain objects.

The filters are documented in [filters][climate_ref.query.filters].

Builders over one kind of record return the ORM objects:

* [select_scalar_values][climate_ref.query.values.select_scalar_values] and
  [select_series_values][climate_ref.query.values.select_series_values]: metric values.
* [select_datasets][climate_ref.query.datasets.select_datasets]: datasets.
* [select_execution_groups][climate_ref.query.executions.select_execution_groups]:
  execution groups, each paired with its newest execution.
* [select_execution_outputs][climate_ref.query.executions.select_execution_outputs]: one execution's outputs.
* [select_latest_execution][climate_ref.query.executions.select_latest_execution]: a group's newest execution.

Builders that count or summarise select labelled columns, listed on each builder:

* [select_diagnostics][climate_ref.query.diagnostics.select_diagnostics],
  [select_group_counts][climate_ref.query.diagnostics.select_group_counts] and
  [select_diagnostics_with_values][climate_ref.query.diagnostics.select_diagnostics_with_values].
* [select_execution_resources][climate_ref.query.resources.select_execution_resources] and
  [select_resource_summary][climate_ref.query.resources.select_resource_summary].

[count][climate_ref.query.paging.count] and [paginate][climate_ref.query.paging.paginate]
give the ``total_count`` and paging that the readers use.

Examples
--------
Scalar values as ORM rows, narrowed further than ``MetricValueFilter`` allows:

```python
from climate_ref.models.metric_value import ScalarMetricValue
from climate_ref.query import MetricValueFilter, count, paginate, select_scalar_values

stmt = select_scalar_values(MetricValueFilter(provider_slug="pmp", latest_only=True))
stmt = stmt.where(ScalarMetricValue.value > 0)
total = count(db.session, stmt)
rows = db.session.execute(paginate(stmt, offset=0, limit=50)).scalars().all()
```
"""

from climate_ref.query.datasets import select_datasets
from climate_ref.query.diagnostics import (
    select_diagnostics,
    select_diagnostics_with_values,
    select_group_counts,
)
from climate_ref.query.executions import (
    filter_by_selectors,
    latest_execution_id,
    select_execution_group_ids,
    select_execution_groups,
    select_execution_outputs,
    select_latest_execution,
    select_latest_executions,
)
from climate_ref.query.filters import (
    DatasetFilter,
    DiagnosticFilter,
    ExecutionGroupFilter,
    InputDatasetFilter,
    MetricValueFilter,
    ResourceFilter,
)
from climate_ref.query.paging import count, paginate
from climate_ref.query.resources import GroupBy, select_execution_resources, select_resource_summary
from climate_ref.query.values import select_scalar_values, select_series_values, select_value_ids

__all__ = [
    "DatasetFilter",
    "DiagnosticFilter",
    "ExecutionGroupFilter",
    "GroupBy",
    "InputDatasetFilter",
    "MetricValueFilter",
    "ResourceFilter",
    "count",
    "filter_by_selectors",
    "latest_execution_id",
    "paginate",
    "select_datasets",
    "select_diagnostics",
    "select_diagnostics_with_values",
    "select_execution_group_ids",
    "select_execution_groups",
    "select_execution_outputs",
    "select_execution_resources",
    "select_group_counts",
    "select_latest_execution",
    "select_latest_executions",
    "select_resource_summary",
    "select_scalar_values",
    "select_series_values",
    "select_value_ids",
]
