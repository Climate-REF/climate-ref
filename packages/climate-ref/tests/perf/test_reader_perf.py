"""
Performance of the `Reader` read paths against a production-shaped synthetic database.

Each scenario is one call the API or a notebook makes.
It fails when a query reads a large table in full,
which the query plan shows at any scale,
or when its median time exceeds the budget, which only means something near full scale.
"""

from collections.abc import Callable
from typing import Any

import attrs
import pytest

from climate_ref.models.metric_value import MetricValueType
from climate_ref.results import (
    DatasetFilter,
    ExecutionGroupFilter,
    MetricValueFilter,
    OutlierPolicy,
    Reader,
    ResourceFilter,
)
from climate_ref_core.source_types import SourceDatasetType

_EVERY_EXECUTION = frozenset({"execution", "execution_group"})


@attrs.frozen(kw_only=True)
class Scenario:
    id: str
    call: Callable[[Reader, Any], object]
    budget_seconds: float = 0.5
    allowed_scans: frozenset[str] = frozenset()
    """Large tables this call has to read in full, e.g. a summary over every execution."""


def _diagnostic(t, **kwargs):
    return MetricValueFilter(
        diagnostic_slug=t.diagnostic_slug, provider_slug=t.provider_slug, include_retracted=True, **kwargs
    )


SCENARIOS = [
    # values: the API's diagnostic-scoped and execution-scoped value pages
    Scenario(
        id="values.scalar.diagnostic-page",
        call=lambda r, t: r.values.scalar_values(_diagnostic(t), limit=100),
    ),
    Scenario(
        id="values.scalar.diagnostic-page-no-facets",
        call=lambda r, t: r.values.scalar_values(_diagnostic(t), limit=100, with_facets=False),
    ),
    Scenario(
        id="values.scalar.diagnostic-page-iqr",
        call=lambda r, t: r.values.scalar_values(
            _diagnostic(t), outliers=OutlierPolicy(method="iqr"), limit=100
        ),
        budget_seconds=5.0,
    ),
    Scenario(
        id="values.scalar.diagnostic-page-iqr-unverified",
        call=lambda r, t: r.values.scalar_values(
            _diagnostic(t), outliers=OutlierPolicy(method="iqr"), include_unverified=True, limit=100
        ),
        budget_seconds=5.0,
    ),
    Scenario(
        id="values.scalar.diagnostic-deep-page-iqr",
        call=lambda r, t: r.values.scalar_values(
            _diagnostic(t), outliers=OutlierPolicy(method="iqr"), offset=10_000, limit=100
        ),
        budget_seconds=5.0,
    ),
    Scenario(
        id="values.scalar.provider-page-iqr",
        call=lambda r, t: r.values.scalar_values(
            MetricValueFilter(provider_slug=t.provider_slug), outliers=OutlierPolicy(method="iqr"), limit=100
        ),
        budget_seconds=10.0,
    ),
    Scenario(
        id="values.scalar.diagnostic-dimension-filter",
        call=lambda r, t: r.values.scalar_values(
            _diagnostic(t, dimensions={"source_id": t.source_id}), limit=100
        ),
    ),
    Scenario(
        id="values.scalar.execution-page",
        call=lambda r, t: r.values.scalar_values(
            MetricValueFilter(execution_ids=[t.execution_id], include_retracted=True), limit=100
        ),
    ),
    Scenario(
        id="values.scalar.group-latest-only",
        call=lambda r, t: r.values.scalar_values(
            MetricValueFilter(execution_group_ids=[t.execution_group_id], latest_only=True), limit=100
        ),
    ),
    Scenario(
        id="values.scalar.all-page",
        call=lambda r, t: r.values.scalar_values(limit=100, with_facets=False),
    ),
    # notebooks: a whole diagnostic as a DataFrame
    Scenario(
        id="values.scalar.diagnostic-frame",
        call=lambda r, t: r.values.scalar_values(_diagnostic(t), with_facets=False).to_pandas(),
        budget_seconds=10.0,
    ),
    Scenario(
        id="values.series.diagnostic-frame",
        call=lambda r, t: r.values.series_values(_diagnostic(t), with_facets=False).to_pandas(explode=False),
        budget_seconds=30.0,
    ),
    Scenario(
        id="values.series.diagnostic-page",
        call=lambda r, t: r.values.series_values(_diagnostic(t), limit=100),
    ),
    Scenario(
        id="values.series.diagnostic-reference-page",
        call=lambda r, t: r.values.series_values(_diagnostic(t, reference_only=True), limit=100),
    ),
    Scenario(
        id="values.series.execution-page",
        call=lambda r, t: r.values.series_values(
            MetricValueFilter(execution_ids=[t.execution_id], include_retracted=True), limit=100
        ),
    ),
    # executions
    Scenario(
        id="executions.groups.diagnostic",
        call=lambda r, t: r.executions.groups(ExecutionGroupFilter(diagnostic_contains=[t.diagnostic_slug])),
    ),
    Scenario(id="executions.groups.page", call=lambda r, t: r.executions.groups(limit=50)),
    Scenario(id="executions.group", call=lambda r, t: r.executions.group(t.execution_group_id)),
    Scenario(
        id="executions.latest-execution",
        call=lambda r, t: r.executions.latest_execution(t.execution_group_id),
    ),
    Scenario(id="executions.outputs", call=lambda r, t: r.executions.outputs(t.execution_id)),
    # diagnostics
    Scenario(
        id="diagnostics.list",
        call=lambda r, t: r.diagnostics.list(),
        allowed_scans=_EVERY_EXECUTION,
    ),
    Scenario(
        id="diagnostics.with-scalar-values",
        call=lambda r, t: r.diagnostics.with_values(MetricValueType.SCALAR),
    ),
    Scenario(
        id="diagnostics.with-series-values",
        call=lambda r, t: r.diagnostics.with_values(MetricValueType.SERIES),
    ),
    # datasets
    Scenario(
        id="datasets.cmip6-page",
        call=lambda r, t: r.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6), limit=50),
    ),
    Scenario(
        id="datasets.execution",
        call=lambda r, t: r.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, execution_ids=t.execution_id),
            include_files=True,
        ),
    ),
    Scenario(id="datasets.get", call=lambda r, t: r.datasets.get(t.dataset_slug)),
    # resources
    Scenario(id="resources.measurements-page", call=lambda r, t: r.resources.measurements(limit=100)),
    Scenario(
        id="resources.summary.diagnostic",
        call=lambda r, t: r.resources.summary(ResourceFilter(diagnostic_slug=t.diagnostic_slug)),
    ),
    Scenario(
        id="resources.summaries.by-diagnostic",
        call=lambda r, t: r.resources.summaries(group_by="diagnostic"),
        allowed_scans=_EVERY_EXECUTION,
    ),
    Scenario(
        id="resources.profiles",
        call=lambda r, t: r.resources.profiles(),
        allowed_scans=_EVERY_EXECUTION,
    ),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_reader_scenario(scenario: Scenario, measure, targets) -> None:
    measurement = measure(scenario.id, lambda reader: scenario.call(reader, targets))

    unexpected = measurement.large_table_scans - scenario.allowed_scans
    assert not unexpected, f"reads every row of {sorted(unexpected)}"
    assert measurement.median_seconds <= scenario.budget_seconds
