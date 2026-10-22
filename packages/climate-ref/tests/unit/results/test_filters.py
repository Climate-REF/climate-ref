import datetime

import pytest

from climate_ref.results.filters import (
    BaseFilter,
    DiagnosticFilter,
    ExecutionGroupFilter,
    ExecutionScope,
    MetricValueFilter,
    ResourceFilter,
    VersionScope,
)

READER_FILTERS = [MetricValueFilter, ResourceFilter, ExecutionGroupFilter, DiagnosticFilter]


@pytest.mark.parametrize("cls", READER_FILTERS)
def test_every_reader_filter_is_a_base_filter(cls):
    assert issubclass(cls, BaseFilter)


@pytest.mark.parametrize("cls", [c for c in READER_FILTERS if issubclass(c, VersionScope)])
def test_promoted_version_only_by_default(cls):
    assert cls().include_superseded is False


def test_execution_scope_window():
    scope = ExecutionScope(
        created_after=datetime.datetime(2026, 1, 1), created_before=datetime.datetime(2027, 1, 1)
    )
    assert len(scope.created_clauses()) == 2
    assert ExecutionScope().created_clauses() == []
