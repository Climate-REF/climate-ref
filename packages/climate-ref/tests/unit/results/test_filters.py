import datetime

import pytest

from climate_ref.models.execution import ExecutionFailureReason
from climate_ref.results.filters import (
    BaseFilter,
    DiagnosticFilter,
    ExecutionGroupFilter,
    ExecutionScope,
    MetricValueFilter,
    ResourceFilter,
    VersionScope,
)
from climate_ref_core.source_types import SourceDatasetType

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
    assert len(scope.execution_clauses()) == 2
    assert ExecutionScope().execution_clauses() == []


@pytest.mark.parametrize("cls", READER_FILTERS)
def test_or_no_model_input_without_source_types_raises(cls):
    with pytest.raises(ValueError, match="or_no_model_input"):
        cls(or_no_model_input=True)


@pytest.mark.parametrize("cls", READER_FILTERS)
def test_execution_ids_on_every_reader_filter(cls):
    f = cls(execution_ids=[1, 2], execution_group_ids=[3])
    assert f.execution_ids == (1, 2)
    assert f.execution_group_ids == (3,)
    assert len(f.execution_clauses()) == 2


@pytest.mark.parametrize("cls", READER_FILTERS)
def test_one_value_or_a_list(cls):
    one = cls(
        diagnostic_slug="enso_tel",
        provider_slug="pmp",
        diagnostic_contains="enso",
        execution_ids=1,
        source_types="cmip6",
    )
    many = cls(
        diagnostic_slug=["enso_tel"],
        provider_slug=("pmp",),
        diagnostic_contains=["enso"],
        execution_ids=[1],
        source_types=[SourceDatasetType.CMIP6],
    )
    assert one == many
    assert one.diagnostic_slug == ("enso_tel",)
    assert one.source_types == (SourceDatasetType.CMIP6,)


def test_source_types_and_failure_reasons_accept_names():
    assert ExecutionGroupFilter(source_types="CMIP6").source_types == (SourceDatasetType.CMIP6,)
    f = ExecutionGroupFilter(failure_reasons=["DiagnosticError", "system_error"])
    assert f.failure_reasons == (ExecutionFailureReason.DiagnosticError, ExecutionFailureReason.SystemError)


def test_unknown_source_type_rejected():
    with pytest.raises(ValueError, match="'cmip9' is not a SourceDatasetType"):
        DiagnosticFilter(source_types="cmip9")


def test_string_ids_rejected():
    with pytest.raises(TypeError, match="not '12'"):
        MetricValueFilter(execution_ids="12")


def test_value_ids_are_tuples():
    f = MetricValueFilter(isolate_ids=[1, 2], exclude_ids=3)
    assert (f.isolate_ids, f.exclude_ids) == ((1, 2), (3,))
    assert f == MetricValueFilter(isolate_ids=(1, 2), exclude_ids=[3])


@pytest.mark.parametrize("cls", READER_FILTERS)
def test_aware_times_become_naive_utc(cls):
    nzdt = datetime.timezone(datetime.timedelta(hours=13))
    f = cls(
        created_after=datetime.datetime(2026, 1, 1, 12, tzinfo=nzdt),
        created_before=datetime.datetime(2026, 1, 2),
    )
    assert f.created_after == datetime.datetime(2025, 12, 31, 23)
    assert f.created_before == datetime.datetime(2026, 1, 2)


@pytest.mark.parametrize(
    ("key", "match"),
    [("nope", "Unknown selector 'nope'"), ("cmip9.source_id", "Unknown dataset type 'cmip9'")],
)
def test_unknown_selector_key_raises(key, match):
    with pytest.raises(ValueError, match=match):
        ExecutionGroupFilter(selectors={key: "x"})
