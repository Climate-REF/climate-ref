"""
Tests for climate_ref_core.summary module.
"""

import pytest

from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement, Diagnostic
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.summary import (
    _extract_facet_values,
    _normalize_requirement_sets,
    collect_variables_by_experiment,
    format_diagnostic_markdown,
    format_provider_markdown,
    summarize_data_requirement,
    summarize_diagnostic,
    summarize_provider,
)


class SimpleDiagnostic(Diagnostic):
    """A simple diagnostic for testing with flat data_requirements."""

    name = "Simple Diagnostic"
    slug = "simple-diagnostic"
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": ("tas", "pr"), "experiment_id": "historical"}),),
            group_by=("source_id", "member_id"),
        ),
    )
    facets = ("source_id", "variable_id", "region")


class OrLogicDiagnostic(Diagnostic):
    """A diagnostic with OR-logic data requirements."""

    name = "OR Logic Diagnostic"
    slug = "or-logic-diagnostic"
    data_requirements = (
        # Option 1: CMIP6
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "tas", "table_id": "Amon"}),),
                group_by=("source_id", "variable_id"),
            ),
        ),
        # Option 2: CMIP7
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(FacetFilter(facets={"variable_id": "tas", "table_id": "Amon"}),),
                group_by=("source_id", "variable_id"),
            ),
        ),
    )
    facets = ("source_id", "variable_id")


class MultiFilterDiagnostic(Diagnostic):
    """A diagnostic with multiple FacetFilters (OR across filters)."""

    name = "Multi Filter"
    slug = "multi-filter"
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(facets={"variable_id": "tas", "experiment_id": "historical"}),
                FacetFilter(facets={"variable_id": "pr", "experiment_id": "ssp585"}),
            ),
            group_by=("source_id",),
        ),
    )
    facets = ("source_id",)


class NoFilterDiagnostic(Diagnostic):
    """A diagnostic with no filters."""

    name = "No Filter"
    slug = "no-filter"
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(),
            group_by=None,
        ),
    )
    facets = ()


class MultiSourceDiagnostic(Diagnostic):
    """A diagnostic requiring multiple source types."""

    name = "Multi Source"
    slug = "multi-source"
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("source_id",),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=None,
        ),
    )
    facets = ("source_id",)


class FrequencyDiagnostic(Diagnostic):
    """A diagnostic that filters by frequency."""

    name = "Frequency Test"
    slug = "frequency-test"
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas", "frequency": "mon"}),),
            group_by=("source_id",),
        ),
    )
    facets = ("source_id",)


@pytest.fixture
def simple_provider():
    provider = DiagnosticProvider("Test Provider", "1.0.0", slug="test-provider")
    provider.register(SimpleDiagnostic())
    return provider


@pytest.fixture
def or_logic_provider():
    provider = DiagnosticProvider("OR Provider", "2.0.0", slug="or-provider")
    provider.register(OrLogicDiagnostic())
    return provider


@pytest.fixture
def multi_provider():
    """Provider with multiple diagnostics for aggregation tests."""
    provider = DiagnosticProvider("Multi Provider", "3.0.0", slug="multi-provider")
    provider.register(SimpleDiagnostic())
    provider.register(MultiFilterDiagnostic())
    return provider


class TestExtractFacetValues:
    def test_single_filter_single_value(self):
        filters = (FacetFilter(facets={"variable_id": "tas"}),)
        result = _extract_facet_values(filters, "variable_id")
        assert result == ("tas",)

    def test_single_filter_multiple_values(self):
        filters = (FacetFilter(facets={"variable_id": ("tas", "pr")}),)
        result = _extract_facet_values(filters, "variable_id")
        assert result == ("pr", "tas")

    def test_multiple_filters_merge_values(self):
        filters = (
            FacetFilter(facets={"variable_id": "tas"}),
            FacetFilter(facets={"variable_id": "pr"}),
        )
        result = _extract_facet_values(filters, "variable_id")
        assert result == ("pr", "tas")

    def test_multiple_filters_deduplicate(self):
        filters = (
            FacetFilter(facets={"variable_id": "tas"}),
            FacetFilter(facets={"variable_id": ("tas", "pr")}),
        )
        result = _extract_facet_values(filters, "variable_id")
        assert result == ("pr", "tas")

    def test_missing_facet_returns_empty(self):
        filters = (FacetFilter(facets={"variable_id": "tas"}),)
        result = _extract_facet_values(filters, "experiment_id")
        assert result == ()

    def test_empty_filters_returns_empty(self):
        result = _extract_facet_values((), "variable_id")
        assert result == ()

    def test_results_are_sorted(self):
        filters = (FacetFilter(facets={"variable_id": ("zeta", "alpha", "mid")}),)
        result = _extract_facet_values(filters, "variable_id")
        assert result == ("alpha", "mid", "zeta")


class TestNormalizeRequirementSets:
    def test_flat_requirements(self):
        """Flat list of DataRequirement -> single set."""
        reqs = (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(),
                group_by=None,
            ),
        )
        result = _normalize_requirement_sets(reqs)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], DataRequirement)

    def test_nested_or_requirements(self):
        """Nested list of lists -> multiple sets (OR-logic)."""
        reqs = (
            (
                DataRequirement(
                    source_type=SourceDatasetType.CMIP6,
                    filters=(),
                    group_by=None,
                ),
            ),
            (
                DataRequirement(
                    source_type=SourceDatasetType.CMIP7,
                    filters=(),
                    group_by=None,
                ),
            ),
        )
        result = _normalize_requirement_sets(reqs)
        assert len(result) == 2
        assert result[0][0].source_type == SourceDatasetType.CMIP6
        assert result[1][0].source_type == SourceDatasetType.CMIP7

    def test_empty_requirements(self):
        result = _normalize_requirement_sets(())
        assert result == []

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected DataRequirement"):
            _normalize_requirement_sets((42,))


class TestSummarizeDataRequirement:
    def test_basic_requirement(self):
        req = DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": ("tas", "pr"), "experiment_id": "historical"}),),
            group_by=("source_id", "member_id"),
        )
        summary = summarize_data_requirement(req)

        assert summary.source_type == "cmip6"
        assert summary.variables == ("pr", "tas")
        assert summary.experiments == ("historical",)
        assert summary.tables == ()
        assert summary.frequencies == ()
        assert summary.group_by == ("source_id", "member_id")

    def test_no_filters(self):
        req = DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(),
            group_by=None,
        )
        summary = summarize_data_requirement(req)

        assert summary.source_type == "obs4mips"
        assert summary.variables == ()
        assert summary.experiments == ()
        assert summary.group_by is None

    def test_with_frequency(self):
        req = DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas", "frequency": "mon"}),),
            group_by=("source_id",),
        )
        summary = summarize_data_requirement(req)
        assert summary.frequencies == ("mon",)

    def test_with_table(self):
        req = DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas", "table_id": "Amon"}),),
            group_by=("source_id",),
        )
        summary = summarize_data_requirement(req)
        assert summary.tables == ("Amon",)


class TestSummarizeDiagnostic:
    def test_flat_diagnostic(self, simple_provider):
        diag = simple_provider.get("simple-diagnostic")
        summary = summarize_diagnostic(diag)

        assert summary.name == "Simple Diagnostic"
        assert summary.slug == "simple-diagnostic"
        assert summary.provider_name == "Test Provider"
        assert summary.provider_slug == "test-provider"
        assert summary.facets == ("source_id", "variable_id", "region")
        assert len(summary.requirement_sets) == 1
        assert len(summary.requirement_sets[0].requirements) == 1

        req = summary.requirement_sets[0].requirements[0]
        assert req.source_type == "cmip6"
        assert req.variables == ("pr", "tas")
        assert req.experiments == ("historical",)

    def test_or_logic_diagnostic(self, or_logic_provider):
        diag = or_logic_provider.get("or-logic-diagnostic")
        summary = summarize_diagnostic(diag)

        assert len(summary.requirement_sets) == 2
        assert summary.requirement_sets[0].requirements[0].source_type == "cmip6"
        assert summary.requirement_sets[1].requirements[0].source_type == "cmip7"

        # Both options should have the same variables
        for req_set in summary.requirement_sets:
            assert req_set.requirements[0].variables == ("tas",)
            assert req_set.requirements[0].tables == ("Amon",)


class TestSummarizeProvider:
    def test_basic_provider(self, simple_provider):
        summary = summarize_provider(simple_provider)

        assert summary.name == "Test Provider"
        assert summary.slug == "test-provider"
        assert summary.version == "1.0.0"
        assert len(summary.diagnostics) == 1
        assert summary.diagnostics[0].slug == "simple-diagnostic"

    def test_multi_diagnostic_provider(self, multi_provider):
        summary = summarize_provider(multi_provider)

        assert len(summary.diagnostics) == 2
        slugs = {d.slug for d in summary.diagnostics}
        assert slugs == {"simple-diagnostic", "multi-filter"}

    def test_empty_provider(self):
        provider = DiagnosticProvider("Empty", "0.0.0", slug="empty")
        summary = summarize_provider(provider)

        assert summary.name == "Empty"
        assert len(summary.diagnostics) == 0


class TestCollectVariablesByExperiment:
    def test_single_provider(self, simple_provider):
        result = collect_variables_by_experiment([simple_provider])

        assert "historical" in result
        assert "cmip6" in result["historical"]
        assert result["historical"]["cmip6"] == {"pr", "tas"}

    def test_multiple_providers(self, simple_provider, or_logic_provider):
        result = collect_variables_by_experiment([simple_provider, or_logic_provider])

        assert "historical" in result
        # OR logic diagnostic has no experiment filter, so appears under "*"
        assert "*" in result
        assert "cmip6" in result["*"]
        assert "tas" in result["*"]["cmip6"]
        assert "cmip7" in result["*"]
        assert "tas" in result["*"]["cmip7"]

    def test_no_experiment_filter_uses_wildcard(self):
        """Diagnostics without experiment_id filters use '*' as experiment key."""
        provider = DiagnosticProvider("NoExp", "1.0.0", slug="noexp")
        provider.register(NoFilterDiagnostic())
        result = collect_variables_by_experiment([provider])

        # No variables to extract (no filters at all), but wildcard entry exists
        assert "*" in result

    def test_multi_filter_diagnostic(self):
        """MultiFilterDiagnostic has two filters with different experiments.

        Since _extract_facet_values merges across all filters in a requirement,
        both variables appear under both experiments.
        """
        provider = DiagnosticProvider("MF", "1.0.0", slug="mf")
        provider.register(MultiFilterDiagnostic())
        result = collect_variables_by_experiment([provider])

        assert "historical" in result
        assert "ssp585" in result
        # Both variables are extracted from all filters in the requirement
        assert result["historical"]["cmip6"] == {"pr", "tas"}
        assert result["ssp585"]["cmip6"] == {"pr", "tas"}

    def test_empty_providers(self):
        result = collect_variables_by_experiment([])
        assert result == {}


class TestFormatDiagnosticMarkdown:
    def test_basic_format(self, simple_provider):
        diag = simple_provider.get("simple-diagnostic")
        summary = summarize_diagnostic(diag)
        md = format_diagnostic_markdown(summary)

        assert "### Simple Diagnostic" in md
        assert "`simple-diagnostic`" in md
        assert "`source_id`" in md
        assert "`cmip6`" in md
        assert "`tas`" in md
        assert "`pr`" in md
        assert "`historical`" in md

    def test_or_logic_format_has_options(self, or_logic_provider):
        diag = or_logic_provider.get("or-logic-diagnostic")
        summary = summarize_diagnostic(diag)
        md = format_diagnostic_markdown(summary)

        assert "#### Option 1" in md
        assert "#### Option 2" in md
        assert "`cmip6`" in md
        assert "`cmip7`" in md

    def test_no_options_header_for_single_set(self, simple_provider):
        diag = simple_provider.get("simple-diagnostic")
        summary = summarize_diagnostic(diag)
        md = format_diagnostic_markdown(summary)

        assert "#### Option" not in md


class TestFormatProviderMarkdown:
    def test_basic_format(self, simple_provider):
        summary = summarize_provider(simple_provider)
        md = format_provider_markdown(summary)

        assert "# Test Provider" in md
        assert "`test-provider`" in md
        assert "`1.0.0`" in md
        assert "**Diagnostics**: 1" in md
        assert "### Simple Diagnostic" in md

    def test_multi_diagnostic_format(self, multi_provider):
        summary = summarize_provider(multi_provider)
        md = format_provider_markdown(summary)

        assert "**Diagnostics**: 2" in md
        assert "### Simple Diagnostic" in md
        assert "### Multi Filter" in md
        assert "---" in md
