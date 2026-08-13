"""
Tests for working out what reference data is required and where it comes from.
"""

import pytest

from climate_ref_core.dataset_registry import DatasetRegistryManager, RegistryUseCase
from climate_ref_core.datasets import FacetFilter
from climate_ref_core.diagnostics import DataRequirement, Diagnostic
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.reference_data import (
    ESGF_OBS4MIPS,
    collect_required_reference_data,
    format_reference_data_markdown,
    source_ids_by_registry,
)
from climate_ref_core.source_types import SourceDatasetType


class _FakePooch:
    def __init__(self, keys):
        self.registry = {key: "hash" for key in keys}


class _FakeEntry:
    def __init__(self, keys, source_type, use_case=RegistryUseCase.reference):
        self.registry = _FakePooch(keys)
        self.source_type = source_type
        self.use_case = use_case


class _FakeManager(DatasetRegistryManager):
    def __init__(self, entries):
        self._entries = entries

    def keys(self):
        return list(self._entries)

    def entry(self, item):
        return self._entries[item]


def _obs4ref_key(source_id, variable_id, version="v20250101"):
    return (
        f"obs4REF/INST/{source_id}/mon/{variable_id}/gn/{version}/"
        f"{variable_id}_mon_{source_id}_INST_gn_198001-201212.nc"
    )


def _provider(requirements, slug="test_provider"):
    class _Diagnostic(Diagnostic):
        name = "test-diagnostic"
        slug = "test-diagnostic"
        facets = ()
        data_requirements = tuple(requirements)

        def run(self, definition, *, capture_regression=False):  # pragma: no cover - never run
            raise NotImplementedError

    provider = DiagnosticProvider(slug, "v0.1.0")
    provider.register(_Diagnostic())  # type: ignore[arg-type]
    return provider


def _requirement(source_type, source_id, variable_id):
    return DataRequirement(
        source_type=source_type,
        filters=(FacetFilter(facets={"source_id": source_id, "variable_id": variable_id}),),
        group_by=None,
    )


class TestSourceIdsByRegistry:
    def test_keys_are_read_per_source_type(self):
        manager = _FakeManager(
            {"obs4ref": _FakeEntry([_obs4ref_key("WECANN-1-0", "gpp")], SourceDatasetType.obs4REF)}
        )

        found = source_ids_by_registry(manager)

        # An obs4REF registry answers for obs4MIPs requirements too, since that is the
        # source type its data is ingested under.
        assert found[(SourceDatasetType.obs4REF.value, "WECANN-1-0")] == ["obs4ref"]
        assert found[(SourceDatasetType.obs4MIPs.value, "WECANN-1-0")] == ["obs4ref"]

    def test_support_registries_are_ignored(self):
        manager = _FakeManager(
            {
                "support": _FakeEntry(
                    [_obs4ref_key("X-1-0", "pr")],
                    SourceDatasetType.obs4REF,
                    use_case=RegistryUseCase.support,
                )
            }
        )

        assert source_ids_by_registry(manager) == {}

    def test_overlapping_registries_are_both_recorded(self):
        manager = _FakeManager(
            {
                "obs4ref": _FakeEntry([_obs4ref_key("HadISST-1-1", "ts")], SourceDatasetType.obs4REF),
                "quickstart": _FakeEntry([_obs4ref_key("HadISST-1-1", "ts")], SourceDatasetType.obs4REF),
            }
        )

        found = source_ids_by_registry(manager)

        assert found[(SourceDatasetType.obs4MIPs.value, "HadISST-1-1")] == ["obs4ref", "quickstart"]


class TestCollectRequiredReferenceData:
    def test_dataset_in_a_registry_is_attributed_to_it(self):
        manager = _FakeManager(
            {"obs4ref": _FakeEntry([_obs4ref_key("WECANN-1-0", "gpp")], SourceDatasetType.obs4REF)}
        )
        provider = _provider([_requirement(SourceDatasetType.obs4MIPs, "WECANN-1-0", "gpp")])

        (dataset,) = collect_required_reference_data([provider], manager)

        assert dataset.source_id == "WECANN-1-0"
        assert dataset.supplier == "obs4ref"
        assert dataset.registry_name == "obs4ref"
        assert dataset.is_from_registry

    def test_dataset_in_no_registry_comes_from_esgf(self):
        manager = _FakeManager({})
        provider = _provider([_requirement(SourceDatasetType.obs4MIPs, "ERA-5", "ta")])

        (dataset,) = collect_required_reference_data([provider], manager)

        assert dataset.supplier == ESGF_OBS4MIPS
        assert dataset.registry_name is None
        assert not dataset.is_from_registry

    def test_a_registry_only_answers_for_its_own_source_type(self):
        # The PMP registry carries an ERA-5 climatology, which does not satisfy a
        # requirement for obs4MIPs ERA-5.
        manager = _FakeManager(
            {
                "pmp-climatology": _FakeEntry(
                    [
                        "PMP_obs4MIPsClims/ta/gr/v20250225/"
                        "ta_mon_ERA-5_PCMDI_gr_198101-200412_AC_v20250225_2.5x2.5.nc"
                    ],
                    SourceDatasetType.PMPClimatology,
                )
            }
        )
        provider = _provider(
            [
                _requirement(SourceDatasetType.obs4MIPs, "ERA-5", "ta"),
                _requirement(SourceDatasetType.PMPClimatology, "ERA-5", "ta"),
            ]
        )

        datasets = {d.source_type: d for d in collect_required_reference_data([provider], manager)}

        assert datasets[SourceDatasetType.obs4MIPs.value].supplier == ESGF_OBS4MIPS
        assert datasets[SourceDatasetType.PMPClimatology.value].supplier == "pmp-climatology"

    def test_model_data_requirements_are_ignored(self):
        provider = _provider([_requirement(SourceDatasetType.CMIP6, "ACCESS-ESM1-5", "tas")])

        assert collect_required_reference_data([provider], _FakeManager({})) == []

    def test_variables_are_unioned_across_diagnostics(self):
        manager = _FakeManager({})
        one = _provider([_requirement(SourceDatasetType.obs4MIPs, "ERA-5", "ta")], slug="one")
        two = _provider([_requirement(SourceDatasetType.obs4MIPs, "ERA-5", "psl")], slug="two")

        (dataset,) = collect_required_reference_data([one, two], manager)

        assert dataset.variable_ids == ("psl", "ta")
        assert len(dataset.diagnostics) == 2


class TestFormatMarkdown:
    def test_page_groups_by_supplier_and_shows_how_to_fetch(self):
        manager = _FakeManager(
            {"obs4ref": _FakeEntry([_obs4ref_key("WECANN-1-0", "gpp")], SourceDatasetType.obs4REF)}
        )
        provider = _provider(
            [
                _requirement(SourceDatasetType.obs4MIPs, "WECANN-1-0", "gpp"),
                _requirement(SourceDatasetType.obs4MIPs, "ERA-5", "ta"),
            ]
        )

        markdown = format_reference_data_markdown(collect_required_reference_data([provider], manager))

        assert "## obs4ref" in markdown
        assert f"## {ESGF_OBS4MIPS}" in markdown
        assert "ref datasets fetch-data --registry obs4ref" in markdown
        assert "`WECANN-1-0`" in markdown
        assert "`ERA-5`" in markdown

    @pytest.mark.parametrize("datasets", [[], ()])
    def test_empty_input_still_renders(self, datasets):
        assert format_reference_data_markdown(datasets).startswith("# Reference data")
