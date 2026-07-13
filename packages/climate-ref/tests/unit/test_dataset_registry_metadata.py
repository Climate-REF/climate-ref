"""
Tests for the registry source-type/use-case annotations.

Importing ``climate_ref``, ``climate_ref_pmp``, ``climate_ref_esmvaltool`` and
``climate_ref_ilamb`` runs the module-level ``dataset_registry_manager.register(...)``
calls that these tests exercise,
so they are imported explicitly here rather than relying on import order from other tests.
"""

import climate_ref_esmvaltool  # noqa: F401
import climate_ref_ilamb  # noqa: F401
import climate_ref_pmp  # noqa: F401
import pytest
from climate_ref_esmvaltool.diagnostics.base import _DATASETS_REGISTRY_NAME
from climate_ref_esmvaltool.recipe import _RECIPES_REGISTRY_NAME

import climate_ref  # noqa: F401
from climate_ref_core.dataset_registry import (
    RegistryUseCase,
    dataset_registry_manager,
    iter_reference_registries,
)
from climate_ref_core.source_types import SourceDatasetType


class TestRegistryAnnotationsRoundTrip:
    """Each ``register()`` call site's annotations round-trip via ``.entry(name)``."""

    def test_obs4ref(self):
        entry = dataset_registry_manager.entry("obs4ref")
        assert entry.source_type is SourceDatasetType.obs4REF
        assert entry.use_case is RegistryUseCase.reference

    def test_quickstart(self):
        entry = dataset_registry_manager.entry("quickstart")
        assert entry.source_type is SourceDatasetType.obs4REF
        assert entry.use_case is RegistryUseCase.reference

    def test_sample_data(self):
        # Genuinely multi-type (CMIP6, obs4MIPs and obs4REF): left unannotated.
        entry = dataset_registry_manager.entry("sample-data")
        assert entry.source_type is None
        assert entry.use_case is RegistryUseCase.support

    def test_pmp_climatology(self):
        entry = dataset_registry_manager.entry("pmp-climatology")
        assert entry.source_type is SourceDatasetType.PMPClimatology
        assert entry.use_case is RegistryUseCase.reference

    def test_esmvaltool_datasets(self):
        # The OBS/OBS6/native6/obs4MIPs data is ingestable through the ESMValCore-DRS adapter,
        # so the registry is annotated as ESMValTool reference data.
        entry = dataset_registry_manager.entry(_DATASETS_REGISTRY_NAME)
        assert entry.source_type is SourceDatasetType.ESMValToolReference
        assert entry.use_case is RegistryUseCase.reference

    def test_esmvaltool_recipes(self):
        entry = dataset_registry_manager.entry(_RECIPES_REGISTRY_NAME)
        assert entry.source_type is None
        assert entry.use_case is RegistryUseCase.support

    def test_ilamb(self):
        # A plain fetch registry: no source type, and the default ``support`` use case, so it is
        # not a reference source type. It holds the few reference obs ILAMB still needs but that
        # are not yet in obs4MIPs/obs4REF. The provider fetches these at execute time rather than
        # ingesting them as a REF source type.
        entry = dataset_registry_manager.entry("ilamb")
        assert entry.source_type is None
        assert entry.use_case is RegistryUseCase.support
        assert set(entry.registry.registry.keys()) == {
            "ilamb/mrsol/WangMao/mrsol_olc.nc",
            "ilamb/tas/CRU4.02/tas.nc",
            "ilamb/evspsbl/GLEAMv3.3a/et.nc",
        }

    def test_ilamb_regions(self):
        entry = dataset_registry_manager.entry("ilamb-regions")
        assert entry.source_type is None
        assert entry.use_case is RegistryUseCase.support
        assert set(entry.registry.registry.keys()) == {
            "ilamb/regions/GlobalLand.nc",
            "ilamb/regions/Koppen_coarse.nc",
        }

    def test_ilamb_test(self):
        entry = dataset_registry_manager.entry("ilamb-test")
        assert entry.source_type is None
        assert entry.use_case is RegistryUseCase.support

    def test_iomb_registry_removed(self):
        with pytest.raises(KeyError):
            dataset_registry_manager["iomb"]


class TestIterReferenceRegistries:
    """``iter_reference_registries`` yields exactly the annotated ``reference`` registries."""

    def test_yields_reference_registries(self):
        results = list(iter_reference_registries(dataset_registry_manager))
        by_id = {id(registry): source_type for registry, source_type in results}

        assert by_id[id(dataset_registry_manager["obs4ref"])] is SourceDatasetType.obs4REF
        assert by_id[id(dataset_registry_manager["quickstart"])] is SourceDatasetType.obs4REF
        assert by_id[id(dataset_registry_manager["pmp-climatology"])] is SourceDatasetType.PMPClimatology
        assert (
            by_id[id(dataset_registry_manager[_DATASETS_REGISTRY_NAME])]
            is SourceDatasetType.ESMValToolReference
        )

    def test_does_not_yield_sample_data_or_recipes_registries(self):
        results = list(iter_reference_registries(dataset_registry_manager))
        yielded_registries = {id(registry) for registry, _ in results}

        assert id(dataset_registry_manager["sample-data"]) not in yielded_registries
        assert id(dataset_registry_manager[_RECIPES_REGISTRY_NAME]) not in yielded_registries

    def test_does_not_yield_ilamb_registries(self):
        results = list(iter_reference_registries(dataset_registry_manager))
        yielded_registries = {id(registry) for registry, _ in results}

        # "ilamb" is now a plain fetch registry, not a reference source type.
        assert id(dataset_registry_manager["ilamb"]) not in yielded_registries
        assert id(dataset_registry_manager["ilamb-regions"]) not in yielded_registries
        assert id(dataset_registry_manager["ilamb-test"]) not in yielded_registries
