"""
Guard that every in-repo obs4MIPs requirement declares obs4REF as a fallback.

The solver no longer folds obs4REF into obs4MIPs on its own,
so a requirement that forgets to declare it silently loses its reference data.
"""

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_example import provider as example_provider
from climate_ref_ilamb import provider as ilamb_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref_core.source_types import SourceDatasetType
from climate_ref_core.summary import normalize_requirement_sets

PROVIDERS = [example_provider, pmp_provider, esmvaltool_provider, ilamb_provider]


@pytest.mark.parametrize(
    "diagnostic",
    [
        pytest.param(diagnostic, id=f"{provider.slug}/{diagnostic.slug}")
        for provider in PROVIDERS
        for diagnostic in provider.diagnostics()
    ],
)
def test_obs4mips_requirements_declare_obs4ref(diagnostic):
    for requirements in normalize_requirement_sets(diagnostic.data_requirements):
        for requirement in requirements:
            if requirement.source_type is SourceDatasetType.obs4MIPs:
                assert SourceDatasetType.obs4REF in requirement.fallback_source_types
