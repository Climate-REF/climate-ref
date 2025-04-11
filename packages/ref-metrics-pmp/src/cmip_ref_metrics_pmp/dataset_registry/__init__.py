"""
Data registries for non-published reference data

"""

from cmip_ref_core.dataset_registry import build_reference_data_registry

ANNUAL_CYCLE_REGISTRY = build_reference_data_registry(
    registry_package="cmip_ref_metrics_pmp.dataset_registry",
    registry_resource="pmp_annual_cycle.txt",
)
