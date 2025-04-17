"""
Rapid evaluating CMIP data
"""

import importlib.metadata

from cmip_ref_core.dataset_registry import dataset_registry_manager
from cmip_ref_core.providers import CondaMetricsProvider
from cmip_ref_metrics_pmp.annual_cycle import AnnualCycle
from cmip_ref_metrics_pmp.variability_modes import ExtratropicalModesOfVariability

__version__ = importlib.metadata.version("cmip_ref_metrics_pmp")

# Initialise the metrics manager and register the example metric
provider = CondaMetricsProvider("PMP", __version__)

# Extratropical modes of variability
provider.register(ExtratropicalModesOfVariability("PDO"))
provider.register(ExtratropicalModesOfVariability("NPGO"))
provider.register(ExtratropicalModesOfVariability("NAO"))
provider.register(ExtratropicalModesOfVariability("NAM"))
provider.register(ExtratropicalModesOfVariability("PNA"))
provider.register(ExtratropicalModesOfVariability("NPO"))
provider.register(ExtratropicalModesOfVariability("SAM"))
provider.register(AnnualCycle())



dataset_registry_manager.register(
    "pmp-annual-cycle",
    "https://pub-b093171261094c4ea9adffa01f94ee06.r2.dev/",
    package="cmip_ref_metrics_pmp.dataset_registry",
    resource="pmp_annual_cycle.txt",
)
