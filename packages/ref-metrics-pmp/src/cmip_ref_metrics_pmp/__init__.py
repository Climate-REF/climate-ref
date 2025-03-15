"""
Rapid evaluating CMIP data
"""

import importlib.metadata

from cmip_ref_core.providers import CondaMetricsProvider
from cmip_ref_metrics_pmp.variability_modes import ExtratropicalModesOfVariability

__version__ = importlib.metadata.version("cmip_ref_metrics_pmp")

# Initialise the metrics manager and register the example metric
provider = CondaMetricsProvider("PMP", __version__)
provider.register(ExtratropicalModesOfVariability("PDO"))
provider.register(ExtratropicalModesOfVariability("NPGO"))
provider.register(ExtratropicalModesOfVariability("NAO"))
provider.register(ExtratropicalModesOfVariability("NAM"))
provider.register(ExtratropicalModesOfVariability("PNA"))
provider.register(ExtratropicalModesOfVariability("NPO"))
provider.register(ExtratropicalModesOfVariability("SAM"))
