"""
Rapid evaluating CMIP data
"""

import importlib.metadata

from cmip_ref_core.providers import MetricsProvider
from cmip_ref_metrics_pmp.example import ExtratropicalModesOfVariability_PDO

__version__ = importlib.metadata.version("cmip_ref_metrics_pmp")

# Initialise the metrics manager and register the example metric
provider = MetricsProvider("PMP", __version__)
provider.register(ExtratropicalModesOfVariability_PDO())
