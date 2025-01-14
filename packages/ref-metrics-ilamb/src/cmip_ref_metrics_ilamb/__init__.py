"""
Rapid evaluating CMIP data with ILAMB.
"""

import importlib.metadata

from cmip_ref_core.providers import MetricsProvider
from cmip_ref_metrics_ilamb.example import ILAMBStandardTAS

__version__ = importlib.metadata.version("cmip_ref_metrics_ilamb")

# Initialise the metrics manager and register the example metric
provider = MetricsProvider("ILAMB", __version__)

# Could I provide reference data and variable info in the Metric constructor and
# loop here registering all metrics I need to do for ILAMB/IOMB?
provider.register(ILAMBStandardTAS())
