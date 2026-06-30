"""
Rapid evaluating CMIP data
"""

import importlib.metadata

from climate_ref_core.providers import DiagnosticProvider
from climate_ref_example.example import GlobalMeanTimeseries
from climate_ref_example.surface_temperature import GlobalMeanSurfaceTemperatureBias

__version__ = importlib.metadata.version("climate-ref-example")

# Initialise the diagnostics manager and register the example diagnostics
provider = DiagnosticProvider("Example", __version__)
provider.register(GlobalMeanTimeseries())
provider.register(GlobalMeanSurfaceTemperatureBias())
