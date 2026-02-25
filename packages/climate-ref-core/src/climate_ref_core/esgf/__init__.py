"""
ESGF dataset fetching

This module provides classes for searching and fetching datasets from ESGF
(Earth System Grid Federation) and other data registries.
"""

from climate_ref_core.esgf.base import ESGFRequest, IntakeESGFMixin
from climate_ref_core.esgf.cmip6 import CMIP6Request
from climate_ref_core.esgf.cmip7 import CMIP7Request
from climate_ref_core.esgf.fetcher import ESGFFetcher
from climate_ref_core.esgf.obs4mips import Obs4MIPsRequest
from climate_ref_core.esgf.registry import RegistryRequest

__all__ = [
    "CMIP6Request",
    "CMIP7Request",
    "ESGFFetcher",
    "ESGFRequest",
    "IntakeESGFMixin",
    "Obs4MIPsRequest",
    "RegistryRequest",
]
