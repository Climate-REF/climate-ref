"""
Diagnostic provider for ILAMB

This module provides a diagnostics provider for ILAMB, a tool for evaluating
climate models against observations.
"""

import importlib.metadata
import importlib.resources

import yaml

from climate_ref_core.dataset_registry import dataset_registry_manager
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_ilamb.datasets import ILAMB_DATA_VERSION
from climate_ref_ilamb.standard import ILAMBStandard

__version__ = importlib.metadata.version("climate-ref-ilamb")

provider = DiagnosticProvider("ILAMB", __version__)

# Register some datasets
dataset_registry_manager.register(
    "ilamb-test",
    base_url="https://www.ilamb.org/ILAMB-Data/DATA",
    package="climate_ref_ilamb.dataset_registry",
    resource="test.txt",
    cache_name="ilamb3",
    version=ILAMB_DATA_VERSION,
)
dataset_registry_manager.register(
    "ilamb",
    base_url="https://www.ilamb.org/ILAMB-Data/DATA",
    package="climate_ref_ilamb.dataset_registry",
    resource="ilamb.txt",
    cache_name="ilamb3",
    version=ILAMB_DATA_VERSION,
)
dataset_registry_manager.register(
    "iomb",
    base_url="https://www.ilamb.org/ilamb3-data/",
    package="climate_ref_ilamb.dataset_registry",
    resource="iomb.txt",
    cache_name="ilamb3",
    version=ILAMB_DATA_VERSION,
)

# Dynamically register ILAMB diagnostics
for yaml_file in importlib.resources.files("climate_ref_ilamb.configure").iterdir():
    with open(str(yaml_file)) as fin:
        metrics = yaml.safe_load(fin)
    registry_filename = metrics.pop("registry")
    for metric, options in metrics.items():
        provider.register(ILAMBStandard(registry_filename, metric, options.pop("sources"), **options))
