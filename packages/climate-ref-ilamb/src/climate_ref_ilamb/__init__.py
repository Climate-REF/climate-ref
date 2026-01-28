"""
Diagnostic provider for ILAMB

This module provides a diagnostics provider for ILAMB, a tool for evaluating
climate models against observations.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
from typing import TYPE_CHECKING

import yaml

from climate_ref_core.dataset_registry import DATASET_URL, dataset_registry_manager, fetch_all_files
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_ilamb.standard import ILAMBStandard

if TYPE_CHECKING:
    from climate_ref.config import Config

__version__ = importlib.metadata.version("climate-ref-ilamb")


class ILAMBProvider(DiagnosticProvider):
    """Provider for ILAMB diagnostics."""

    def fetch_data(self, config: Config) -> None:
        """Fetch ILAMB reference data from all registries."""
        for name in ("ilamb-test", "ilamb", "iomb"):
            registry = dataset_registry_manager[name]
            fetch_all_files(registry, name, output_dir=None)


provider = ILAMBProvider("ILAMB", __version__)

# Register some datasets
dataset_registry_manager.register(
    "ilamb-test",
    base_url=DATASET_URL,
    package="climate_ref_ilamb.dataset_registry",
    resource="test.txt",
)
dataset_registry_manager.register(
    "ilamb",
    base_url=DATASET_URL,
    package="climate_ref_ilamb.dataset_registry",
    resource="ilamb.txt",
)
dataset_registry_manager.register(
    "iomb",
    base_url=DATASET_URL,
    package="climate_ref_ilamb.dataset_registry",
    resource="iomb.txt",
)

# Dynamically register ILAMB diagnostics
for yaml_file in importlib.resources.files("climate_ref_ilamb.configure").iterdir():
    with open(str(yaml_file)) as fin:
        metrics = yaml.safe_load(fin)
    registry_filename = metrics.pop("registry")
    for metric, options in metrics.items():
        provider.register(ILAMBStandard(registry_filename, metric, options.pop("sources"), **options))
