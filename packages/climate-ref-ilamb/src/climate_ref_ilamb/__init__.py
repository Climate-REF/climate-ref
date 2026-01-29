"""
Diagnostic provider for ILAMB

This module provides a diagnostics provider for ILAMB, a tool for evaluating
climate models against observations.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING

import pooch
import yaml
from loguru import logger

from climate_ref_core.dataset_registry import (
    DATASET_URL,
    dataset_registry_manager,
    fetch_all_files,
    validate_registry_cache,
)
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_ilamb.standard import ILAMBStandard

if TYPE_CHECKING:
    from climate_ref.config import Config

__version__ = importlib.metadata.version("climate-ref-ilamb")

# Registry names used by ILAMB
_REGISTRY_NAMES = ("ilamb-test", "ilamb", "iomb")


class ILAMBProvider(DiagnosticProvider):
    """Provider for ILAMB diagnostics."""

    def fetch_data(self, config: Config) -> None:
        """Fetch ILAMB reference data from all registries."""
        for name in _REGISTRY_NAMES:
            registry = dataset_registry_manager[name]
            fetch_all_files(registry, name, output_dir=None)

    def validate_setup(self, config: Config) -> bool:
        """Validate that all ILAMB data is cached with correct checksums."""
        all_errors: list[str] = []
        for name in _REGISTRY_NAMES:
            registry = dataset_registry_manager[name]
            errors = validate_registry_cache(registry, name)
            all_errors.extend(errors)

        if all_errors:
            for error in all_errors:
                logger.error(f"ILAMB validation failed: {error}")
            logger.error(
                f"Data for {self.slug} is missing or corrupted. "
                f"Please run `ref providers setup --provider {self.slug}` to fetch data."
            )
            return False
        return True

    def get_data_path(self) -> Path | None:
        """Get the path where ILAMB data is cached."""
        # All ILAMB registries use the same cache
        return Path(pooch.os_cache("climate_ref"))


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
