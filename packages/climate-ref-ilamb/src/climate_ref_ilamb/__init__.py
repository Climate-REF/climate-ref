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

import yaml
from loguru import logger

from climate_ref_core.data import resolve_cache_dir
from climate_ref_core.dataset_registry import (
    DATASET_URL,
    RegistryUseCase,
    dataset_registry_manager,
    fetch_all_files,
    validate_registry_cache,
)
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_ilamb.standard import ILAMBStandard

if TYPE_CHECKING:
    from climate_ref.config import Config

__version__ = importlib.metadata.version("climate-ref-ilamb")

_REGISTRY_NAMES = ("ilamb-test", "ilamb", "ilamb-regions")


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
        # The "ilamb-regions" registry is registered with cache_name="ilamb" so its
        # files (region masks) land in the same cache directory as the "ilamb" registry (reference obs).
        # Together those two registries are the data an ingest or `ref providers setup` run cares about,
        # so this single path covers them both.
        return resolve_cache_dir("ilamb")


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
    # Holds the few reference observations  that are not yet published to obs4MIPs/obs4REF
    # (currently WangMao, GLEAMv3 and CRU4.02).
    # The provider fetches these at execute time via registry_to_collection; they are not ingested.
    # As each dataset is published, it moves to the obs4ref registry and drops out of ilamb.txt.
)
dataset_registry_manager.register(
    "ilamb-regions",
    base_url=DATASET_URL,
    package="climate_ref_ilamb.dataset_registry",
    resource="ilamb_regions.txt",
    # Shares the "ilamb" cache directory for legacy reasons.
    cache_name="ilamb",
    use_case=RegistryUseCase.support,
)

# Dynamically register ILAMB diagnostics.
# Sorted so the registration order does not depend on the filesystem
_configure_files = sorted(
    importlib.resources.files("climate_ref_ilamb.configure").iterdir(),
    key=lambda entry: entry.name,
)
for yaml_file in _configure_files:
    if not yaml_file.name.endswith(".yaml"):
        continue
    metrics = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    realm = metrics.pop("realm")
    region_masks = metrics.pop("region_masks", None)
    for metric, options in metrics.items():
        provider.register(
            ILAMBStandard(realm, metric, options.pop("sources"), region_masks=region_masks, **options)
        )
