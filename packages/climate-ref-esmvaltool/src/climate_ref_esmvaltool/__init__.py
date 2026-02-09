"""
Rapid evaluating CMIP data with ESMValTool.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pooch
from loguru import logger

import climate_ref_esmvaltool.diagnostics
from climate_ref_core.dataset_registry import (
    DATASET_URL,
    dataset_registry_manager,
    fetch_all_files,
    validate_registry_cache,
)
from climate_ref_core.providers import CondaDiagnosticProvider
from climate_ref_esmvaltool._version import __version__
from climate_ref_esmvaltool.recipe import _ESMVALCORE_URL, _ESMVALTOOL_URL

if TYPE_CHECKING:
    from climate_ref.config import Config

_REGISTRY_NAME = "esmvaltool"


class ESMValToolProvider(CondaDiagnosticProvider):
    """Provider for ESMValTool diagnostics."""

    def fetch_data(self, config: Config) -> None:
        """Fetch ESMValTool reference data."""
        registry = dataset_registry_manager[_REGISTRY_NAME]
        fetch_all_files(registry, _REGISTRY_NAME, output_dir=None)

    def validate_setup(self, config: Config) -> bool:
        """Validate conda environment and data checksums."""
        # First check conda environment
        if not super().validate_setup(config):
            return False

        # Then check data checksums
        registry = dataset_registry_manager[_REGISTRY_NAME]
        errors = validate_registry_cache(registry, _REGISTRY_NAME)
        if errors:
            for error in errors:
                logger.error(f"{self.slug} validation failed: {error}")
            logger.error(
                f"Data for {self.slug} is missing or corrupted. "
                f"Please run `ref providers setup --provider {self.slug}` to fetch data."
            )
            return False
        return True

    def get_data_path(self) -> Path | None:
        """Get the path where ESMValTool data is cached."""
        return Path(pooch.os_cache("climate_ref"))


# Initialise the diagnostics manager.
provider = ESMValToolProvider(
    "ESMValTool",
    __version__,
)
provider.pip_packages = [_ESMVALTOOL_URL, _ESMVALCORE_URL]

# Register the diagnostics.
for _diagnostic_cls_name in climate_ref_esmvaltool.diagnostics.__all__:
    _diagnostic_cls = getattr(climate_ref_esmvaltool.diagnostics, _diagnostic_cls_name)
    provider.register(_diagnostic_cls())

# Register OBS, OBS6, and raw data
dataset_registry_manager.register(
    "esmvaltool",
    base_url=DATASET_URL,
    package="climate_ref_esmvaltool.dataset_registry",
    resource="data.txt",
)
