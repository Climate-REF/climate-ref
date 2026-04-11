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
    DatasetRegistryManager,
    dataset_registry_manager,
    fetch_all_files,
    validate_registry_cache,
)
from climate_ref_core.providers import CondaDiagnosticProvider
from climate_ref_esmvaltool._version import __version__
from climate_ref_esmvaltool.diagnostics.base import _DATASETS_REGISTRY_NAME
from climate_ref_esmvaltool.recipe import (
    _ESMVALCORE_URL,
    _ESMVALTOOL_URL,
    _RECIPES_REGISTRY_NAME,
    _RECIPES_URL,
)

if TYPE_CHECKING:
    from climate_ref.config import Config


class ESMValToolProvider(CondaDiagnosticProvider):
    """Provider for ESMValTool diagnostics."""

    def fetch_data(self, config: Config) -> None:
        """Fetch ESMValTool reference data."""
        for registry_name in [_DATASETS_REGISTRY_NAME, _RECIPES_REGISTRY_NAME]:
            registry = dataset_registry_manager[registry_name]
            fetch_all_files(registry, registry_name, output_dir=None)

    def validate_setup(self, config: Config) -> bool:
        """Validate conda environment and data checksums."""
        # First check conda environment
        if not super().validate_setup(config):
            return False

        # Then check data checksums
        errors = []
        for registry_name in [_DATASETS_REGISTRY_NAME, _RECIPES_REGISTRY_NAME]:
            errors.extend(validate_registry_cache(dataset_registry_manager[registry_name], registry_name))
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
    name=_DATASETS_REGISTRY_NAME,
    base_url=DATASET_URL,
    package="climate_ref_esmvaltool.dataset_registry",
    resource="data.txt",
    cache_name=_DATASETS_REGISTRY_NAME.replace("-", "/"),
    legacy_cache_dirs=[
        # As of v0.12.3, cached under pooch.os_cache("climate_ref") with no subdirectory.
        DatasetRegistryManager._resolve_cache_dir("climate_ref")
    ],
)
# Register the ESMValTool recipes.
dataset_registry_manager.register(
    name=_RECIPES_REGISTRY_NAME,
    base_url=_RECIPES_URL,
    package="climate_ref_esmvaltool",
    resource="recipes.txt",
    cache_name=_RECIPES_REGISTRY_NAME.replace("-", "/"),
    legacy_cache_dirs=[
        # As of v0.12.3, cached under pooch.os_cache("climate_ref_esmvaltool").
        Path(pooch.os_cache("climate_ref_esmvaltool"))
    ],
)
