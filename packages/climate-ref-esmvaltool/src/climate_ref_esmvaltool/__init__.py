"""
Rapid evaluating CMIP data with ESMValTool.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
from loguru import logger

import climate_ref_esmvaltool.diagnostics
from climate_ref_core.dataset_registry import (
    DATASET_URL,
    dataset_registry_manager,
    fetch_all_files,
    resolve_cache_dir,
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
        return resolve_cache_dir("esmvaltool")

    def ingest_data(self, config: Config, db: Any) -> None:
        """
        Ingest fetched ESMValTool reference data into the database.

        This records the observational/reanalysis datasets bundled with ESMValTool so
        their use can be tracked for provenance and surfaced in the frontend. The data is
        not CMOR compliant, so it is registered under its own dataset type.

        Note: requires the ``climate-ref`` package. When ``climate-ref-esmvaltool`` is used
        standalone the ingestion is skipped with a warning, mirroring the PMP provider.
        """
        try:
            from climate_ref.datasets import ingest_datasets  # noqa: PLC0415
            from climate_ref.datasets.esmvaltool_reference import (  # noqa: PLC0415
                ESMValToolReferenceDatasetAdapter,
            )
        except ImportError:
            logger.info(
                f"Skipping {self.slug} data ingestion: climate-ref package not installed. "
                "Run `ref datasets ingest --source-type esmvaltool-reference` manually if needed."
            )
            return

        # Reference data is fetched under ``<datasets-registry-cache>/ESMValTool``; the same
        # location build_cmd() configures as the ESMValCore rootpath.
        registry = dataset_registry_manager[_DATASETS_REGISTRY_NAME]
        data_path = registry.abspath / "ESMValTool"  # type: ignore[attr-defined]
        if not data_path.exists():
            logger.warning(
                f"ESMValTool reference data not found at {data_path}. "
                f"Run `ref providers setup --provider {self.slug}` first."
            )
            return

        try:
            stats = ingest_datasets(ESMValToolReferenceDatasetAdapter(), data_path, db, skip_invalid=True)
            stats.log_summary("ESMValTool reference ingestion complete:")
        except ValueError as e:
            logger.warning(f"No valid ESMValTool reference datasets found: {e}")


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
