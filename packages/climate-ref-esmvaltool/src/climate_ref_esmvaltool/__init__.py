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
    RegistryUseCase,
    dataset_registry_manager,
    fetch_all_files,
    resolve_cache_dir,
    validate_registry_cache,
)
from climate_ref_core.providers import CondaDiagnosticProvider
from climate_ref_core.source_types import SourceDatasetType
from climate_ref_esmvaltool._version import __version__
from climate_ref_esmvaltool.diagnostics.base import _DATASETS_REGISTRY_NAME, registry_data_root
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

    def ingest_data(self, config: Config, db: Any) -> None:
        """
        Ingest the fetched ESMValTool reference data into the database.

        A diagnostic that declares an `esmvaltool-reference` requirement
        selects its reference data from the catalog,
        so the downloaded files have to be ingested before the solver can find them.

        Ingestion needs the climate-ref package, so it is skipped when the provider is
        installed on its own.
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

        reference_path = registry_data_root()
        if not reference_path.exists():
            logger.warning(
                f"ESMValTool reference data not found at {reference_path}. "
                f"Run `ref providers setup --provider {self.slug}` first."
            )
            return

        try:
            stats = ingest_datasets(ESMValToolReferenceDatasetAdapter(), reference_path, db)
            stats.log_summary("ESMValTool reference ingestion complete:")
        except ValueError as e:
            logger.warning(f"No valid ESMValTool reference datasets found: {e}")

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

# Register OBS, OBS6, and raw data as ESMValTool reference datasets.
#
# data.txt mixes OBS/OBS6 Tier2/3, native6 raw ERA5 and an obs4MIPs GPCP-V2.3 subset.
# The ESMValCore-DRS adapter (`ESMValToolReferenceDatasetAdapter`) reads metadata from the DRS path
# rather than the file contents, so the whole registry ingests as the `esmvaltool-reference` source type.
# Declaring an `esmvaltool-reference` data requirement only gives the files the solver selected.
dataset_registry_manager.register(
    name=_DATASETS_REGISTRY_NAME,
    base_url=DATASET_URL,
    package="climate_ref_esmvaltool.dataset_registry",
    resource="data.txt",
    cache_name=_DATASETS_REGISTRY_NAME.replace("-", "/"),
    source_type=SourceDatasetType.ESMValToolReference,
    use_case=RegistryUseCase.reference,
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
    use_case=RegistryUseCase.support,
)
