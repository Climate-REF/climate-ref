"""
Rapid evaluating CMIP data
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
from loguru import logger

from climate_ref_core.dataset_registry import (
    DATASET_URL,
    dataset_registry_manager,
    fetch_all_files,
    validate_registry_cache,
)
from climate_ref_core.providers import CondaDiagnosticProvider
from climate_ref_pmp.diagnostics import ENSO, AnnualCycle, ExtratropicalModesOfVariability

if TYPE_CHECKING:
    from climate_ref.config import Config

__version__ = importlib.metadata.version("climate-ref-pmp")

_REGISTRY_NAME = "pmp-climatology"


# Create the PMP diagnostics provider
# PMP uses a conda environment to run the diagnostics
class PMPDiagnosticProvider(CondaDiagnosticProvider):
    """
    Provider for PMP diagnostics.
    """

    def configure(self, config: Config) -> None:
        """Configure the provider."""
        super().configure(config)
        self.env_vars["PCMDI_CONDA_EXE"] = str(self.get_conda_exe())
        # This is a workaround for a fatal error in internal_Finalize of MPICH
        # when running in a conda environment on MacOS.
        # It is not clear if this is a bug in MPICH or a problem with the conda environment.
        if "FI_PROVIDER" not in os.environ:  # pragma: no branch
            logger.debug("Setting env variable 'FI_PROVIDER=tcp'")
            self.env_vars["FI_PROVIDER"] = "tcp"

    def fetch_data(self, config: Config) -> None:
        """Fetch PMP climatology data."""
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
        """Get the path where PMP data is cached."""
        return Path(pooch.os_cache("climate_ref"))

    def ingest_data(self, config: Config, db: Any) -> None:
        """
        Ingest PMP climatology data into the database.

        This registers the climatology datasets so they can be used by diagnostics.

        Note: This method requires the climate-ref package to be installed.
        When using climate-ref-pmp standalone (without climate-ref), ingestion
        will be skipped with a warning message.
        """
        try:
            from climate_ref.datasets import ingest_datasets  # noqa: PLC0415
            from climate_ref.datasets.pmp_climatology import PMPClimatologyDatasetAdapter  # noqa: PLC0415
        except ImportError:
            logger.info(
                f"Skipping {self.slug} data ingestion: climate-ref package not installed. "
                "Run `ref datasets ingest --source-type pmp-climatology` manually if needed."
            )
            return

        data_path = self.get_data_path()
        if data_path is None or not data_path.exists():
            logger.warning(
                f"PMP data path does not exist. Run `ref providers setup --provider {self.slug}` first."
            )
            return

        # Find the pmp-climatology subdirectory
        climatology_path = data_path / _REGISTRY_NAME
        if not climatology_path.exists():
            logger.warning(f"PMP climatology data not found at {climatology_path}")
            return

        adapter = PMPClimatologyDatasetAdapter()

        try:
            stats = ingest_datasets(adapter, climatology_path, config, db, skip_invalid=True)
            stats.log_summary("PMP climatology ingestion complete:")
        except ValueError as e:
            logger.warning(f"No valid PMP climatology datasets found: {e}")


provider = PMPDiagnosticProvider("PMP", __version__)


# Annual cycle diagnostics and metrics
provider.register(AnnualCycle())

# ENSO diagnostics and metrics
# provider.register(ENSO("ENSO_perf"))  # Assigned to ESMValTool
provider.register(ENSO("ENSO_tel"))
provider.register(ENSO("ENSO_proc"))

# Extratropical modes of variability diagnostics and metrics
provider.register(ExtratropicalModesOfVariability("PDO"))
provider.register(ExtratropicalModesOfVariability("NPGO"))
provider.register(ExtratropicalModesOfVariability("NAO"))
provider.register(ExtratropicalModesOfVariability("NAM"))
provider.register(ExtratropicalModesOfVariability("PNA"))
provider.register(ExtratropicalModesOfVariability("NPO"))
provider.register(ExtratropicalModesOfVariability("SAM"))


dataset_registry_manager.register(
    "pmp-climatology",
    base_url=DATASET_URL,
    package="climate_ref_pmp.dataset_registry",
    resource="pmp_climatology.txt",
)
