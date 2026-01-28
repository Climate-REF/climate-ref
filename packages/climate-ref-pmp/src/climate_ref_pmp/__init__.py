"""
Rapid evaluating CMIP data
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import TYPE_CHECKING

from loguru import logger

from climate_ref_core.dataset_registry import DATASET_URL, dataset_registry_manager, fetch_all_files
from climate_ref_core.providers import CondaDiagnosticProvider
from climate_ref_pmp.diagnostics import ENSO, AnnualCycle, ExtratropicalModesOfVariability

if TYPE_CHECKING:
    from climate_ref.config import Config

__version__ = importlib.metadata.version("climate-ref-pmp")


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
        registry = dataset_registry_manager["pmp-climatology"]
        fetch_all_files(registry, "pmp-climatology", output_dir=None)


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
