"""
Rapid evaluating CMIP data with ESMValTool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import climate_ref_esmvaltool.diagnostics
from climate_ref_core.dataset_registry import DATASET_URL, dataset_registry_manager, fetch_all_files
from climate_ref_core.providers import CondaDiagnosticProvider
from climate_ref_esmvaltool._version import __version__
from climate_ref_esmvaltool.recipe import _ESMVALTOOL_COMMIT

if TYPE_CHECKING:
    from climate_ref.config import Config


class ESMValToolProvider(CondaDiagnosticProvider):
    """Provider for ESMValTool diagnostics."""

    def fetch_data(self, config: Config) -> None:
        """Fetch ESMValTool reference data."""
        registry = dataset_registry_manager["esmvaltool"]
        fetch_all_files(registry, "esmvaltool", output_dir=None)


# Initialise the diagnostics manager.
provider = ESMValToolProvider(
    "ESMValTool",
    __version__,
    repo="https://github.com/ESMValGroup/ESMValTool.git",
    tag_or_commit=_ESMVALTOOL_COMMIT,
)

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
