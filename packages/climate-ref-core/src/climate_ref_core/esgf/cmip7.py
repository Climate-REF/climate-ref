"""
CMIP7 dataset request implementation.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import intake_esgf
import pandas as pd
from loguru import logger

from climate_ref_core.esgf.base import IntakeESGFMixin
from climate_ref_core.exceptions import DatasetResolutionError

STAC_INDICES = ("discovery.east.esgf.io", "discovery.west.esgf.io")
"""The ESGF indices that publish CMIP7 data."""


def _esgf_id(instance_id: str) -> str:
    """Spell a REF CMIP7 ``instance_id`` the way ESGF identifies the dataset."""
    return f"MIP-DRS7.{instance_id}"


class CMIP7Request(IntakeESGFMixin):
    """
    Represents a request for CMIP7 datasets published on ESGF.
    """

    source_type = "CMIP7"
    project = "CMIP7"
    facet_renames: ClassVar[Mapping[str, str]] = {"branding_suffix": "variable_branding_suffix"}

    def _pinned_facets(self, instance_ids: tuple[str, ...]) -> dict[str, Any]:
        # The STAC indices cannot be searched by id, only by the id without its version
        base_ids, versions = zip(*(_esgf_id(instance_id).rsplit(".", 1) for instance_id in instance_ids))
        return {"base_id": sorted(set(base_ids)), "version": sorted({v.removeprefix("v") for v in versions})}

    def _select_pinned(self, results: pd.DataFrame) -> pd.DataFrame:
        wanted = {_esgf_id(instance_id) for instance_id in self.pinned_instance_ids or ()}
        found = results["id"].map(lambda ids: {str(i).split("|")[0] for i in ids} & wanted)

        # intake-esgf keeps only the newest version it finds of each dataset
        # TODO: fix this upstream in intake-esgf as ESGF has the information
        missing = wanted - set().union(*found)
        if missing:
            raise DatasetResolutionError(
                f"ESGF no longer resolves {len(missing)} pinned datasets for {self.slug}: {sorted(missing)}"
            )
        return results[found.map(bool)]

    def fetch_datasets(self) -> pd.DataFrame:
        """Fetch dataset metadata from ESGF, warning if no index that publishes CMIP7 is enabled."""
        enabled = intake_esgf.conf["stac_indices"]

        if not any(enabled.get(index) for index in STAC_INDICES):
            logger.warning(
                f"No CMIP7 index enabled in {intake_esgf.conf.filename}: enable {', '.join(STAC_INDICES)}"
            )
        return super().fetch_datasets()

    def __repr__(self) -> str:
        return f"CMIP7Request(slug={self.slug!r}, facets={self.facets!r})"
