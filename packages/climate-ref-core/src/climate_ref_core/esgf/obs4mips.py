"""
Obs4MIPs dataset request implementation.
"""

from __future__ import annotations

import intake_esgf
import pandas as pd

from climate_ref_core.esgf.base import IntakeESGFMixin


class Obs4MIPsRequest(IntakeESGFMixin):
    """
    Represents an Obs4MIPs dataset request.

    These data are fetched from ESGF based on the provided facets.
    """

    source_type = "obs4MIPs"
    project = "obs4MIPs"

    obs4mips_path_items = (
        "activity_id",
        "institution_id",
        "source_id",
        "variable_id",
        "grid_label",
    )

    obs4mips_filename_paths = (
        "variable_id",
        "source_id",
        "grid_label",
    )

    avail_facets = (
        "activity_id",
        "institution_id",
        "source_id",
        "frequency",
        "variable_id",
        "grid_label",
        "version",
        "data_node",
        "project",
    )

    def fetch_datasets(self) -> pd.DataFrame:
        """Fetch dataset metadata from ESGF, also searching the CEDA Solr index."""
        # C3S-GTO-ECV-9-0 (the ozone reference) is missing from the default index but CEDA still serves it
        with intake_esgf.conf.set(indices={"esgf.ceda.ac.uk": True}):
            return super().fetch_datasets()

    def __repr__(self) -> str:
        return f"Obs4MIPsRequest(slug={self.slug!r}, facets={self.facets!r})"
