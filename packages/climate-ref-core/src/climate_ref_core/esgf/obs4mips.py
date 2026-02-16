"""
Obs4MIPs dataset request implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from climate_ref_core.esgf.base import IntakeESGFMixin

if TYPE_CHECKING:
    pass


class Obs4MIPsRequest(IntakeESGFMixin):
    """
    Represents an Obs4MIPs dataset request.

    These data are fetched from ESGF based on the provided facets.
    """

    source_type = "obs4MIPs"

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

    def __init__(
        self,
        slug: str,
        facets: dict[str, Any],
        remove_ensembles: bool = False,
        time_span: tuple[str, str] | None = None,
    ):
        """
        Initialize an Obs4MIPs request.

        Parameters
        ----------
        slug
            Unique identifier for this request
        facets
            ESGF search facets (e.g., source_id, variable_id)
        remove_ensembles
            If True, keep only one ensemble member (typically not relevant for obs)
        time_span
            Optional time range filter (start, end) in YYYY-MM format
        """
        self.slug = slug
        self.facets = facets
        self.remove_ensembles = remove_ensembles
        self.time_span = time_span

        for key in self.obs4mips_path_items:
            if key not in self.avail_facets:
                raise ValueError(f"Path item {key!r} not in available facets")
        for key in self.obs4mips_filename_paths:
            if key not in self.avail_facets:
                raise ValueError(f"Filename path {key!r} not in available facets")

    def fetch_datasets(self) -> pd.DataFrame:
        """Fetch dataset metadata from ESGF with project=obs4MIPs."""
        # Ensure project facet is set to obs4MIPs
        if "project" not in self.facets:
            self.facets["project"] = "obs4MIPs"

        return super().fetch_datasets()

    def __repr__(self) -> str:
        return f"Obs4MIPsRequest(slug={self.slug!r}, facets={self.facets!r})"
