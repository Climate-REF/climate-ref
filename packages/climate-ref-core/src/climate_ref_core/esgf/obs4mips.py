"""
Obs4MIPs dataset request implementation.
"""

from __future__ import annotations

import os.path
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from climate_ref_core.esgf.base import IntakeESGFMixin
from climate_ref_core.esgf.cmip6 import prefix_to_filename

if TYPE_CHECKING:
    import xarray as xr


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

    def generate_output_path(
        self, metadata: pd.Series[Any], ds: xr.Dataset, ds_filename: pathlib.Path
    ) -> Path:
        """
        Create the output path for the dataset following Obs4MIPs DRS.

        Parameters
        ----------
        metadata
            Row from the DataFrame returned by fetch_datasets
        ds
            Loaded xarray dataset
        ds_filename
            Original filename of the dataset

        Returns
        -------
        Path
            Relative path where the dataset should be stored
        """
        output_path = (
            Path(os.path.join(*[metadata[item] for item in self.obs4mips_path_items]))
            / f"v{metadata['version']}"
        )

        # Handle case where filename prefix doesn't match variable_id
        if ds_filename.name.split("_")[0] == ds.variable_id:
            filename_prefix = "_".join([metadata[item] for item in self.obs4mips_filename_paths])
        else:
            filename_prefix = ds_filename.name.split("_")[0] + "_"
            filename_prefix += "_".join(
                [metadata[item] for item in self.obs4mips_filename_paths if item != "variable_id"]
            )

        return output_path / prefix_to_filename(ds, filename_prefix)

    def __repr__(self) -> str:
        return f"Obs4MIPsRequest(slug={self.slug!r}, facets={self.facets!r})"
