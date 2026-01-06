"""
CMIP6 dataset request implementation.
"""

from __future__ import annotations

import os.path
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from climate_ref_core.esgf.base import IntakeESGFMixin

if TYPE_CHECKING:
    import xarray as xr


def prefix_to_filename(ds: xr.Dataset, filename_prefix: str) -> str:
    """
    Create a filename from a dataset and a prefix.

    Optionally includes the time range of the dataset if it has a time dimension.

    Parameters
    ----------
    ds
        Dataset
    filename_prefix
        Prefix for the filename (includes the different facets of the dataset)

    Returns
    -------
    str
        Filename for the dataset
    """
    if "time" in ds.dims:
        time_range = f"{ds.time.min().dt.strftime('%Y%m').item()}-{ds.time.max().dt.strftime('%Y%m').item()}"
        filename = f"{filename_prefix}_{time_range}.nc"
    else:
        filename = f"{filename_prefix}.nc"
    return filename


class CMIP6Request(IntakeESGFMixin):
    """
    Represents a CMIP6 dataset request.

    These data are fetched from ESGF based on the provided facets.
    """

    source_type = "CMIP6"

    cmip6_path_items = (
        "mip_era",
        "activity_drs",
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "grid_label",
    )

    cmip6_filename_paths = (
        "variable_id",
        "table_id",
        "source_id",
        "experiment_id",
        "member_id",
        "grid_label",
    )

    available_facets = (
        "mip_era",
        "activity_drs",
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
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
        Initialize a CMIP6 request.

        Parameters
        ----------
        slug
            Unique identifier for this request
        facets
            ESGF search facets (e.g., source_id, variable_id, experiment_id)
        remove_ensembles
            If True, keep only one ensemble member per model
        time_span
            Optional time range filter (start, end) in YYYY-MM format
        """
        self.slug = slug
        self.facets = facets
        self.remove_ensembles = remove_ensembles
        self.time_span = time_span

        for key in self.cmip6_path_items:
            if key not in self.available_facets:
                raise ValueError(f"Path item {key!r} not in available facets")
        for key in self.cmip6_filename_paths:
            if key not in self.available_facets:
                raise ValueError(f"Filename path {key!r} not in available facets")

    def generate_output_path(
        self, metadata: pd.Series[Any], ds: xr.Dataset, ds_filename: pathlib.Path
    ) -> Path:
        """
        Create the output path for the dataset following CMIP6 DRS.

        Parameters
        ----------
        metadata
            Row from the DataFrame returned by fetch_datasets
        ds
            Loaded xarray dataset
        ds_filename
            Original filename of the dataset (unused for CMIP6)

        Returns
        -------
        Path
            Relative path where the dataset should be stored
        """
        output_path = (
            Path(os.path.join(*[metadata[item] for item in self.cmip6_path_items]))
            / f"v{metadata['version']}"
        )
        filename_prefix = "_".join([metadata[item] for item in self.cmip6_filename_paths])

        return output_path / prefix_to_filename(ds, filename_prefix)

    def __repr__(self) -> str:
        return f"CMIP6Request(slug={self.slug!r}, facets={self.facets!r})"
