"""
CMIP6 dataset request implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    project = "CMIP6"

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

    def __repr__(self) -> str:
        return f"CMIP6Request(slug={self.slug!r}, facets={self.facets!r})"
