"""
Shared utilities for reading NetCDF metadata using netCDF4 directly.

These functions avoid the overhead of xarray for metadata-only reads,
providing significant speedup per file by skipping Store construction,
dask array creation, and full CF time decoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cftime

if TYPE_CHECKING:
    import netCDF4

VERTICAL_DIM_NAMES = ("lev", "plev", "olevel", "height", "depth", "level", "altitude")


def read_global_attrs(ds: netCDF4.Dataset, keys: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """
    Read global attributes from a netCDF4 Dataset

    Parameters
    ----------
    ds
        Open netCDF4 Dataset
    keys
        Attribute names to read

    Returns
    -------
    :
        Dictionary mapping attribute names to values (None if missing)
    """
    return {key: getattr(ds, key, None) for key in keys}


def read_variable_attrs(
    ds: netCDF4.Dataset,
    variable_id: str,
    attr_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """
    Read attributes from a specific variable in a netCDF4 Dataset

    Parameters
    ----------
    ds
        Open netCDF4 Dataset
    variable_id
        Name of the variable to read attributes from
    attr_names
        Attribute names to read

    Returns
    -------
    :
        Dictionary mapping attribute names to values (None if missing or variable not found)
    """
    if variable_id and variable_id in ds.variables:
        var = ds.variables[variable_id]
        return {attr: getattr(var, attr, None) for attr in attr_names}
    return {attr: None for attr in attr_names}


def read_vertical_levels(ds: netCDF4.Dataset) -> int:
    """
    Count the number of vertical levels in a netCDF4 Dataset

    Checks known vertical dimension names and returns the size of the first match.

    Parameters
    ----------
    ds
        Open netCDF4 Dataset

    Returns
    -------
    :
        Number of vertical levels (defaults to 1 if no vertical dimension found)
    """
    for dim_name in VERTICAL_DIM_NAMES:
        if dim_name in ds.dimensions:
            return len(ds.dimensions[dim_name])
    return 1


def read_time_bounds(ds: netCDF4.Dataset) -> tuple[str | None, str | None]:
    """
    Read the first and last time values from a netCDF4 Dataset

    Reads only two raw numeric values and decodes them with ``cftime.num2date``,
    matching xarray's CF time decoding output exactly.

    Parameters
    ----------
    ds
        Open netCDF4 Dataset

    Returns
    -------
    :
        Tuple of (start_time, end_time) as strings, or (None, None) if no time variable
        or time dimension is empty
    """
    if "time" not in ds.variables:
        return None, None

    time_var = ds.variables["time"]
    n = len(time_var)
    if n == 0:
        return None, None

    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")

    if not units:
        return None, None

    times = cftime.num2date([time_var[0], time_var[-1]], units, calendar)
    return str(times[0]), str(times[1])


def read_time_metadata(ds: netCDF4.Dataset) -> tuple[str | None, str | None]:
    """
    Read time encoding metadata from a netCDF4 Dataset.

    Parameters
    ----------
    ds
        Open netCDF4 Dataset

    Returns
    -------
    :
        Tuple of (time_units, calendar). Returns (None, None) if no time variable.
    """
    if "time" not in ds.variables:
        return None, None

    time_var = ds.variables["time"]
    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    return units, calendar
