"""
CMIP7 parser functions for extracting metadata from netCDF files

Includes both a fast DRS parser (filename/directory only) and a complete parser
that opens netCDF files to extract all metadata.
"""

import traceback
from pathlib import Path
from typing import Any

import netCDF4
from loguru import logger

from climate_ref.datasets.netcdf_utils import (
    read_mandatory_attr,
    read_time_bounds,
    read_time_metadata,
    read_variable_attrs,
)
from climate_ref.datasets.utils import extract_version_from_path, parse_drs_daterange


def parse_cmip7_using_directories(file: str) -> dict[str, Any]:
    """
    Extract attributes of a file using information from CMIP7 DRS

    The CMIP7 filename convention (MIP-DRS7 spec) uses ``_`` as a delimiter between fields.
    Fields may contain ``-`` internally but never ``_``.

    Filename format::

        <variable_id>_<branding_suffix>_<frequency>_<region>_<grid_label>_
        <source_id>_<experiment_id>_<variant_label>[_<timeRangeDD>].nc

    Directory structure::

        <drs_specs>/<mip_era>/<activity_id>/<institution_id>/<source_id>/
        <experiment_id>/<variant_label>/<region>/<frequency>/<variable_id>/
        <branding_suffix>/<grid_label>/<version>

    Parameters
    ----------
    file
        Path to the CMIP7 file

    Returns
    -------
    :
        Dictionary with extracted metadata, or INVALID_ASSET dict on failure
    """
    filepath = Path(file)
    stem = filepath.stem  # strip .nc
    parts = stem.split("_")

    try:
        if len(parts) == 9:  # noqa: PLR2004
            (
                variable_id,
                branding_suffix,
                frequency,
                region,
                grid_label,
                source_id,
                experiment_id,
                variant_label,
                time_range,
            ) = parts
        elif len(parts) == 8:  # noqa: PLR2004
            (
                variable_id,
                branding_suffix,
                frequency,
                region,
                grid_label,
                source_id,
                experiment_id,
                variant_label,
            ) = parts
            time_range = None
        else:
            return {"INVALID_ASSET": file, "TRACEBACK": f"Cannot parse CMIP7 filename: {stem}"}

        fileparts: dict[str, Any] = {
            "variable_id": variable_id,
            "branding_suffix": branding_suffix,
            "frequency": frequency,
            "region": region,
            "grid_label": grid_label,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "variant_label": variant_label,
        }
        if time_range is not None:
            fileparts["time_range"] = time_range

        # Extract directory-based metadata
        # activity_id and institution_id sit above source_id in the DRS directory tree
        parent = str(filepath.parent)
        parent_split = parent.split(f"/{source_id}/")
        part_before_source = parent_split[0].strip("/").split("/")
        fileparts["activity_id"] = part_before_source[-2]
        fileparts["institution_id"] = part_before_source[-1]

        # Version from directory path (vYYYYMMDD or vN)
        version = extract_version_from_path(parent)
        fileparts["version"] = version
        fileparts["path"] = file

    except Exception:
        return {"INVALID_ASSET": file, "TRACEBACK": traceback.format_exc()}

    return fileparts


def parse_cmip7_complete(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Complete parser for CMIP7 files

    This parser loads each file and extracts all available metadata
    from netCDF global attributes.

    For some filesystems this may be slow, as it involves a lot of I/O operations.

    Parameters
    ----------
    file
        File to parse
    kwargs
        Additional keyword arguments (not used, but required for compatibility)

    Returns
    -------
    :
        Dictionary with extracted metadata
    """
    # We don't extract all attributes (subvalues of the branding_suffix and variant_label components)
    # These can be added later if needed or derived
    try:
        with netCDF4.Dataset(file, "r") as ds:
            start_time, end_time = read_time_bounds(ds)
            time_units, calendar = read_time_metadata(ds)

            variable_id = read_mandatory_attr(ds, "variable_id")
            var_attrs = read_variable_attrs(ds, variable_id, ["standard_name", "long_name", "units"])

            return {
                # Core DRS attributes
                "activity_id": read_mandatory_attr(ds, "activity_id"),
                "institution_id": read_mandatory_attr(ds, "institution_id"),
                "source_id": read_mandatory_attr(ds, "source_id"),
                "experiment_id": read_mandatory_attr(ds, "experiment_id"),
                "variant_label": read_mandatory_attr(ds, "variant_label"),
                "variable_id": variable_id,
                "grid_label": read_mandatory_attr(ds, "grid_label"),
                "frequency": read_mandatory_attr(ds, "frequency"),
                "region": read_mandatory_attr(ds, "region"),
                "branding_suffix": read_mandatory_attr(ds, "branding_suffix"),
                "branded_variable": read_mandatory_attr(ds, "branded_variable"),
                "version": extract_version_from_path(str(Path(file).parent)),
                # Additional mandatory attributes
                "mip_era": read_mandatory_attr(ds, "mip_era"),
                "realm": read_mandatory_attr(ds, "realm"),
                "nominal_resolution": read_mandatory_attr(ds, "nominal_resolution"),
                "license_id": read_mandatory_attr(ds, "license_id"),
                # Parent info (nullable)
                "branch_time_in_child": getattr(ds, "branch_time_in_child", None),
                "branch_time_in_parent": getattr(ds, "branch_time_in_parent", None),
                "parent_activity_id": getattr(ds, "parent_activity_id", None),
                "parent_experiment_id": getattr(ds, "parent_experiment_id", None),
                "parent_mip_era": getattr(ds, "parent_mip_era", None),
                "parent_source_id": getattr(ds, "parent_source_id", None),
                "parent_time_units": getattr(ds, "parent_time_units", None),
                "parent_variant_label": getattr(ds, "parent_variant_label", None),
                # Conditionally required attributes
                "external_variables": getattr(ds, "external_variables", None),
                # Variable metadata
                "standard_name": var_attrs["standard_name"],
                "long_name": var_attrs["long_name"],
                "units": var_attrs["units"],
                # File-level metadata
                "tracking_id": read_mandatory_attr(ds, "tracking_id"),
                # Time information
                "start_time": start_time,
                "end_time": end_time,
                "time_range": f"{start_time}-{end_time}" if start_time and end_time else None,
                "time_units": time_units,
                "calendar": calendar,
                # Path
                "path": file,
                # Finalisation status
                "finalised": True,
            }
    except Exception:
        return {
            "INVALID_ASSET": file,
            "TRACEBACK": traceback.format_exc(),
        }


def parse_cmip7_drs(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    DRS parser for CMIP7 files

    This parser extracts metadata according to the CMIP7 Data Reference Syntax (MIP-DRS7).
    This includes the essential metadata required to identify the dataset
    and is included in the filename and directory structure.

    Parameters
    ----------
    file
        File to parse
    kwargs
        Additional keyword arguments (not used, but required for compatibility)

    Returns
    -------
    :
        Dictionary with extracted metadata
    """
    info: dict[str, Any] = parse_cmip7_using_directories(file)

    if "INVALID_ASSET" in info:
        logger.warning(f"Failed to parse {file}: {info['INVALID_ASSET']}")
        return info

    if info.get("time_range"):
        start_time, end_time = parse_drs_daterange(info["time_range"])
        info["start_time"] = start_time
        info["end_time"] = end_time

    # mip_era is always CMIP7 for this parser
    info["mip_era"] = "CMIP7"

    info["finalised"] = False

    return info
