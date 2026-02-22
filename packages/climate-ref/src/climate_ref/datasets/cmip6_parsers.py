"""
CMIP6 parser functions for extracting metadata from netCDF files

Additional non-official DRS's may be added in the future.
"""

import math
import re
import traceback
from pathlib import Path
from typing import Any

import xarray as xr
from loguru import logger


def extract_attr_with_regex(
    input_str: str,
    regex: str,
    strip_chars: str | None = None,
    ignore_case: bool = True,
) -> str | None:
    """
    Extract an attribute from a string using a regex pattern

    If multiple matches are found, the longest match is returned.

    Parameters
    ----------
    input_str
        String to search
    regex
        Regular expression pattern
    strip_chars
        Characters to strip from the match
    ignore_case
        Whether to ignore case when matching

    Returns
    -------
    :
        The longest match, or None if no match is found
    """
    flags = re.IGNORECASE if ignore_case else 0
    matches: list[str] = re.findall(regex, input_str, flags)
    if matches:
        match = max(matches, key=len)
        return match.strip(strip_chars) if strip_chars else match.strip()
    return None


def parse_cmip6_using_directories(file: str) -> dict[str, Any]:
    """
    Extract attributes of a file using information from CMIP6 DRS

    The CMIP6 filename convention uses ``_`` as a delimiter between fields.
    Fields may contain ``-`` internally but never ``_``.

    Filename format::

        <variable_id>_<table_id>_<source_id>_<experiment_id>_<member_id>_<grid_label>[_<time_range>].nc

    For time-invariant fields the time_range segment is omitted.

    Parameters
    ----------
    file
        Path to the CMIP6 file

    Returns
    -------
    :
        Dictionary with extracted metadata, or INVALID_ASSET dict on failure
    """
    filepath = Path(file)
    stem = filepath.stem  # strip .nc
    parts = stem.split("_")

    try:
        if len(parts) == 7:  # noqa: PLR2004
            variable_id, table_id, source_id, experiment_id, member_id, grid_label, time_range = parts
        elif len(parts) == 6:  # noqa: PLR2004
            variable_id, table_id, source_id, experiment_id, member_id, grid_label = parts
            time_range = None
        else:
            return {"INVALID_ASSET": file, "TRACEBACK": f"Cannot parse filename: {stem}"}

        fileparts: dict[str, Any] = {
            "variable_id": variable_id,
            "table_id": table_id,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "member_id": member_id,
            "grid_label": grid_label,
        }
        if time_range is not None:
            fileparts["time_range"] = time_range

        # Extract directory-based metadata
        parent = str(filepath.parent)
        parent_split = parent.split(f"/{source_id}/")
        part_1 = parent_split[0].strip("/").split("/")
        grid_label_from_dir = parent.split(f"/{variable_id}/")[1].strip("/").split("/")[0]
        fileparts["grid_label"] = grid_label_from_dir
        fileparts["activity_id"] = part_1[-2]
        fileparts["institution_id"] = part_1[-1]

        version_regex = r"v\d{4}\d{2}\d{2}|v\d{1}"
        version = extract_attr_with_regex(parent, regex=version_regex) or "v0"
        fileparts["version"] = version
        fileparts["path"] = file

        # Handle DCPP sub-experiments where member_id starts with 's'
        if fileparts["member_id"].startswith("s"):
            fileparts["dcpp_init_year"] = float(fileparts["member_id"].split("-")[0][1:])
            fileparts["member_id"] = fileparts["member_id"].split("-")[-1]
        else:
            fileparts["dcpp_init_year"] = math.nan

    except Exception:
        return {"INVALID_ASSET": file, "TRACEBACK": traceback.format_exc()}

    return fileparts


def _parse_daterange(date_range: str) -> tuple[str | None, str | None]:
    """
    Parse a date range string into start and end dates

    The output from this is an estimated date range until the file is completely parsed.

    Parameters
    ----------
    date_range
        Date range string in the format "YYYYMM-YYYYMM"

    Returns
    -------
    :
        Tuple containing start and end dates as strings in the format "YYYY-MM-DD"
    """
    try:
        start, end = date_range.split("-")
        if len(start) != 6 or len(end) != 6:  # noqa: PLR2004
            raise ValueError("Date range must be in the format 'YYYYMM-YYYYMM'")

        start = f"{start[:4]}-{start[4:6]}-01"
        # Up to the 30th of the month, assuming a 30-day month
        # These values will be corrected later when the file is parsed
        end = f"{end[:4]}-{end[4:6]}-30"

        return start, end
    except ValueError:
        logger.error(f"Invalid date range format: {date_range}")
        return None, None


def parse_cmip6_complete(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Complete parser for CMIP6 files

    This parser loads each file and extracts all available metadata.

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
    keys = sorted(
        {
            "activity_id",
            "branch_method",
            "branch_time_in_child",
            "branch_time_in_parent",
            "experiment",
            "experiment_id",
            "frequency",
            "grid",
            "grid_label",
            "institution_id",
            "nominal_resolution",
            "parent_activity_id",
            "parent_experiment_id",
            "parent_source_id",
            "parent_time_units",
            "parent_variant_label",
            "realm",
            "product",
            "source_id",
            "source_type",
            "sub_experiment",
            "sub_experiment_id",
            "table_id",
            "variable_id",
            "variant_label",
        }
    )

    try:
        with xr.open_dataset(file, chunks={}, use_cftime=True) as ds:
            info = {key: ds.attrs.get(key) for key in keys}
            info["member_id"] = info["variant_label"]

            variable_id = info["variable_id"]
            if variable_id:  # pragma: no branch
                attrs = ds[variable_id].attrs
                for attr in ["standard_name", "long_name", "units"]:
                    info[attr] = attrs.get(attr)

            # Set the default of # of vertical levels to 1
            vertical_levels = 1
            start_time, end_time = None, None
            init_year = None
            try:
                vertical_levels = ds[ds.cf["vertical"].name].size
            except (KeyError, AttributeError, ValueError):
                ...

            try:
                start_time, end_time = str(ds.cf["T"][0].data), str(ds.cf["T"][-1].data)
            except (KeyError, AttributeError, ValueError):
                ...
            if info.get("sub_experiment_id"):  # pragma: no branch
                init_year_str = extract_attr_with_regex(str(info["sub_experiment_id"]), r"\d{4}")
                if init_year_str:  # pragma: no cover
                    init_year = int(init_year_str)
            info["vertical_levels"] = vertical_levels
            info["init_year"] = init_year
            info["start_time"] = start_time
            info["end_time"] = end_time
            if not (start_time and end_time):
                info["time_range"] = None
            else:
                info["time_range"] = f"{start_time}-{end_time}"
        info["path"] = str(file)
        info["version"] = extract_attr_with_regex(str(file), regex=r"v\d{4}\d{2}\d{2}|v\d{1}") or "v0"

        # Mark the dataset as finalised
        # This is used to indicate that the dataset has been fully parsed and is ready for use
        info["finalised"] = True

        return info

    except Exception:
        logger.exception(f"Failed to parse {file}")
        return {"INVALID_ASSET": file, "TRACEBACK": traceback.format_exc()}


def parse_cmip6_drs(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    DRS parser for CMIP6 files

    This parser extracts metadata according to the CMIP6 Data Reference Syntax (DRS).
    This includes the essential metadata required to identify the dataset and is included in the filename.

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
    info: dict[str, Any] = parse_cmip6_using_directories(file)

    if "INVALID_ASSET" in info:
        logger.warning(f"Failed to parse {file}: {info['INVALID_ASSET']}")
        return info

    # The member_id is technically incorrect
    # but for simplicity we are going to ignore sub-experiments for the DRS parser
    info["variant_label"] = info["member_id"]

    # Rename the `dcpp_init_year` key to `init_year` if it exists
    if "dcpp_init_year" in info:
        info["init_year"] = info.pop("dcpp_init_year")

    if info.get("time_range"):
        # Parse the time range if it exists
        start_time, end_time = _parse_daterange(info["time_range"])
        info["start_time"] = start_time
        info["end_time"] = end_time

    info["finalised"] = False

    return info
