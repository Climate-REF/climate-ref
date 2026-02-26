"""
Shared utility functions for dataset adapters
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


def parse_datetime(dt_str: pd.Series[str]) -> pd.Series[datetime | Any]:
    """
    Parse datetime strings from dataset files.

    Pandas tries to coerce everything to their own datetime format, which is not what we want here.
    """

    def _inner(date_string: str | datetime | None) -> datetime | None:
        if date_string is None or (not isinstance(date_string, datetime) and pd.isnull(date_string)):
            return None

        # Already parsed — return as-is
        if isinstance(date_string, datetime):
            return date_string

        # Try to parse the date string with and without milliseconds
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        # If all parsing attempts fail, log an error and return None
        logger.error(f"Failed to parse date string: {date_string}")
        return None

    return pd.Series(
        [_inner(dt) for dt in dt_str],
        index=dt_str.index,
        dtype="object",
    )


def clean_branch_time(branch_time: pd.Series[str]) -> pd.Series[float]:
    """
    Clean branch time values, handling missing values and EC-Earth3 suffixes.

    This handles the EC-Earth3 encoding where `branch_time_in_child` and
    `branch_time_in_parent` have a trailing 'D' suffix (e.g. "123D").
    We strip the 'D' and coerce the remaining value to a float,
    treating any missing or malformed entries as NaN.
    """
    return pd.to_numeric(branch_time.astype(str).str.replace("D", ""), errors="coerce")


_VERSION_SEGMENT_RE = re.compile(r"^v\d{8}$|^v\d+$")


def extract_version_from_path(parent: str) -> str:
    """
    Extract the dataset version from a directory path.

    Splits the path into individual directory segments and matches
    version patterns (vYYYYMMDD or vN) against standalone segments only.
    When multiple segments match, the longest (most specific) match wins.
    Falls back to "v0" if no segment matches.

    Parameters
    ----------
    parent
        Parent directory path

    Returns
    -------
    :
        Version string (e.g., "v20250622", "v1", or "v0" as fallback)
    """
    matches = [segment for segment in Path(parent).parts if _VERSION_SEGMENT_RE.match(segment)]
    if matches:
        return max(matches, key=len)
    return "v0"


def parse_drs_daterange(date_range: str) -> tuple[str | None, str | None]:
    """
    Parse a DRS date range string into start and end dates.

    The output from this is an estimated date range until the file is completely parsed.

    Supports date formats used in CMIP6 and CMIP7 filenames:

    - YYYY-YYYY (4 chars, yearly)
    - YYYYMM-YYYYMM (6 chars, monthly)
    - YYYYMMDD-YYYYMMDD (8 chars, daily)
    - YYYYMMDDhhmm-YYYYMMDDhhmm (12 chars, sub-daily)

    Parameters
    ----------
    date_range
        Date range string

    Returns
    -------
    :
        Tuple containing start and end dates as strings in the format "YYYY-MM-DD"
    """
    try:
        start, end = date_range.split("-")
        if len(start) != len(end):
            raise ValueError(f"Mismatched date component lengths: {len(start)} vs {len(end)}")

        if len(start) == 4:  # noqa: PLR2004
            # YYYY — yearly resolution
            start_date = f"{start}-01-01"
            end_date = f"{end}-12-30"
        elif len(start) == 6:  # noqa: PLR2004
            # YYYYMM — monthly resolution
            start_date = f"{start[:4]}-{start[4:6]}-01"
            end_date = f"{end[:4]}-{end[4:6]}-30"
        elif len(start) == 8:  # noqa: PLR2004
            # YYYYMMDD — daily resolution
            start_date = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
            end_date = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
        elif len(start) == 12:  # noqa: PLR2004
            # YYYYMMDDhhmm — sub-daily resolution (time-of-day ignored for date estimate)
            start_date = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
            end_date = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
        else:
            raise ValueError(f"Unsupported date component length: {len(start)}")

        return start_date, end_date
    except ValueError:
        logger.error(f"Invalid date range format: {date_range}")
        return None, None


def validate_path(raw_path: str) -> Path:
    """
    Validate the prefix of a dataset against the data directory
    """
    prefix = Path(raw_path)

    if not prefix.exists():
        raise FileNotFoundError(prefix)

    if not prefix.is_absolute():
        raise ValueError(f"Path {prefix} must be absolute")

    return prefix
