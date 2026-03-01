"""
Shared utility functions for dataset adapters
"""

from __future__ import annotations

import re
from pathlib import Path

import cftime
import pandas as pd
from loguru import logger


def sort_data_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a dataset catalog DataFrame by instance_id and start_time (with NA values last).

    This provides a stable ordering for testing and debugging.

    Parameters
    ----------
    catalog
        Dataset catalog DataFrame with at least "instance_id" and "start_time" columns

    Returns
    -------
    :
        Sorted DataFrame
    """

    def _sort_key(col: pd.Series) -> pd.Series:
        return col.apply(str) if col.name == "start_time" else col

    return catalog.sort_values(
        ["instance_id", "start_time"],
        key=_sort_key,
    ).reset_index(drop=True)


def parse_cftime_dates(
    dt_str: pd.Series[str],
    calendar: pd.Series[str] | str = "standard",
) -> pd.Series:
    """
    Parse date strings to cftime.datetime objects

    Parameters
    ----------
    dt_str
        Series of date strings in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" format
    calendar
        Calendar name(s). Either a single string applied to all rows,
        or a Series with per-row calendar values.
    """
    # regex to parse a iso formatted date string with optional time component
    _DATE_RE = re.compile(
        r"^(\d{4})-(\d{2})-(\d{2})"
        r"(?:\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?)?"
        r"$"
    )

    def _inner(date_value: object, cal_value: object) -> cftime.datetime | None:
        # Resolve calendar, defaulting to "standard" for missing/NA values
        cal = cal_value if isinstance(cal_value, str) and cal_value else "standard"

        # Pass through cftime objects unchanged
        if isinstance(date_value, cftime.datetime):
            return date_value

        # Handle None and pandas NA/NaN
        if date_value is None:
            return None
        try:
            if pd.isnull(date_value):  # type: ignore[call-overload]
                return None
        except (TypeError, ValueError):
            pass

        # Convert any date-like value (str, pd.Timestamp, datetime) to string for regex parsing
        date_str = date_value if isinstance(date_value, str) else str(date_value)

        # Parse using regex as strptime doesn't support all calendar types
        m = _DATE_RE.match(date_str.strip())
        if not m:
            logger.error(f"Failed to parse date string: {date_str}")
            return None

        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour = int(m.group(4)) if m.group(4) else 0
        minute = int(m.group(5)) if m.group(5) else 0
        second = int(m.group(6)) if m.group(6) else 0
        microsecond = 0
        if m.group(7):
            frac = m.group(7).ljust(6, "0")[:6]
            microsecond = int(frac)

        try:
            return cftime.datetime(  # type: ignore[call-arg]
                year, month, day, hour, minute, second, microsecond, calendar=cal
            )
        except ValueError:
            logger.error(f"Failed to create cftime date from: {date_str} (calendar={cal})")
            return None

    # Determine per-row calendar values
    if isinstance(calendar, str):
        calendars = [calendar] * len(dt_str)
    else:
        calendars = list(calendar)

    return pd.Series(
        [_inner(dt, cal) for dt, cal in zip(dt_str, calendars)],
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
