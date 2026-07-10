"""
Shared utility functions for dataset adapters
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cftime
import pandas as pd
from loguru import logger


def _is_na(value: Any) -> bool:
    """Check if a value is NA/NaN/None, safely handling all types."""
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_db_str(value: Any) -> str | None:
    """
    Coerce a value to its on-disk string form.

    Matches the @validates coercion used by DatasetFile (e.g. cftime.datetime -> str(cftime_obj)).
    This is the normalised form for comparing a freshly-parsed in-memory value
    against the str loaded back from the database.
    Without it, a cross-type ``str != cftime.datetime`` comparison always evaluates True
    and every file appears changed onre-ingest.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


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

        if _is_na(date_value):
            return None

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


def coerce_catalog_times(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce a catalog's stored ``start_time``/``end_time`` strings to cftime objects.

    A no-op when the catalog has no ``start_time`` column (dataset-level catalogs). Each of
    ``start_time``/``end_time`` is coerced independently, so a catalog with only one of the two
    columns present does not raise a ``KeyError``. Each row's ``calendar`` is used when present,
    otherwise ``"standard"``. Mutates and returns ``catalog``.
    """
    if "start_time" in catalog.columns or "end_time" in catalog.columns:
        cal = catalog["calendar"] if "calendar" in catalog.columns else "standard"
        for column in ("start_time", "end_time"):
            if column in catalog.columns:
                catalog[column] = parse_cftime_dates(catalog[column], cal)
    return catalog


def clean_branch_time(branch_time: pd.Series[str]) -> pd.Series[float]:
    """
    Clean branch time values, handling missing values and EC-Earth3 suffixes.

    This handles the EC-Earth3 encoding where `branch_time_in_child` and
    `branch_time_in_parent` have a trailing 'D' suffix (e.g. "123D").
    We strip the 'D' and coerce the remaining value to a float,
    treating any missing or malformed entries as NaN.
    """
    return pd.to_numeric(branch_time.astype(str).str.replace("D", ""), errors="coerce")


# CMIP6 CMOR table -> CMIP6 ``frequency`` CV value.
#
# ESMValTool's ``OBS``/``OBS6`` layouts encode the MIP table (e.g. ``Amon``) rather than the
# frequency, while ``native6`` encodes the frequency directly (e.g. ``mon``). Reference datasets
# store only ``frequency``, so the table has to be reduced to its frequency at parse time.
#
# A CMOR table name is a realm prefix plus a frequency suffix, but the reduction is not a plain
# suffix strip: ``Oclim`` is a monthly climatology (``monC``), ``E1hrClimMon`` is ``1hrCM``, the
# zonal-mean tables (``AERmonZ``, ``EmonZ``, ``EdayZ``, ``E6hrZ``) keep the frequency of their
# non-zonal counterpart, and the ``Pt`` (point-sampled) tables map to distinct ``*Pt``
# frequencies. So the mapping is enumerated rather than derived.
_MIP_TABLE_FREQUENCIES: dict[str, str] = {
    "3hr": "3hr",
    "6hrLev": "6hr",
    "6hrPlev": "6hr",
    "6hrPlevPt": "6hrPt",
    "AERday": "day",
    "AERhr": "1hr",
    "AERmon": "mon",
    "AERmonZ": "mon",
    "Amon": "mon",
    "CF3hr": "3hr",
    "CFday": "day",
    "CFmon": "mon",
    "CFsubhr": "subhrPt",
    "day": "day",
    "E1hr": "1hr",
    "E1hrClimMon": "1hrCM",
    "E3hr": "3hr",
    "E3hrPt": "3hrPt",
    "E6hrZ": "6hr",
    "Eday": "day",
    "EdayZ": "day",
    "Efx": "fx",
    "Emon": "mon",
    "EmonZ": "mon",
    "Esubhr": "subhrPt",
    "Eyr": "yr",
    "IfxAnt": "fx",
    "IfxGre": "fx",
    "ImonAnt": "mon",
    "ImonGre": "mon",
    "IyrAnt": "yr",
    "IyrGre": "yr",
    "LImon": "mon",
    "Lmon": "mon",
    "Oclim": "monC",
    "Oday": "day",
    "Odec": "dec",
    "Ofx": "fx",
    "Omon": "mon",
    "Oyr": "yr",
    "SIday": "day",
    "SImon": "mon",
    "fx": "fx",
}

# The CMIP6 ``frequency`` CV. Values already in this set pass through ``frequency_from_mip_table``
# untouched, which is what lets the ``native6`` layout (which stores a frequency, not a table)
# use the same call site.
_FREQUENCIES: frozenset[str] = frozenset(
    {
        "1hr",
        "1hrCM",
        "1hrPt",
        "3hr",
        "3hrPt",
        "6hr",
        "6hrPt",
        "day",
        "dec",
        "fx",
        "mon",
        "monC",
        "monPt",
        "subhrPt",
        "yr",
        "yrPt",
    }
)


def frequency_from_mip_table(value: str) -> str:
    """
    Reduce a CMOR MIP table name to its CMIP6 ``frequency`` CV value.

    Reference datasets record ``frequency`` and not the MIP table, because ESMValTool's
    ``native6`` layout never carries a table in the first place. This maps the ``OBS``/``OBS6``
    table (e.g. ``Amon`` -> ``mon``) onto the same axis.

    A value that is already a valid frequency is returned unchanged, so a caller parsing a
    ``native6`` path (which yields ``mon``, ``day``, ...) can use this without branching.
    ``day`` and ``fx`` are both a table name and a frequency, and map to themselves either way.

    Parameters
    ----------
    value
        A CMOR MIP table name (``Amon``, ``Omon``, ``SIday``) or an existing frequency (``mon``).

    Returns
    -------
    :
        The corresponding CMIP6 frequency.

    Raises
    ------
    ValueError
        If the value is neither a known MIP table nor a known frequency. Failing loudly is
        deliberate: silently defaulting would let a mis-parsed path collapse two datasets that
        differ only by frequency onto one ``instance_id``.
    """
    if value in _MIP_TABLE_FREQUENCIES:
        return _MIP_TABLE_FREQUENCIES[value]
    if value in _FREQUENCIES:
        return value
    raise ValueError(f"Unknown MIP table or frequency: {value!r}")


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


def build_instance_id(
    datasets: pd.DataFrame,
    drs_items: list[str],
    prefix: str,
    transform: Callable[[str, Any], str] | None = None,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add an ``instance_id`` column built from DRS components.

    Rows where any required DRS component is None/NA are dropped with a warning
    so a single malformed file does not abort the whole ingestion batch.

    Parameters
    ----------
    datasets
        Data catalog with one row per file.
    drs_items
        Column names that make up the instance id, in order.
    prefix
        Prefix to use for the instance id (e.g. ``"CMIP6"``).
    transform
        Optional per-column value transform; defaults to ``str(value)``.
    copy
        If ``True`` (the default), the input DataFrame is left untouched and a
        new one is returned. Set to ``False`` when the caller owns ``datasets``
        and wants to avoid the extra full-table copy — important for streaming
        ingest where ``datasets`` already represents a transient chunk.

    Returns
    -------
    :
        Catalog with the ``instance_id`` column added and invalid rows removed.
    """
    if datasets.empty:
        if copy:
            datasets = datasets.copy()
        datasets["instance_id"] = pd.Series(dtype="object")
        return datasets

    # Build instance_id from individual column views to avoid the per-row pandas
    # accessor overhead of iterrows() and avoid materialising one Series per row.
    columns = [datasets[item].to_numpy() for item in drs_items]
    instance_ids: list[str | None] = []
    for values in zip(*columns):
        parts: list[str] = []
        valid = True
        for item, val in zip(drs_items, values):
            if _is_na(val):
                valid = False
                break
            parts.append(transform(item, val) if transform else str(val))
        instance_ids.append(f"{prefix}." + ".".join(parts) if valid else None)

    if copy:
        datasets = datasets.copy()
    datasets["instance_id"] = pd.array(instance_ids, dtype="object")

    invalid_mask = datasets["instance_id"].isna()
    if not invalid_mask.any():
        return datasets

    invalid_cols = list(drs_items) + (["path"] if "path" in datasets.columns else [])
    invalid_rows = datasets.loc[invalid_mask, invalid_cols]
    for _, row in invalid_rows.iterrows():
        missing = [c for c in drs_items if _is_na(row[c])]
        path = row.get("path", "<unknown>") if "path" in invalid_rows.columns else "<unknown>"
        logger.warning(f"Skipping {path}: missing required DRS components for instance_id: {missing}")
    return datasets.loc[~invalid_mask]


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
