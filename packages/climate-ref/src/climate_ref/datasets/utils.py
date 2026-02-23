"""
Shared utility functions for dataset adapters
"""

from __future__ import annotations

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
