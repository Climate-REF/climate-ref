"""
Catalog builder for discovering and parsing dataset files into a DataFrame
"""

from __future__ import annotations

import fnmatch
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from climate_ref.datasets.base import DatasetParsingFunction

INVALID_ASSET = "INVALID_ASSET"
TRACEBACK = "TRACEBACK"


def discover_files(
    paths: list[str],
    include_patterns: list[str] | None = None,
    depth: int = 0,
) -> list[str]:
    """
    Discover files matching the given glob patterns within the specified paths

    Parameters
    ----------
    paths
        Root directories (or single files) to search
    include_patterns
        Glob patterns to include (e.g. ``["*.nc"]``).
        Defaults to ``["*"]`` if not provided.
    depth
        Maximum directory depth below each root to search.
        ``0`` means only files directly inside the root directory.

    Returns
    -------
    :
        Sorted, deduplicated list of matching file paths
    """
    include_patterns = include_patterns or ["*"]
    assets: list[str] = []

    for root_path in paths:
        root = Path(root_path)
        if not root.exists():
            continue

        if root.is_file():
            if any(fnmatch.fnmatch(root.name, pat) for pat in include_patterns):
                assets.append(str(root))
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            current_depth = len(Path(dirpath).relative_to(root).parts)
            if current_depth >= depth:
                # Still process files at this level, but don't descend further
                dirnames.clear()

            for filename in filenames:
                if any(fnmatch.fnmatch(filename, pat) for pat in include_patterns):
                    assets.append(os.path.join(dirpath, filename))

    return sorted(set(assets))


def _parse_files(
    assets: list[str],
    parsing_func: DatasetParsingFunction,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """
    Parse files using the given parsing function, optionally in parallel

    Parsing is I/O-bound (opening netCDF files), so threads are used
    rather than processes.

    Parameters
    ----------
    assets
        List of file paths to parse
    parsing_func
        Function to extract metadata from each file
    n_jobs
        Number of parallel workers.
        ``1`` = sequential, ``-1`` = all CPUs, ``>1`` = that many threads.

    Returns
    -------
    :
        List of parsed metadata dictionaries
    """
    if n_jobs == 1:
        return [parsing_func(asset) for asset in assets]

    max_workers = None if n_jobs == -1 else n_jobs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(parsing_func, assets))


def build_catalog(
    paths: list[str],
    parsing_func: DatasetParsingFunction,
    include_patterns: list[str] | None = None,
    depth: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Build a catalog DataFrame by discovering and parsing dataset files

    Orchestrates file discovery, parallel parsing, DataFrame construction,
    and INVALID_ASSET row filtering.

    Parameters
    ----------
    paths
        Root directories to search for files
    parsing_func
        Function that parses each file and returns a metadata dictionary.
        Must return a dict with an ``INVALID_ASSET`` key on failure.
    include_patterns
        Glob patterns to include (e.g. ``["*.nc"]``)
    depth
        Maximum directory depth to search
    n_jobs
        Number of parallel workers for parsing.
        ``1`` = sequential, ``-1`` = all CPUs, ``>1`` = that many threads.

    Returns
    -------
    :
        DataFrame containing parsed metadata for all valid files

    Raises
    ------
    ValueError
        If no files matching the include patterns are found in the specified paths
    """
    assets = discover_files(paths, include_patterns=include_patterns, depth=depth)

    if not assets:
        raise ValueError(f"No files matching {include_patterns} found in {paths}")

    entries = _parse_files(assets, parsing_func, n_jobs=n_jobs)
    df = pd.DataFrame(entries)

    # Remove invalid assets
    if INVALID_ASSET in df.columns:
        invalid = df[df[INVALID_ASSET].notnull()]
        if not invalid.empty:
            warnings.warn(
                f"Unable to parse {len(invalid)} assets.",
                stacklevel=2,
            )
            for _, row in invalid.iterrows():
                logger.warning(f"Invalid asset: {row[INVALID_ASSET]}")
        df = df[df[INVALID_ASSET].isnull()].drop(columns=[INVALID_ASSET, TRACEBACK])

    return df
