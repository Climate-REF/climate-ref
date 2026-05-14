"""
Catalog builder for discovering and parsing dataset files into a DataFrame
"""

from __future__ import annotations

import fnmatch
import os
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

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


def parse_files(
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
        List of parsed metadata dictionaries, in the same order as ``assets``.
    """
    if n_jobs == 1:
        return [parsing_func(asset) for asset in tqdm(assets, desc="Parsing files", unit="file")]

    max_workers = None if n_jobs == -1 else n_jobs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(executor.map(parsing_func, assets), total=len(assets), desc="Parsing files", unit="file")
        )


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

    logger.info(f"Discovered {len(assets)} files matching {include_patterns} in {paths}")

    entries = parse_files(assets, parsing_func, n_jobs=n_jobs)
    df = pd.DataFrame(entries)
    df = _filter_invalid_rows(df)

    logger.info(f"Built catalog with {len(df)} valid entries ({len(assets) - len(df)} invalid)")

    return df


def iter_discovered_chunks(
    paths: list[str],
    include_patterns: list[str] | None = None,
    depth: int = 0,
    chunk_size: int = 10_000,
) -> Iterator[list[str]]:
    """
    Yield batches of discovered file paths in chunks.

    Walks the directory tree once and yields batches of up to ``chunk_size``
    paths. Batches only flush at directory boundaries, so files within a
    single directory are kept together — important for DRS-style layouts
    where one directory holds all files for a single dataset.

    Parameters
    ----------
    paths
        Root directories (or single files) to search.
    include_patterns
        Glob patterns to include (e.g. ``["*.nc"]``).
        Defaults to ``["*"]``.
    depth
        Maximum directory depth below each root to search.
    chunk_size
        Soft target for the number of files per batch. A batch may exceed
        this if a single directory contains more matching files.

    Yields
    ------
    :
        Lists of file paths. Each list is sorted and deduplicated.
    """
    include_patterns = include_patterns or ["*"]
    buffer: list[str] = []

    def _flush() -> Iterator[list[str]]:
        nonlocal buffer
        if buffer:
            yield sorted(set(buffer))
            buffer = []

    for root_path in paths:
        root = Path(root_path)
        if not root.exists():
            continue

        if root.is_file():
            if any(fnmatch.fnmatch(root.name, pat) for pat in include_patterns):
                buffer.append(str(root))
            if len(buffer) >= chunk_size:
                yield from _flush()
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            current_depth = len(Path(dirpath).relative_to(root).parts)
            if current_depth >= depth:
                dirnames.clear()

            matched = [
                os.path.join(dirpath, fn)
                for fn in filenames
                if any(fnmatch.fnmatch(fn, pat) for pat in include_patterns)
            ]
            if not matched:
                continue

            if buffer and len(buffer) + len(matched) > chunk_size:
                yield from _flush()
            buffer.extend(matched)

    yield from _flush()


def _filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop INVALID_ASSET rows from a parsed catalog DataFrame.

    Warns once per call summarising the count, logs each invalid path
    at warning level, then returns a DataFrame with the INVALID_ASSET
    and TRACEBACK columns removed.
    """
    if INVALID_ASSET not in df.columns:
        return df

    invalid = df[df[INVALID_ASSET].notnull()]
    if not invalid.empty:
        warnings.warn(f"Unable to parse {len(invalid)} assets.", stacklevel=3)
        for _, row in invalid.iterrows():
            logger.warning(f"Invalid asset: {row[INVALID_ASSET]}")
    return df[df[INVALID_ASSET].isnull()].drop(columns=[INVALID_ASSET, TRACEBACK])


def iter_built_catalogs(  # noqa: PLR0913
    paths: list[str],
    parsing_func: DatasetParsingFunction,
    include_patterns: list[str] | None = None,
    depth: int = 0,
    n_jobs: int = 1,
    chunk_size: int = 10_000,
) -> Iterator[pd.DataFrame]:
    """
    Yield catalog DataFrames in chunks, parsing files chunk by chunk.

    Peak memory is bounded by ``chunk_size`` files because each chunk's
    parsed entries and DataFrame are released before the next chunk
    starts parsing.

    Parameters
    ----------
    paths
        Root directories to search for files.
    parsing_func
        Function that parses each file and returns a metadata dictionary.
        Must return a dict with an ``INVALID_ASSET`` key on failure.
    include_patterns
        Glob patterns to include (e.g. ``["*.nc"]``).
    depth
        Maximum directory depth to search.
    n_jobs
        Number of parallel workers per chunk for parsing.
    chunk_size
        Soft target for the number of files per chunk.

    Yields
    ------
    :
        DataFrames with parsed metadata for each chunk. Empty chunks
        (all invalid) are skipped.
    """
    any_emitted = False
    for chunk_paths in iter_discovered_chunks(
        paths, include_patterns=include_patterns, depth=depth, chunk_size=chunk_size
    ):
        logger.info(f"Parsing chunk of {len(chunk_paths)} files")
        entries = parse_files(chunk_paths, parsing_func, n_jobs=n_jobs)
        df = pd.DataFrame(entries)
        del entries

        df = _filter_invalid_rows(df)
        if df.empty:
            continue

        any_emitted = True
        yield df

    if not any_emitted:
        logger.info(f"No valid files found in {paths} matching {include_patterns}")
