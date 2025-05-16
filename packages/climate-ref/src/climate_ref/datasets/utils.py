import functools
import itertools
import os
import re
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, cast

import cf_xarray  # noqa
import pandas as pd
import tqdm.contrib.concurrent
from attrs import define, field
from loguru import logger

from climate_ref_core.env import env

_VERSION_REGEX = re.compile(r"v\d{4}\d{2}\d{2}|v\d{1}")
_MAX_CHUNK_SIZE = env.int("REF_INGEST_MAX_CHUNK_SIZE", 1000)
_DEFAULT_PARALLEL_THRESHOLD = env.int("REF_INGEST_PARALLEL_THRESHOLD", 100)
_DEFAULT_PMAP = functools.partial(
    tqdm.contrib.concurrent.process_map, max_workers=env.int("REF_INGEST_MAX_WORKERS", os.cpu_count())
)


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


def _ensure_list(raw_path: str | Path | Collection[str | Path]) -> list[Path]:
    """
    Ensure that the input is a list of paths
    """
    if not isinstance(raw_path, Collection):
        raw_path = [raw_path]

    return [Path(p) for p in raw_path]


def get_version_from_filename(input_str: str | Path) -> str | None:
    """
    Extract version information from attribute with regular expressions.
    """
    match = re.findall(_VERSION_REGEX, str(input_str))
    if match:
        return cast(str, max(match, key=len))
    else:
        return None


def _walk_directory(directory: Path, include_pattern: str) -> list[Path]:
    """
    Walk a directory and return a list of all files
    """
    return list(Path(directory).rglob(include_pattern))


def _parse(
    file: Path,
    parsing_func: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return parsing_func(file)
    except Exception as e:
        logger.warning(f"Error parsing {file}: {e}")
        return {"INVALID_ASSET": file, "TRACEBACK": str(e)}


@define
class ParallelBuilder:
    """
    A subclass of Builder that uses joblib to parallelize the directory walking

    This is based on a strongly simplified version of the ecgtools
    """

    paths: Collection[Path] = field(converter=_ensure_list)
    include_pattern: str = "*.nc"
    pmap: Callable[..., Any] = _DEFAULT_PMAP
    """
    Use a parallel map function to parallelize processing

    The default is to use the process_map function from tqdm.contrib.concurrent
    which uses process-based parallelization and generates a process bar.
    """
    parallel_threshold: int = _DEFAULT_PARALLEL_THRESHOLD
    """
    If the number of items being processed is less than this threshold,
    use the default map function instead of the parallel map function.
    The parallel map function is slower for small numbers of items due to the startup time for
    the parallel processes.
    """

    def _find_files(self) -> list[Path]:
        """
        Get the assets in the root directories
        """
        func = functools.partial(_walk_directory, include_pattern=self.include_pattern)

        if len(self.paths) < self.parallel_threshold:
            assets = map(func, self.paths)
        else:
            chunksize = min(len(self.paths) // 10, _MAX_CHUNK_SIZE)
            assets = self.pmap(func, self.paths, chunksize=chunksize)
        # Flatten the list of lists and remove duplicates, then sort
        return sorted(set(itertools.chain.from_iterable(assets)))

    def _parse_datasets(
        self,
        *,
        assets: Collection[Path],
        parsing_func: Callable[[Path], dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Parse dataset metadata from the assets

        Parameters
        ----------
        assets
        parsing_func

        Returns
        -------
        :
            DataFrame containing the metadata for the dataset

        """
        func = functools.partial(_parse, parsing_func=parsing_func)

        if len(assets) < self.parallel_threshold:
            entries = map(func, assets)
        else:
            chunksize = min(len(assets) // 10, _MAX_CHUNK_SIZE)
            entries = self.pmap(func, assets, chunksize=chunksize)
        return pd.DataFrame(entries)

    def get_datasets(self, parsing_func: Callable[[Path], dict[str, Any]]) -> pd.DataFrame:
        """
        Get the datasets in the root directories
        """
        assets = self._find_files()
        logger.info(f"Found {len(assets)} potential datasets")
        datasets = self._parse_datasets(assets=assets, parsing_func=parsing_func)

        if "INVALID_ASSET" in datasets.columns:
            mask = datasets["INVALID_ASSET"].isna()
            logger.error(f"Invalid datasets found. Removing {mask.sum()} datasets")

            datasets = datasets.loc[mask].drop(columns=["INVALID_ASSET"])

        return datasets
