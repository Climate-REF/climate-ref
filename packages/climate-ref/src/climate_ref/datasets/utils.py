from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import toolz  # type: ignore[import-untyped]
from ecgtools import Builder


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


def _walk_directory(directory: Any) -> list[Path]:
    """
    Walk a directory and return a list of all files
    """
    return directory.walk()  # type: ignore


class ParallelBuilder(Builder):
    """
    A subclass of Builder that uses joblib to parallelize the directory walking
    """

    def get_assets(self) -> "ParallelBuilder":
        """
        Get the assets in the root directories
        """
        assets = joblib.Parallel(**self.joblib_parallel_kwargs)(  # type: ignore
            joblib.delayed(_walk_directory)(directory)
            for directory in self._root_dirs  # type: ignore
        )
        self.assets = sorted(toolz.unique(toolz.concat(assets)))
        return self
