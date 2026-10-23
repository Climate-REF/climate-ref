"""Shared files that test sessions and xdist workers build once."""

import fcntl
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_file_lock(path: Path) -> Generator[None]:
    """Hold an advisory process lock for a shared test artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def build_cached_file[T](cache_path: Path, builder: Callable[[Path], T]) -> T | None:
    """Build and atomically publish a file once across xdist workers."""
    with exclusive_file_lock(cache_path.with_suffix(".lock")):
        if cache_path.exists():
            return None

        temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            result = builder(temporary_path)
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return result
