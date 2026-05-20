from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from climate_ref.datasets.catalog_builder import (
    build_catalog,
    discover_files,
    iter_built_catalogs,
    iter_discovered_chunks,
)


@pytest.fixture
def tmp_tree(tmp_path):
    """Create a directory tree with .nc and .txt files at various depths."""
    # depth 0
    (tmp_path / "root.nc").touch()
    (tmp_path / "root.txt").touch()

    # depth 1
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.nc").touch()

    # depth 2
    deep = sub / "deep"
    deep.mkdir()
    (deep / "b.nc").touch()

    # depth 3
    very_deep = deep / "level3"
    very_deep.mkdir()
    (very_deep / "c.nc").touch()

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    return d


class TestDiscoverFiles:
    def test_finds_nc_files(self, tmp_tree):
        files = discover_files([str(tmp_tree)], include_patterns=["*.nc"], depth=10)
        names = [Path(f).name for f in files]
        assert "root.nc" in names
        assert "a.nc" in names
        assert "b.nc" in names
        assert "c.nc" in names
        # .txt should be excluded
        assert "root.txt" not in names

    def test_depth_limiting(self, tmp_tree):
        # depth=0: only root directory
        files_d0 = discover_files([str(tmp_tree)], include_patterns=["*.nc"], depth=0)
        names_d0 = [Path(f).name for f in files_d0]
        assert names_d0 == ["root.nc"]

        # depth=1: root + sub
        files_d1 = discover_files([str(tmp_tree)], include_patterns=["*.nc"], depth=1)
        names_d1 = sorted(Path(f).name for f in files_d1)
        assert names_d1 == ["a.nc", "root.nc"]

        # depth=2: root + sub + deep
        files_d2 = discover_files([str(tmp_tree)], include_patterns=["*.nc"], depth=2)
        names_d2 = sorted(Path(f).name for f in files_d2)
        assert names_d2 == ["a.nc", "b.nc", "root.nc"]

    def test_nonexistent_path(self):
        files = discover_files(["/nonexistent/path"], include_patterns=["*.nc"], depth=5)
        assert files == []

    def test_single_file_path(self, tmp_tree):
        nc_file = str(tmp_tree / "root.nc")
        files = discover_files([nc_file], include_patterns=["*.nc"], depth=0)
        assert files == [nc_file]

        # Non-matching pattern
        files = discover_files([nc_file], include_patterns=["*.txt"], depth=0)
        assert files == []

    def test_empty_directory(self, empty_dir):
        files = discover_files([str(empty_dir)], include_patterns=["*.nc"], depth=5)
        assert files == []

    def test_default_include_all(self, tmp_tree):
        files = discover_files([str(tmp_tree)], depth=0)
        names = sorted(Path(f).name for f in files)
        assert "root.nc" in names
        assert "root.txt" in names


def _good_parser(file: str, **kwargs: Any) -> dict[str, Any]:
    """Parser that returns valid metadata."""
    return {"path": file, "name": Path(file).name}


def _mixed_parser(file: str, **kwargs: Any) -> dict[str, Any]:
    """Parser that marks .txt files as invalid."""
    if file.endswith(".txt"):
        return {"INVALID_ASSET": file, "TRACEBACK": "not a netCDF file"}
    return {"path": file, "name": Path(file).name}


class TestBuildCatalog:
    def test_sequential(self, tmp_tree):
        df = build_catalog(
            paths=[str(tmp_tree)],
            parsing_func=_good_parser,
            include_patterns=["*.nc"],
            depth=10,
            n_jobs=1,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "path" in df.columns
        assert "name" in df.columns

    def test_parallel(self, tmp_tree):
        df = build_catalog(
            paths=[str(tmp_tree)],
            parsing_func=_good_parser,
            include_patterns=["*.nc"],
            depth=10,
            n_jobs=2,
        )
        assert len(df) == 4

    def test_parallel_all_cpus(self, tmp_tree):
        df = build_catalog(
            paths=[str(tmp_tree)],
            parsing_func=_good_parser,
            include_patterns=["*.nc"],
            depth=10,
            n_jobs=-1,
        )
        assert len(df) == 4

    def test_invalid_asset_filtering(self, tmp_tree):
        # Include both .nc and .txt, parser marks .txt as invalid
        with pytest.warns(UserWarning, match="Unable to parse"):
            df = build_catalog(
                paths=[str(tmp_tree)],
                parsing_func=_mixed_parser,
                include_patterns=["*.nc", "*.txt"],
                depth=0,
                n_jobs=1,
            )
        # Only root.nc should survive (depth=0, .txt filtered as INVALID)
        assert len(df) == 1
        assert "INVALID_ASSET" not in df.columns
        assert "TRACEBACK" not in df.columns

    def test_empty_directory_raises(self, empty_dir):
        with pytest.raises(ValueError, match="No files matching"):
            build_catalog(
                paths=[str(empty_dir)],
                parsing_func=_good_parser,
                include_patterns=["*.nc"],
                depth=5,
            )

    def test_all_invalid_returns_empty(self, tmp_tree):
        def _all_invalid(file: str, **kwargs: Any) -> dict[str, Any]:
            return {"INVALID_ASSET": file, "TRACEBACK": "always fails"}

        with pytest.warns(UserWarning, match="Unable to parse"):
            df = build_catalog(
                paths=[str(tmp_tree)],
                parsing_func=_all_invalid,
                include_patterns=["*.nc"],
                depth=10,
            )
        assert df.empty


def _flat_tree(tmp_path: Path, n: int) -> Path:
    """Create ``n`` .nc files in a single directory."""
    root = tmp_path / "flat"
    root.mkdir()
    for i in range(n):
        (root / f"file_{i:04d}.nc").touch()
    return root


def _wide_tree(tmp_path: Path, num_dirs: int, files_per_dir: int) -> Path:
    """Create ``num_dirs`` sibling directories, each with ``files_per_dir`` .nc files."""
    root = tmp_path / "wide"
    root.mkdir()
    for d in range(num_dirs):
        sub = root / f"ds_{d:03d}"
        sub.mkdir()
        for i in range(files_per_dir):
            (sub / f"file_{d:03d}_{i:03d}.nc").touch()
    return root


class TestIterDiscoveredChunks:
    def test_yields_all_files(self, tmp_path):
        root = _flat_tree(tmp_path, n=25)
        chunks = list(iter_discovered_chunks([str(root)], include_patterns=["*.nc"], depth=5, chunk_size=10))
        flat = [p for chunk in chunks for p in chunk]
        assert len(flat) == 25
        assert all(p.endswith(".nc") for p in flat)

    def test_chunk_size_respected_at_directory_boundaries(self, tmp_path):
        # 5 directories x 4 files = 20 files. chunk_size=8 forces splits between dirs.
        root = _wide_tree(tmp_path, num_dirs=5, files_per_dir=4)
        chunks = list(iter_discovered_chunks([str(root)], include_patterns=["*.nc"], depth=5, chunk_size=8))
        # Every chunk groups whole directories together (no directory split across chunks).
        for chunk in chunks:
            dirs = {str(Path(p).parent) for p in chunk}
            for d in dirs:
                files_in_dir = sum(1 for p in chunk if str(Path(p).parent) == d)
                total_in_tree = sum(1 for p in root.rglob("*.nc") if str(p.parent) == d)
                assert files_in_dir == total_in_tree, "directory was split across chunks"

        total = sum(len(c) for c in chunks)
        assert total == 20

    def test_empty_root_yields_nothing(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        chunks = list(iter_discovered_chunks([str(empty)], include_patterns=["*.nc"], depth=5, chunk_size=10))
        assert chunks == []

    def test_single_file_path(self, tmp_tree):
        nc_file = tmp_tree / "root.nc"
        chunks = list(
            iter_discovered_chunks([str(nc_file)], include_patterns=["*.nc"], depth=0, chunk_size=10)
        )
        assert chunks == [[str(nc_file)]]


class TestIterBuiltCatalogs:
    def test_streams_chunks(self, tmp_path):
        root = _wide_tree(tmp_path, num_dirs=4, files_per_dir=3)
        chunks = list(
            iter_built_catalogs(
                paths=[str(root)],
                parsing_func=_good_parser,
                include_patterns=["*.nc"],
                depth=5,
                n_jobs=1,
                chunk_size=5,
            )
        )
        # Each chunk is a DataFrame, total rows == total files.
        assert len(chunks) >= 2, "expected at least two chunks for chunk_size=5"
        total_rows = sum(len(df) for df in chunks)
        assert total_rows == 12
        for df in chunks:
            assert isinstance(df, pd.DataFrame)
            assert "path" in df.columns

    def test_invalid_rows_filtered_per_chunk(self, tmp_tree):
        with pytest.warns(UserWarning, match="Unable to parse"):
            chunks = list(
                iter_built_catalogs(
                    paths=[str(tmp_tree)],
                    parsing_func=_mixed_parser,
                    include_patterns=["*.nc", "*.txt"],
                    depth=10,
                    chunk_size=2,
                )
            )
        for df in chunks:
            assert "INVALID_ASSET" not in df.columns
            assert "TRACEBACK" not in df.columns
        total = sum(len(df) for df in chunks)
        # 4 .nc files survive, .txt files filtered as INVALID.
        assert total == 4

    def test_empty_input_yields_nothing(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        chunks = list(
            iter_built_catalogs(
                paths=[str(empty)],
                parsing_func=_good_parser,
                include_patterns=["*.nc"],
                depth=5,
                chunk_size=10,
            )
        )
        assert chunks == []

    def test_chunk_larger_than_total_yields_single_chunk(self, tmp_path):
        root = _flat_tree(tmp_path, n=3)
        chunks = list(
            iter_built_catalogs(
                paths=[str(root)],
                parsing_func=_good_parser,
                include_patterns=["*.nc"],
                depth=5,
                n_jobs=1,
                chunk_size=100,
            )
        )
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_all_invalid_chunk_skipped(self, tmp_tree):
        def _all_invalid(file: str, **kwargs: Any) -> dict[str, Any]:
            return {"INVALID_ASSET": file, "TRACEBACK": "always fails"}

        with pytest.warns(UserWarning, match="Unable to parse"):
            chunks = list(
                iter_built_catalogs(
                    paths=[str(tmp_tree)],
                    parsing_func=_all_invalid,
                    include_patterns=["*.nc"],
                    depth=10,
                    chunk_size=2,
                )
            )
        # Every chunk filters down to empty, so nothing is yielded.
        assert chunks == []

    def test_no_invalid_column_passes_through(self, tmp_path):
        """A parser that never emits INVALID_ASSET goes through the no-filter branch."""
        root = _flat_tree(tmp_path, n=4)
        chunks = list(
            iter_built_catalogs(
                paths=[str(root)],
                parsing_func=_good_parser,
                include_patterns=["*.nc"],
                depth=5,
                chunk_size=2,
            )
        )
        for df in chunks:
            assert "INVALID_ASSET" not in df.columns


class TestDiscoverEdgeCases:
    def test_iter_discovered_chunks_skips_nonexistent_paths(self, tmp_path):
        root = _flat_tree(tmp_path, n=3)
        chunks = list(
            iter_discovered_chunks(
                ["/does/not/exist", str(root)],
                include_patterns=["*.nc"],
                depth=5,
                chunk_size=10,
            )
        )
        flat = [p for chunk in chunks for p in chunk]
        assert len(flat) == 3

    def test_iter_discovered_chunks_directory_overflow_kept_together(self, tmp_path):
        """A single directory larger than chunk_size still flushes intact."""
        root = tmp_path / "big"
        root.mkdir()
        for i in range(6):
            (root / f"f_{i}.nc").touch()

        chunks = list(iter_discovered_chunks([str(root)], include_patterns=["*.nc"], depth=5, chunk_size=2))
        # All six files come from one directory; the buffer flushes once at the end.
        assert sum(len(c) for c in chunks) == 6
        for chunk in chunks:
            dirs = {str(Path(p).parent) for p in chunk}
            assert len(dirs) == 1

    def test_iter_discovered_chunks_multiple_roots(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()
        (a / "1.nc").touch()
        b = tmp_path / "b"
        b.mkdir()
        (b / "2.nc").touch()

        chunks = list(
            iter_discovered_chunks([str(a), str(b)], include_patterns=["*.nc"], depth=5, chunk_size=10)
        )
        flat = [p for chunk in chunks for p in chunk]
        assert len(flat) == 2
