from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from climate_ref.datasets.catalog_builder import build_catalog, discover_files


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
