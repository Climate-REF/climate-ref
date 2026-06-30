"""Tests for the path-safety primitive in :mod:`climate_ref_core.paths`."""

from pathlib import Path

import pytest

from climate_ref_core.paths import safe_path


class TestLexicalLayer:
    """The lexical layer runs with or without a base directory."""

    @pytest.mark.parametrize(
        "relpath",
        ["a.nc", "sub/dir/file.png", "..hidden/file.txt", "a/b/c.json"],
    )
    def test_accepts_contained_relpaths(self, relpath):
        assert safe_path(relpath) == Path(relpath)

    @pytest.mark.parametrize(
        "relpath",
        ["", "/abs/path.nc", "../escape.nc", "../../etc/passwd", "sub/../../escape.nc", "a\x00b"],
    )
    def test_rejects_escaping_relpaths(self, relpath):
        with pytest.raises(ValueError, match="contained relative path"):
            safe_path(relpath)


class TestSingleSegment:
    """``single_segment`` requires a lone path component (e.g. a slug)."""

    @pytest.mark.parametrize("relpath", ["esmvaltool", "my-diag", "foo_bar", "..hidden"])
    def test_accepts_single_segments(self, relpath):
        assert safe_path(relpath, single_segment=True) == Path(relpath)

    @pytest.mark.parametrize("relpath", ["sub/dir", "a/b/c.json", "foo\\bar"])
    def test_rejects_separators(self, relpath):
        with pytest.raises(ValueError, match="single path segment"):
            safe_path(relpath, single_segment=True)

    @pytest.mark.parametrize("relpath", ["", "/abs", "../escape", "."])
    def test_lexical_layer_still_applies(self, relpath):
        with pytest.raises(ValueError, match="contained relative path"):
            safe_path(relpath, single_segment=True)


class TestContainmentLayer:
    """When a base is supplied the join is resolved and confirmed within base."""

    def test_returns_joined_path_within_base(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        assert safe_path("subdir/file.txt", base) == base / "subdir" / "file.txt"

    @pytest.mark.parametrize("relpath", ["../escape.nc", "/abs/path.nc"])
    def test_rejects_paths_escaping_base(self, tmp_path, relpath):
        base = tmp_path / "base"
        base.mkdir()
        with pytest.raises(ValueError, match=r"contained relative path|escapes"):
            safe_path(relpath, base)

    def test_symlink_escaping_base_raises(self, tmp_path):
        """A lexically-contained path that resolves through a symlink out of base is rejected."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        # 'link' is a contained name, but resolves outside base.
        (base / "link").symlink_to(outside)
        with pytest.raises(ValueError, match="escapes"):
            safe_path("link/file.txt", base)
