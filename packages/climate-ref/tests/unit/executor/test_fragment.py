"""Tests for the fragment allocation and group-short helpers."""

import datetime
from unittest.mock import patch

import pytest

from climate_ref.executor.fragment import allocate_output_fragment, compute_group_short


class TestAllocateOutputFragment:
    def test_appends_timestamp_suffix(self, tmp_path):
        """Should append a UTC timestamp suffix to the base fragment."""
        result = allocate_output_fragment("provider/diag/abc123", tmp_path)
        assert result.startswith("provider/diag/abc123_")
        # Suffix should be a valid timestamp: YYYYMMDDTHHMMSS followed by 6 microsecond digits
        suffix = result.split("_", 1)[1]
        assert len(suffix) == 21  # 8 date + T + 6 time + 6 microseconds
        assert "T" in suffix

    def test_preserves_base_fragment(self, tmp_path):
        """The original fragment should be a prefix of the result."""
        base = "my_provider/my_diag/hash123"
        result = allocate_output_fragment(base, tmp_path)
        assert result.startswith(base + "_")

    def test_different_calls_produce_different_fragments(self, tmp_path):
        """Two rapid calls should produce different fragments (microsecond resolution)."""
        result1 = allocate_output_fragment("provider/diag/abc123", tmp_path)
        result2 = allocate_output_fragment("provider/diag/abc123", tmp_path)
        assert result1 != result2

    def test_raises_if_directory_already_exists(self, tmp_path):
        """Should raise FileExistsError when the target directory already exists."""
        fixed_time = datetime.datetime(2026, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
        with patch("climate_ref.executor.fragment.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fixed_time
            mock_dt.timezone = datetime.timezone
            # First call succeeds
            fragment = allocate_output_fragment("provider/diag/abc123", tmp_path)
            # Create the directory so a second call with the same timestamp collides
            (tmp_path / fragment).mkdir(parents=True)
            with pytest.raises(FileExistsError, match="Output directory already exists"):
                allocate_output_fragment("provider/diag/abc123", tmp_path)


class TestComputeGroupShort:
    """Tests for the ``compute_group_short`` helper."""

    def test_compute_group_short_is_deterministic(self):
        """Same inputs should always produce the same output."""
        selectors = {"cmip6": [("source_id", "ACCESS-ESM1-5"), ("variable_id", "tas")]}
        out1 = compute_group_short(selectors, group_id=7, diagnostic_version=1)
        out2 = compute_group_short(selectors, group_id=7, diagnostic_version=1)
        assert out1 == out2

    def test_compute_group_short_includes_group_id_and_version(self):
        """The result should include human-readable ``g{id}`` and ``v{version}`` markers."""
        selectors = {"cmip6": [("source_id", "MODEL")]}
        out = compute_group_short(selectors, group_id=42, diagnostic_version=2)
        assert "g42" in out
        assert "v2" in out
        # Should also be ASCII-only
        assert out.isascii()

    def test_compute_group_short_truncation(self):
        """Selector strings longer than the token limit should be truncated cleanly."""
        # Build selectors whose joined values are well over 100 characters.
        long_value = "X" * 30
        selectors = {
            "cmip6": [(f"facet_{i}", f"{long_value}_{i}") for i in range(6)],
        }
        out = compute_group_short(selectors, group_id=1, diagnostic_version=1)
        # Whole result is capped at ~96 chars; suffix is preserved.
        assert len(out) <= 96
        assert out.endswith("_g1_v1_" + out.split("_")[-1])
        # Truncation should not leave a stray boundary character.
        assert "g1" in out and "v1" in out

    def test_compute_group_short_collision_resistance(self):
        """Two selector sets sharing a prefix should yield distinct hash suffixes."""
        # Both selector sets start with the same value but differ further on.
        a = {"cmip6": [("source_id", "MODEL"), ("variable_id", "tas")]}
        b = {"cmip6": [("source_id", "MODEL"), ("variable_id", "pr")]}
        out_a = compute_group_short(a, group_id=1, diagnostic_version=1)
        out_b = compute_group_short(b, group_id=1, diagnostic_version=1)
        assert out_a != out_b
        # The 8-char hash digest is the trailing ``_xxxxxxxx`` segment.
        digest_a = out_a.rsplit("_", 1)[-1]
        digest_b = out_b.rsplit("_", 1)[-1]
        assert digest_a != digest_b
        assert len(digest_a) == 8
        assert len(digest_b) == 8

    def test_compute_group_short_handles_unicode_selector_values(self):
        """Non-ASCII selector values should be sanitized to ASCII tokens."""
        selectors = {"cmip6": [("source_id", "MODéL")]}
        out = compute_group_short(selectors, group_id=1, diagnostic_version=1)
        assert out.isascii()

    def test_compute_group_short_empty_selectors(self):
        """An empty selector mapping still produces a valid suffix."""
        out = compute_group_short({}, group_id=3, diagnostic_version=1)
        assert "g3" in out
        assert "v1" in out
        assert out.isascii()

    def test_compute_group_short_hard_cap_enforced(self):
        """Result must never exceed _GROUP_SHORT_MAX even with a huge group_id and long selectors."""
        long_value = "X" * 30
        selectors = {
            "cmip6": [(f"facet_{i}", f"{long_value}_{i}") for i in range(10)],
        }
        out = compute_group_short(selectors, group_id=10**9, diagnostic_version=1)
        assert len(out) <= 96, f"Expected len <= 96, got {len(out)}: {out!r}"
