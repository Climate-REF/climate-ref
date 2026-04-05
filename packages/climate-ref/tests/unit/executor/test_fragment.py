"""Tests for the allocate_output_fragment helper."""

import datetime
from unittest.mock import patch

import pytest

from climate_ref.executor.fragment import allocate_output_fragment


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
