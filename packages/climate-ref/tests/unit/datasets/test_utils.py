from pathlib import Path

import pytest

from climate_ref.datasets.utils import parse_drs_daterange, validate_path


@pytest.mark.parametrize(
    "raw, expected",
    [
        [
            "/data/file.csv",
            Path("/data/file.csv"),
        ]
    ],
)
def test_validate_prefix(raw, expected, mocker):
    mocker.patch.object(Path, "exists", return_value=True)

    assert validate_path(raw) == expected


def test_validate_prefix_with_relative_path(mocker):
    mocker.patch.object(Path, "exists", return_value=True)
    raw_path = "data/subfolder/file.csv"

    with pytest.raises(ValueError):
        validate_path(raw_path)


def test_validate_prefix_missing(mocker):
    mocker.patch.object(Path, "exists", return_value=False)

    raw_path = "/other_dir/file.csv"
    with pytest.raises(FileNotFoundError):
        validate_path(raw_path)


class TestParseDaterange:
    """Tests for parse_drs_daterange covering all CMIP6 filename date formats."""

    def test_monthly_format(self):
        """YYYYMM-YYYYMM (6 chars) produces approximate start/end dates."""
        start, end = parse_drs_daterange("185001-201412")
        assert start == "1850-01-01"
        assert end == "2014-12-30"

    def test_daily_format(self):
        """YYYYMMDD-YYYYMMDD (8 chars) produces exact start/end dates."""
        start, end = parse_drs_daterange("20100315-20101231")
        assert start == "2010-03-15"
        assert end == "2010-12-31"

    def test_subdaily_format(self):
        """YYYYMMDDhhmm-YYYYMMDDhhmm (12 chars) produces date-only start/end."""
        start, end = parse_drs_daterange("201501011030-201512312330")
        assert start == "2015-01-01"
        assert end == "2015-12-31"

    def test_mismatched_lengths_returns_none(self):
        """Mismatched date component lengths return (None, None)."""
        start, end = parse_drs_daterange("185001-20141201")
        assert start is None
        assert end is None

    def test_yearly_format(self):
        """YYYY-YYYY (4 chars) produces approximate start/end dates."""
        start, end = parse_drs_daterange("1850-2014")
        assert start == "1850-01-01"
        assert end == "2014-12-30"

    def test_unsupported_length_returns_none(self):
        """Unsupported date component lengths (e.g. 5 chars) return (None, None)."""
        start, end = parse_drs_daterange("18500-20145")
        assert start is None
        assert end is None

    def test_no_hyphen_returns_none(self):
        """Input without a hyphen separator returns (None, None)."""
        start, end = parse_drs_daterange("185001201412")
        assert start is None
        assert end is None

    def test_empty_string_returns_none(self):
        """Empty string returns (None, None)."""
        start, end = parse_drs_daterange("")
        assert start is None
        assert end is None
