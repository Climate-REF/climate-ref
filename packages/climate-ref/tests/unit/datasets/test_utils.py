import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_ref.datasets.utils import clean_branch_time, parse_datetime, parse_drs_daterange, validate_path


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


def test_clean_branch_time():
    inp = pd.Series(["0D", "12", "12.0", "12.000", "12.0000", "12.00000", None, np.nan])
    exp = pd.Series([0.0, 12.0, 12.0, 12.0, 12.0, 12.0, np.nan, np.nan])

    pd.testing.assert_series_equal(clean_branch_time(inp), exp)


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


class TestParseDatetime:
    def test_parse_datetime_with_various_formats(self):
        pd.testing.assert_series_equal(
            parse_datetime(pd.Series(["2021-01-01 00:00:00", "1850-01-17 00:29:59.999993", None])),
            pd.Series(
                [
                    datetime.datetime(2021, 1, 1, 0, 0),
                    datetime.datetime(1850, 1, 17, 0, 29, 59, 999993),
                    None,
                ],
                dtype="object",
            ),
        )

    def test_date_only_format(self):
        """Parses dates without time component."""
        result = parse_datetime(pd.Series(["2021-01-15"]))
        assert result.iloc[0] == datetime.datetime(2021, 1, 15)

    def test_already_parsed_datetime(self):
        """Returns already-parsed datetime objects unchanged."""
        dt = datetime.datetime(2021, 6, 15, 12, 0, 0)
        result = parse_datetime(pd.Series([dt]))
        assert result.iloc[0] == dt

    def test_unparseable_string_returns_none(self):
        """Unparseable strings return None."""
        result = parse_datetime(pd.Series(["not-a-date"]))
        assert result.iloc[0] is None

    def test_nan_returns_none(self):
        """NaN values return None."""
        result = parse_datetime(pd.Series([np.nan]))
        assert result.iloc[0] is None

    def test_preserves_series_index(self):
        """Output series preserves the original index."""
        s = pd.Series(["2021-01-01", "2022-06-15"], index=[10, 20])
        result = parse_datetime(s)
        assert result.index.tolist() == [10, 20]

    def test_mixed_formats(self):
        """Handles a series with mixed date formats and null values."""
        result = parse_datetime(
            pd.Series(
                [
                    "2021-01-01",
                    "2021-01-01 12:30:00",
                    "2021-01-01 12:30:00.500000",
                    None,
                ]
            )
        )
        assert result.iloc[0] == datetime.datetime(2021, 1, 1)
        assert result.iloc[1] == datetime.datetime(2021, 1, 1, 12, 30, 0)
        assert result.iloc[2] == datetime.datetime(2021, 1, 1, 12, 30, 0, 500000)
        assert result.iloc[3] is None

    def test_empty_series(self):
        """Empty series returns empty series."""
        result = parse_datetime(pd.Series([], dtype=object))
        assert result.empty
