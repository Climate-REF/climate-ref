from pathlib import Path
from typing import ClassVar

import cftime
import numpy as np
import pandas as pd
import pytest

from climate_ref.datasets.utils import (
    build_instance_id,
    clean_branch_time,
    coerce_catalog_times,
    frequency_from_mip_table,
    parse_cftime_dates,
    parse_drs_daterange,
    validate_path,
)


class TestFrequencyFromMipTable:
    """``Amon`` -> ``mon``, for ESMValTool OBS/OBS6 reference data."""

    @pytest.mark.parametrize(
        "mip_table, expected",
        [
            # The common monthly tables across realms.
            ("Amon", "mon"),
            ("Omon", "mon"),
            ("Lmon", "mon"),
            ("LImon", "mon"),
            ("SImon", "mon"),
            ("AERmon", "mon"),
            ("Emon", "mon"),
            # Daily.
            ("day", "day"),
            ("Oday", "day"),
            ("SIday", "day"),
            ("CFday", "day"),
            # Sub-daily, including the point-sampled variants.
            ("3hr", "3hr"),
            ("E3hrPt", "3hrPt"),
            ("6hrLev", "6hr"),
            ("6hrPlevPt", "6hrPt"),
            ("AERhr", "1hr"),
            # Fixed fields and yearly.
            ("fx", "fx"),
            ("Ofx", "fx"),
            ("Oyr", "yr"),
            ("IyrGre", "yr"),
            # The irregular ones the enumeration exists for.
            ("Oclim", "monC"),
            ("E1hrClimMon", "1hrCM"),
            ("Esubhr", "subhrPt"),
            # Zonal-mean tables keep the frequency of their non-zonal counterpart.
            ("AERmonZ", "mon"),
            ("EdayZ", "day"),
            ("E6hrZ", "6hr"),
        ],
    )
    def test_maps_mip_table_to_frequency(self, mip_table, expected):
        assert frequency_from_mip_table(mip_table) == expected

    @pytest.mark.parametrize("frequency", ["mon", "day", "fx", "yr", "3hr", "subhrPt"])
    def test_existing_frequency_passes_through(self, frequency):
        """``native6`` paths already carry a frequency, so the same call site handles them."""
        assert frequency_from_mip_table(frequency) == frequency

    @pytest.mark.parametrize("value", ["day", "fx"])
    def test_names_that_are_both_table_and_frequency_are_idempotent(self, value):
        assert frequency_from_mip_table(value) == value
        assert frequency_from_mip_table(frequency_from_mip_table(value)) == value

    @pytest.mark.parametrize("value", ["Amonthly", "", "monthly", "Xmon", "AMON"])
    def test_unknown_value_raises(self, value):
        """Failing loudly beats defaulting: a silent fallback would collapse two datasets that
        differ only by frequency onto one ``instance_id``."""
        with pytest.raises(ValueError, match="Unknown MIP table or frequency"):
            frequency_from_mip_table(value)


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


class TestParseCftimeDates:
    def test_standard_dates(self):
        """Parses standard calendar dates to cftime.datetime objects."""
        result = parse_cftime_dates(pd.Series(["2021-01-01 00:00:00", "1850-01-17 00:29:59.999993", None]))
        assert result.iloc[0] == cftime.datetime(2021, 1, 1, 0, 0, calendar="standard")
        assert result.iloc[1] == cftime.datetime(1850, 1, 17, 0, 29, 59, 999993, calendar="standard")
        assert result.iloc[2] is None

    def test_date_only_format(self):
        """Parses dates without time component."""
        result = parse_cftime_dates(pd.Series(["2021-01-15"]))
        assert result.iloc[0] == cftime.datetime(2021, 1, 15, calendar="standard")

    def test_already_parsed_cftime(self):
        """Returns already-parsed cftime.datetime objects unchanged."""
        dt = cftime.datetime(2021, 6, 15, 12, 0, 0, calendar="360_day")
        result = parse_cftime_dates(pd.Series([dt]), calendar="360_day")
        assert result.iloc[0] == dt

    def test_unparseable_string_returns_none(self):
        """Unparseable strings return None."""
        result = parse_cftime_dates(pd.Series(["not-a-date"]))
        assert result.iloc[0] is None

    def test_nan_returns_none(self):
        """NaN values return None."""
        result = parse_cftime_dates(pd.Series([np.nan]))
        assert result.iloc[0] is None

    def test_preserves_series_index(self):
        """Output series preserves the original index."""
        s = pd.Series(["2021-01-01", "2022-06-15"], index=[10, 20])
        result = parse_cftime_dates(s)
        assert result.index.tolist() == [10, 20]

    def test_mixed_formats(self):
        """Handles a series with mixed date formats and null values."""
        result = parse_cftime_dates(
            pd.Series(
                [
                    "2021-01-01",
                    "2021-01-01 12:30:00",
                    "2021-01-01 12:30:00.500000",
                    None,
                ]
            )
        )
        assert result.iloc[0] == cftime.datetime(2021, 1, 1, calendar="standard")
        assert result.iloc[1] == cftime.datetime(2021, 1, 1, 12, 30, 0, calendar="standard")
        assert result.iloc[2] == cftime.datetime(2021, 1, 1, 12, 30, 0, 500000, calendar="standard")
        assert result.iloc[3] is None

    def test_empty_series(self):
        """Empty series returns empty series."""
        result = parse_cftime_dates(pd.Series([], dtype=object))
        assert result.empty

    def test_360_day_calendar(self):
        """Parses dates with 360_day calendar (Feb 30 is valid)."""
        result = parse_cftime_dates(pd.Series(["2000-02-30"]), calendar="360_day")
        expected = cftime.datetime(2000, 2, 30, calendar="360_day")
        assert result.iloc[0] == expected

    def test_noleap_calendar(self):
        """Parses dates with noleap calendar."""
        result = parse_cftime_dates(pd.Series(["2000-03-01"]), calendar="noleap")
        expected = cftime.datetime(2000, 3, 1, calendar="noleap")
        assert result.iloc[0] == expected

    def test_per_row_calendar_series(self):
        """Handles per-row calendar values via a Series."""
        result = parse_cftime_dates(
            pd.Series(["2000-01-15", "2000-01-15"]),
            calendar=pd.Series(["standard", "360_day"]),
        )
        assert result.iloc[0] == cftime.datetime(2000, 1, 15, calendar="standard")
        assert result.iloc[1] == cftime.datetime(2000, 1, 15, calendar="360_day")


class TestBuildInstanceId:
    drs_items: ClassVar[list[str]] = ["activity_id", "variable_id", "version"]

    def _frame(self, rows):
        return pd.DataFrame(rows)

    def test_all_components_present(self):
        df = self._frame(
            [
                {"activity_id": "CMIP", "variable_id": "tas", "version": "v1", "path": "/a.nc"},
                {"activity_id": "CMIP", "variable_id": "pr", "version": "v1", "path": "/b.nc"},
            ]
        )
        out = build_instance_id(df, self.drs_items, prefix="CMIP6")
        assert out["instance_id"].tolist() == ["CMIP6.CMIP.tas.v1", "CMIP6.CMIP.pr.v1"]
        assert len(out) == 2

    def test_drops_rows_with_none_component(self, caplog):
        df = self._frame(
            [
                {"activity_id": "CMIP", "variable_id": "tas", "version": "v1", "path": "/a.nc"},
                {"activity_id": "CMIP", "variable_id": None, "version": "v1", "path": "/bad.nc"},
                {"activity_id": "CMIP", "variable_id": "pr", "version": np.nan, "path": "/bad2.nc"},
            ]
        )
        out = build_instance_id(df, self.drs_items, prefix="CMIP6")

        assert out["instance_id"].tolist() == ["CMIP6.CMIP.tas.v1"]
        # Two warnings, one per dropped row, naming the path + missing column(s).
        warning_messages = [r.message for r in caplog.records]
        assert any("/bad.nc" in m and "variable_id" in m for m in warning_messages)
        assert any("/bad2.nc" in m and "version" in m for m in warning_messages)

    def test_empty_input(self):
        df = pd.DataFrame(columns=["activity_id", "variable_id", "version", "path"])
        out = build_instance_id(df, self.drs_items, prefix="CMIP6")
        assert out.empty
        assert "instance_id" in out.columns

    def test_custom_transform(self):
        df = self._frame(
            [
                {
                    "activity_id": "obs4MIPs",
                    "variable_id": "tas",
                    "nominal_resolution": "100 km",
                    "version": "v1",
                    "path": "/a.nc",
                }
            ]
        )
        out = build_instance_id(
            df,
            ["activity_id", "nominal_resolution", "variable_id", "version"],
            prefix="obs4MIPs",
            transform=lambda item, value: (
                str(value).replace(" ", "") if item == "nominal_resolution" else str(value)
            ),
        )
        assert out["instance_id"].iloc[0] == "obs4MIPs.obs4MIPs.100km.tas.v1"

    def test_does_not_mutate_input(self):
        df = self._frame(
            [
                {"activity_id": "CMIP", "variable_id": "tas", "version": "v1", "path": "/a.nc"},
            ]
        )
        original = df.copy()
        build_instance_id(df, self.drs_items, prefix="CMIP6")
        pd.testing.assert_frame_equal(df, original)


class TestCoerceCatalogTimes:
    def test_no_time_columns_is_noop(self):
        catalog = pd.DataFrame({"instance_id": ["a"]})
        result = coerce_catalog_times(catalog)
        assert "start_time" not in result.columns
        assert "end_time" not in result.columns

    def test_both_columns_present(self):
        catalog = pd.DataFrame({"start_time": ["2000-01-01"], "end_time": ["2000-12-31"]})
        result = coerce_catalog_times(catalog)
        assert isinstance(result["start_time"].iloc[0], cftime.datetime)
        assert isinstance(result["end_time"].iloc[0], cftime.datetime)

    def test_start_time_without_end_time_does_not_raise(self):
        """A catalog with only ``start_time`` must not KeyError looking up ``end_time``."""
        catalog = pd.DataFrame({"start_time": ["2000-01-01"]})
        result = coerce_catalog_times(catalog)
        assert isinstance(result["start_time"].iloc[0], cftime.datetime)
        assert "end_time" not in result.columns

    def test_end_time_without_start_time_does_not_raise(self):
        catalog = pd.DataFrame({"end_time": ["2000-12-31"]})
        result = coerce_catalog_times(catalog)
        assert isinstance(result["end_time"].iloc[0], cftime.datetime)
        assert "start_time" not in result.columns
