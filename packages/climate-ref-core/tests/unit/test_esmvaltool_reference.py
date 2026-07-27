import pytest

from climate_ref_core.esmvaltool_reference import (
    drs_relative_parts,
    frequency_from_mip_table,
    parse_reference_path,
    tier_from_segment,
)


@pytest.mark.parametrize(
    "segment, expected",
    [("Tier2", 2), ("Tier10", 10), ("CERES-EBAF", None), ("Tier", None), ("TierX", None)],
)
def test_tier_from_segment(segment, expected):
    assert tier_from_segment(segment) == expected


@pytest.mark.parametrize(
    "path",
    [
        # OBS tolerates extra directories between the dataset and the file.
        "/data/OBS/Tier2/CERES-EBAF/sub/x.nc",
        "/data/native6/Tier3/ERA5/v1/mon/tas/x.nc",
        "/data/obs4MIPs/GPCP-V2.3/v20180519/x.nc",
    ],
)
def test_drs_relative_parts_accepts_each_project_layout(path):
    assert drs_relative_parts(path)[0] in ("OBS", "native6", "obs4MIPs")


@pytest.mark.parametrize(
    "path",
    [
        # No tier directory under a tiered project.
        "/data/OBS/CERES-EBAF/x.nc",
        # No dataset directory, which would otherwise read the filename as the dataset.
        "/data/OBS/Tier2/OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc",
        # native6 and obs4MIPs are fixed depth, so neither may carry extra directories.
        "/data/native6/Tier3/ERA5/v1/mon/tas/sub/x.nc",
        "/data/obs4MIPs/GPCP-V2.3/x.nc",
        # A tier directory under an untiered project.
        "/data/obs4MIPs/Tier2/GPCP-V2.3/x.nc",
    ],
)
def test_drs_relative_parts_rejects_a_path_that_fits_no_layout(path):
    with pytest.raises(ValueError, match=r"unexpected \w+ path structure"):
        drs_relative_parts(path)


@pytest.mark.parametrize(
    "path, expected",
    [
        (
            "/data/ESMValTool/OBS/Tier2/CERES-EBAF/OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc",
            ("OBS", "Tier2", "CERES-EBAF", "OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc"),
        ),
        # OBS6 data lives under the OBS directory, so its anchor is OBS too.
        (
            "/data/ESMValTool/OBS/Tier2/TROPFLUX/OBS6_TROPFLUX_reanaly_v1_Omon_tos_197901-201812.nc",
            ("OBS", "Tier2", "TROPFLUX", "OBS6_TROPFLUX_reanaly_v1_Omon_tos_197901-201812.nc"),
        ),
        (
            "/data/ESMValTool/native6/Tier3/ERA5/v1/mon/tas/era5_tas_1980_monthly.nc",
            ("native6", "Tier3", "ERA5", "v1", "mon", "tas", "era5_tas_1980_monthly.nc"),
        ),
        (
            "/data/ESMValTool/obs4MIPs/GPCP-V2.3/v20180519/pr_mon_GPCP-V2-3_gn_200001-200012.nc",
            ("obs4MIPs", "GPCP-V2.3", "v20180519", "pr_mon_GPCP-V2-3_gn_200001-200012.nc"),
        ),
        # A relative path is handled the same way.
        (
            "OBS/Tier2/CERES-EBAF/OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc",
            ("OBS", "Tier2", "CERES-EBAF", "OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc"),
        ),
    ],
)
def test_drs_relative_parts(path, expected):
    assert drs_relative_parts(path) == expected


def test_drs_relative_parts_ignores_an_anchor_in_the_data_root():
    # A data root containing a directory named after a project must not truncate the
    # path at the root instead of at the real anchor.
    path = "/data/OBS/store/native6/Tier3/ERA5/v1/mon/tas/era5_tas_1980_monthly.nc"

    assert drs_relative_parts(path)[0] == "native6"


def test_drs_relative_parts_ignores_an_anchor_inside_the_tree():
    # A dataset directory named after another project must not win over the real anchor,
    # which would dispatch the file to the wrong parser and silently mis-read its metadata.
    path = "/data/ESMValTool/OBS/Tier2/obs4MIPs/OBS_obs4MIPs_sat_Ed4.2_Amon_rlut_200003-202311.nc"

    assert drs_relative_parts(path)[0] == "OBS"


def test_drs_relative_parts_ignores_an_untiered_anchor_inside_a_tiered_tree():
    # obs4MIPs has no TierN directory, so a dataset directory of that name inside a
    # native6 tree is long enough to look plausible on length alone.
    path = "/data/ESMValTool/native6/Tier3/obs4MIPs/v1/mon/tas/era5_tas_1980_monthly.nc"

    assert drs_relative_parts(path)[0] == "native6"


def test_drs_relative_parts_names_the_project_it_could_not_fit():
    # The message points at the leftmost candidate, which is the project the caller meant.
    with pytest.raises(ValueError, match="unexpected OBS path structure"):
        drs_relative_parts("/data/OBS/odd.nc")


def test_drs_relative_parts_rejects_a_path_with_no_anchor():
    with pytest.raises(ValueError, match="not under a known ESMValTool reference project"):
        drs_relative_parts("/data/somewhere/mystery.nc")


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
            # CMIP5-era sea-ice tables (used by ESMValTool OBS data, e.g. OSI-450 sic).
            ("OImon", "mon"),
            ("OIday", "day"),
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


class TestParseReferencePath:
    def test_obs(self):
        facets = parse_reference_path(
            "/data/ESMValTool/OBS/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_OImon_sic_197901-197912.nc"
        )

        assert facets.project == "OBS"
        assert facets.source_id == "OSI-450-nh"
        assert facets.variable_id == "sic"
        assert facets.frequency == "mon"
        assert facets.version == "v3"
        assert facets.data_type == "reanaly"
        assert facets.tier == 2
        assert facets.timerange == "197901-197912"

    def test_obs6_keeps_its_own_project(self):
        """OBS6 data lives under the OBS directory, so only the filename distinguishes it."""
        facets = parse_reference_path(
            "/data/ESMValTool/OBS/Tier2/TROPFLUX/OBS6_TROPFLUX_reanaly_v1_Omon_tos_197901-201812.nc"
        )

        assert facets.project == "OBS6"

    def test_obs_without_a_timerange(self):
        """A fixed field carries no date range, which is how a supplementary is spelled."""
        facets = parse_reference_path(
            "/data/ESMValTool/OBS/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_fx_areacello.nc"
        )

        assert facets.variable_id == "areacello"
        assert facets.frequency == "fx"
        assert facets.timerange is None

    def test_native6(self):
        facets = parse_reference_path(
            "/data/ESMValTool/native6/Tier3/ERA5/v1/mon/tas/era5_tas_1980_monthly.nc"
        )

        assert facets.project == "native6"
        assert facets.source_id == "ERA5"
        assert facets.variable_id == "tas"
        assert facets.frequency == "mon"
        assert facets.version == "v1"
        assert facets.data_type is None
        assert facets.tier == 3
        # The filename is raw, so it carries no reliable date range.
        assert facets.timerange is None

    def test_obs4mips(self):
        facets = parse_reference_path(
            "/data/ESMValTool/obs4MIPs/GPCP-V2.3/v20180519/pr_mon_GPCP-V2-3_gn_200001-200012.nc"
        )

        assert facets.project == "obs4MIPs"
        assert facets.source_id == "GPCP-V2.3"
        assert facets.variable_id == "pr"
        assert facets.frequency == "mon"
        assert facets.version == "v20180519"
        assert facets.tier is None
        assert facets.timerange == "200001-200012"

    def test_obs_filename_missing_fields_raises(self):
        with pytest.raises(ValueError, match="unexpected OBS filename structure"):
            parse_reference_path("/data/ESMValTool/OBS/Tier2/CERES-EBAF/OBS_CERES-EBAF_sat.nc")

    def test_obs4mips_filename_without_a_variable_raises(self):
        with pytest.raises(ValueError, match="unexpected obs4MIPs filename structure"):
            parse_reference_path("/data/ESMValTool/obs4MIPs/GPCP-V2.3/v20180519/_mon_gn.nc")
