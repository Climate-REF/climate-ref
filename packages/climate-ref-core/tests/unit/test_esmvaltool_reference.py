import pytest

from climate_ref_core.esmvaltool_reference import drs_relative_parts


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


def test_drs_relative_parts_falls_back_to_the_leftmost_anchor():
    # Nothing structurally fits, so the leftmost candidate is used and the caller's
    # own parser reports what is wrong with it.
    assert drs_relative_parts("/data/OBS/odd.nc") == ("OBS", "odd.nc")


def test_drs_relative_parts_rejects_a_path_with_no_anchor():
    with pytest.raises(ValueError, match="not under a known ESMValTool reference project"):
        drs_relative_parts("/data/somewhere/mystery.nc")
