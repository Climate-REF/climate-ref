import pytest

from climate_ref_core.esmvaltool_reference import PROJECT_ANCHORS, relative_parts


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
            "/data/ESMValTool/obs4MIPs/GPCP-V2-3/v20250101/pr_mon_GPCP-V2-3_gn_200001-200012.nc",
            ("obs4MIPs", "GPCP-V2-3", "v20250101", "pr_mon_GPCP-V2-3_gn_200001-200012.nc"),
        ),
        # A relative path is handled the same way.
        (
            "OBS/Tier2/CERES-EBAF/OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc",
            ("OBS", "Tier2", "CERES-EBAF", "OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc"),
        ),
    ],
)
def test_relative_parts(path, expected):
    assert relative_parts(path) == expected


def test_relative_parts_anchors_on_the_last_project_directory():
    # A data root containing a directory named after a project must not truncate the
    # path at the root instead of at the real anchor.
    path = "/data/OBS/store/native6/Tier3/ERA5/v1/mon/tas/era5_tas_1980_monthly.nc"

    assert relative_parts(path)[0] == "native6"


def test_relative_parts_rejects_a_path_with_no_anchor():
    with pytest.raises(ValueError, match="not under a known ESMValTool reference project"):
        relative_parts("/data/somewhere/mystery.nc")


def test_project_anchors_are_the_documented_set():
    assert PROJECT_ANCHORS == ("OBS", "native6", "obs4MIPs")
