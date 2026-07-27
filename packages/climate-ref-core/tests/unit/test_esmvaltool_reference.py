import pytest

from climate_ref_core.esmvaltool_reference import drs_relative_parts, tier_from_segment


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
