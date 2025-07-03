from pathlib import Path

import pandas as pd
import pytest
from climate_ref_esmvaltool.recipe import prepare_climate_data


@pytest.mark.parametrize(
    ("datasets", "expected"),
    [
        (
            pd.DataFrame(
                {
                    "instance_id": [
                        "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp126.r1i1p1f1.Amon.pr.gn.v20190429",
                    ],
                    "source_id": ["CanESM5"],
                    "path": [
                        "pr_Amon_CanESM5_ssp126_r1i1p1f1_gn_210101-230012.nc",
                    ],
                }
            ),
            [
                "CMIP6/ScenarioMIP/CCCma/CanESM5/ssp126/r1i1p1f1/Amon/pr/gn/v20190429/pr_Amon_CanESM5_ssp126_r1i1p1f1_gn_210101-230012.nc",
            ],
        ),
        (
            pd.DataFrame(
                {
                    "instance_id": [
                        "obs4MIPs.obs4MIPs.ECMWF.ERA-5.ta.gn.v20250220",
                        "obs4MIPs.obs4MIPs.ECMWF.ERA-5.ta.gn.v20250220",
                    ],
                    "source_id": ["ERA-5", "ERA-5"],
                    "path": [
                        "ta_mon_ERA-5_PCMDI_gn_200701-200712.nc",
                        "ta_mon_ERA-5_PCMDI_gn_200801-200812.nc",
                    ],
                }
            ),
            [
                "obs4MIPs/ERA-5/v20250220/ta_mon_ERA-5_PCMDI_gn_200701-200712.nc",
                "obs4MIPs/ERA-5/v20250220/ta_mon_ERA-5_PCMDI_gn_200801-200812.nc",
            ],
        ),
    ],
)
def test_prepare_climate_data(tmp_path, datasets, expected):
    climate_data_dir = tmp_path / "climate_data"
    climate_data_dir.mkdir()

    source_data_dir = tmp_path / "source_data"
    source_data_dir.mkdir()

    datasets["path"] = [f"{source_data_dir / path}" for path in datasets["path"]]
    expected = [f"{climate_data_dir / path}" for path in expected]
    for path in datasets["path"]:
        Path(path).touch()

    prepare_climate_data(datasets, climate_data_dir)

    for source_path, symlink in zip(datasets["path"], expected):
        assert Path(symlink).is_symlink()
        assert Path(symlink).resolve() == Path(source_path).resolve()
