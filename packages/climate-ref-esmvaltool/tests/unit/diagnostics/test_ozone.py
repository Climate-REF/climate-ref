import pandas as pd
import pytest
from climate_ref_esmvaltool import provider

from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.exceptions import InvalidDiagnosticException


def _cmip7_o3_catalog(
    *, end: str = "2014-12-31", branded_variable: str = "o3_tavg-p19-hxy-air"
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instance_id": (
                    "CMIP7.CMIP.NOAA-GFDL.GFDL-ESM4.historical.r1i1p1f1."
                    "glb.mon.o3.tavg-p19-hxy-air.gr1.v20190726"
                ),
                "source_id": "GFDL-ESM4",
                "experiment_id": "historical",
                "variant_label": "r1i1p1f1",
                "grid_label": "gr1",
                "variable_id": "o3",
                "branded_variable": branded_variable,
                "frequency": "mon",
                "region": "glb",
                "start_time": pd.Timestamp("2000-01-01"),
                "end_time": pd.Timestamp(end),
                "path": "o3.nc",
            },
        ]
    )


def _ozone_zonal():
    return next(diagnostic for diagnostic in provider.diagnostics() if diagnostic.slug == "ozone-zonal")


def test_ozone_zonal_solves_with_cmip7_o3() -> None:
    """Accept CMIP7 pressure-level ozone covering the diagnostic's averaging period."""
    diagnostic = _ozone_zonal()
    assert diagnostic.version == 5
    assert [requirements[0].source_type for requirements in diagnostic.data_requirements] == [
        SourceDatasetType.CMIP6,
        SourceDatasetType.CMIP7,
    ]

    executions = list(
        solve_executions(
            {SourceDatasetType.CMIP7: _cmip7_o3_catalog()},
            diagnostic,
            diagnostic.provider,
        )
    )

    assert len(executions) == 1
    datasets = executions[0].datasets[SourceDatasetType.CMIP7].datasets
    assert datasets["branded_variable"].tolist() == ["o3_tavg-p19-hxy-air"]


def test_ozone_zonal_rejects_cmip7_o3_without_full_period() -> None:
    """Require CMIP7 ozone through December 2014, as the recipe does."""
    diagnostic = _ozone_zonal()

    with pytest.raises(InvalidDiagnosticException, match="No data catalog matches"):
        list(
            solve_executions(
                {SourceDatasetType.CMIP7: _cmip7_o3_catalog(end="2014-11-30")},
                diagnostic,
                diagnostic.provider,
            )
        )


def test_ozone_zonal_rejects_cmip7_model_level_o3() -> None:
    """Require pressure-level ozone because the recipe extracts air-pressure levels."""
    diagnostic = _ozone_zonal()

    with pytest.raises(InvalidDiagnosticException, match="No data catalog matches"):
        list(
            solve_executions(
                {SourceDatasetType.CMIP7: _cmip7_o3_catalog(branded_variable="o3_tavg-al-hxy-u")},
                diagnostic,
                diagnostic.provider,
            )
        )
