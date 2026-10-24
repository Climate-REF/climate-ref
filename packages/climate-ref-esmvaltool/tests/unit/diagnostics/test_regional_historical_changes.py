import numpy as np
import pandas as pd
import xarray as xr
from climate_ref_esmvaltool.diagnostics.regional_historical_changes import RegionalHistoricalTrend

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


def _write_trends(result_dir, variable_id, datasets, values):
    plot_dir = result_dir / "work" / f"{variable_id}_trends" / "plot"
    plot_dir.mkdir(parents=True)
    xr.Dataset(
        {variable_id: ("dim0", np.array(values, dtype=np.float32))},
        coords={
            "dataset": ("dim0", np.array([name.encode() for name in datasets for _ in range(2)])),
            "shape_id": ("dim0", np.array([b"NWN ", b"NEN "] * len(datasets))),
        },
    ).to_netcdf(plot_dir / "seaborn_barplot.nc")


def _format_result(result_dir, source_id):
    execution_dataset = ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                pd.DataFrame({"instance_id": ["a"], "source_id": [source_id]}), "instance_id"
            )
        }
    )
    metric, _ = RegionalHistoricalTrend.format_result(
        result_dir, execution_dataset, CMECMetric.create_template(), CMECOutput.create_template()
    )
    return metric


def test_trend_format_result_keeps_only_the_execution_dataset(tmp_path):
    """Report the execution's trends, not those of the other datasets in the same file."""
    _write_trends(tmp_path, "tas", ["ERA-5", "HadCRUT5"], [1.0, 2.0, 10.0, 20.0])

    metric = _format_result(tmp_path, "ERA-5")

    assert metric.RESULTS["tas"] == {"NWN": {"trend": 1.0}, "NEN": {"trend": 2.0}}


def test_trend_format_result_skips_variables_without_the_execution_dataset(tmp_path):
    """Leave out a variable whose file only holds other reference datasets."""
    _write_trends(tmp_path, "tas", ["ERA-5"], [1.0, 2.0])
    _write_trends(tmp_path, "pr", ["ERA5", "GPCP-V2.3"], [3.0, 4.0, 5.0, 6.0])

    metric = _format_result(tmp_path, "ERA-5")

    assert set(metric.RESULTS) == {"tas"}
    assert set(metric.DIMENSIONS.root["variable_id"]) == {"tas"}
