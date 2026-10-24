import numpy as np
import pandas as pd
import xarray as xr
from climate_ref_esmvaltool.diagnostics.regional_historical_changes import RegionalHistoricalTrend

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


def test_trend_format_result_keeps_only_the_model(tmp_path):
    """Report the model's trends, not those of the reference datasets in the same file."""
    plot_dir = tmp_path / "work" / "tas_trends" / "plot"
    plot_dir.mkdir(parents=True)
    xr.Dataset(
        {"tas": ("dim0", np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32))},
        coords={
            "dataset": ("dim0", np.array([b"CanESM5", b"CanESM5", b"ERA5   ", b"ERA5   "])),
            "shape_id": ("dim0", np.array([b"NWN ", b"NEN ", b"NWN ", b"NEN "])),
        },
    ).to_netcdf(plot_dir / "seaborn_barplot.nc")
    execution_dataset = ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                pd.DataFrame({"instance_id": ["a"], "source_id": ["CanESM5"]}), "instance_id"
            )
        }
    )

    metric, _ = RegionalHistoricalTrend.format_result(
        tmp_path, execution_dataset, CMECMetric.create_template(), CMECOutput.create_template()
    )

    assert metric.RESULTS["tas"] == {"NWN": {"trend": 1.0}, "NEN": {"trend": 2.0}}
