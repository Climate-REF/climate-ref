import pytest
from climate_ref_example.example import GlobalMeanTimeseries, calculate_annual_mean_timeseries

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType


@pytest.fixture
def metric_dataset(cmip6_data_catalog) -> ExecutionDatasetCollection:
    selected_dataset = cmip6_data_catalog[
        cmip6_data_catalog["instance_id"].isin(
            {
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn.v20210318",
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn.v20210318",
            }
        )
    ]
    return ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                selected_dataset,
                "instance_id",
            )
        }
    )


def test_annual_mean(sample_data_dir, metric_dataset):
    annual_mean = calculate_annual_mean_timeseries(metric_dataset["cmip6"].path.to_list())

    assert annual_mean.time.size == 86


def test_example_metric(metric_dataset, cmip6_data_catalog, mocker, definition_factory):
    metric = GlobalMeanTimeseries()
    ds = cmip6_data_catalog.groupby("instance_id").first()

    mock_calc = mocker.patch("climate_ref_example.example.calculate_annual_mean_timeseries")

    mock_calc.return_value.attrs.__getitem__.return_value = "ABC"

    definition = definition_factory(cmip6=DatasetCollection(ds, "instance_id"))

    result = metric.run(definition)

    assert mock_calc.call_count == 1

    assert str(result.output_bundle_filename) == "output.json"

    output_bundle_path = definition.output_directory / result.output_bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()

    assert str(result.metric_bundle_filename) == "diagnostic.json"

    output_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()
