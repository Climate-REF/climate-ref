import pytest
from climate_ref_example.example import GlobalMeanTimeseries, calculate_annual_mean_timeseries

from climate_ref_core.datasets import DatasetCollection


@pytest.fixture
def metric_dataset(cmip6_data_catalog) -> DatasetCollection:
    selected_dataset = cmip6_data_catalog[
        cmip6_data_catalog["instance_id"].isin(
            {
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn.v20210318",
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn.v20210318",
            }
        )
    ]
    return DatasetCollection(
        selected_dataset,
        "instance_id",
        selector=(
            ("source_id", "CSIRO.ACCESS-ESM1-5"),
            ("variable_id", "tas"),
            ("experiment_id", "ssp126"),
            ("variant_label", "r1i1p1f1"),
        ),
    )


def test_annual_mean(sample_data_dir, metric_dataset):
    annual_mean = calculate_annual_mean_timeseries(metric_dataset.path.to_list())

    assert annual_mean.time.size == 86


def test_example_metric(metric_dataset, cmip6_data_catalog, definition_factory):
    diagnostic = GlobalMeanTimeseries()

    definition = definition_factory(diagnostic=diagnostic, cmip6=metric_dataset)
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert str(result.output_bundle_filename) == "output.json"

    output_bundle_path = definition.output_directory / result.output_bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()

    assert str(result.metric_bundle_filename) == "diagnostic.json"

    output_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()
