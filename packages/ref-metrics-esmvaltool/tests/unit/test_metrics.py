import cmip_ref_metrics_esmvaltool
import pytest

from cmip_ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType


@pytest.fixture
def metric_dataset(cmip6_data_catalog) -> MetricDataset:
    selected_dataset = cmip6_data_catalog[
        cmip6_data_catalog["instance_id"] == cmip6_data_catalog.instance_id.iloc[0]
    ]
    return MetricDataset(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                selected_dataset,
                "instance_id",
            )
        }
    )


def test_example_metric(mocker, tmp_path, metric_dataset, cmip6_data_catalog, definition_factory):
    provider = cmip_ref_metrics_esmvaltool.provider

    metric = next(
        metric for metric in provider.metrics() if metric.slug == "esmvaltool-global-mean-timeseries"
    )
    ds = cmip6_data_catalog.groupby("instance_id", as_index=False).first()

    definition = definition_factory(cmip6=DatasetCollection(ds, "instance_id"))
    definition.output_directory.mkdir(parents=True)

    result_dir = definition.output_directory / "results" / "recipe_test_a"
    result = result_dir / "work" / "timeseries" / "script1" / "result.nc"

    def mock_run_fn(cmd, *args, **kwargs):
        result.parent.mkdir(parents=True)
        result.touch()

    mock_run = mocker.patch.object(
        provider,
        "run",
        autospec=True,
        spec_set=True,
        side_effect=mock_run_fn,
    )

    result = metric.run(definition)

    mock_run.assert_called_with(
        [
            "esmvaltool",
            "run",
            f"--config-dir={definition.to_output_path('config')}",
            f"{definition.to_output_path('recipe.yml')}",
        ],
    )

    output_bundle_path = definition.output_directory / result.output_bundle_filename
    metric_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()
    assert result.output_bundle_filename.name == "output.json"

    assert metric_bundle_path.exists()
    assert metric_bundle_path.is_file()
    assert result.metric_bundle_filename.name == "metric.json"
