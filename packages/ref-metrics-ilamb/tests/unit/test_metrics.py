import cmip_ref_metrics_ilamb
import pytest
from cmip_ref_metrics_ilamb.example import ILAMBStandardTAS

from cmip_ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType
from cmip_ref_core.metrics import MetricExecutionDefinition


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


def test_example_metric(tmp_path, mocker, metric_dataset, cmip6_data_catalog):
    metric = ILAMBStandardTAS()
    ds = cmip6_data_catalog.groupby("instance_id", as_index=False).first()
    output_directory = tmp_path / "output"

    definition = MetricExecutionDefinition(
        output_directory=output_directory,
        output_fragment=tmp_path,
        key="ilamb-standard-tas",
        metric_dataset=MetricDataset(
            {
                SourceDatasetType.CMIP6: DatasetCollection(ds, "instance_id"),
            }
        ),
    )

    result_dir = definition.output_fragment / "results"
    result = result_dir / "result.nc"

    def mock_check_call(cmd, *args, **kwargs):
        result.parent.mkdir(parents=True)
        result.touch()

    mocker.patch.object(
        cmip_ref_metrics_ilamb.example.subprocess,
        "check_call",
        autospec=True,
        spec_set=True,
        side_effect=mock_check_call,
    )

    result = metric.run(definition)

    output_bundle_path = definition.output_directory / definition.output_fragment / result.bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()
    assert result.bundle_filename.name == "output.json"
