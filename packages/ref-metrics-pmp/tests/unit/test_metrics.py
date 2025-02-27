import pandas as pd
from cmip_ref_metrics_pmp.example import ExtratropicalModesOfVariability_PDO

from cmip_ref.solver import extract_covered_datasets
from cmip_ref_core.datasets import DatasetCollection
from cmip_ref_core.metrics import Metric


def get_first_metric_match(data_catalog: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    datasets = extract_covered_datasets(data_catalog, metric.data_requirements[0])
    assert len(datasets) > 0
    return datasets[0]


def test_example_metric(cmip6_data_catalog, mocker, definition_factory):
    metric = ExtratropicalModesOfVariability_PDO()
    metric_dataset = get_first_metric_match(cmip6_data_catalog, metric)

    definition = definition_factory(cmip6=DatasetCollection(metric_dataset, "instance_id"))

    result = metric.run(definition)

    assert str(result.output_bundle_filename) == "output.json"

    output_bundle_path = definition.output_directory / result.output_bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()

    assert str(result.metric_bundle_filename) == "metric.json"

    metric_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert result.successful
    assert metric_bundle_path.exists()
    assert metric_bundle_path.is_file()
