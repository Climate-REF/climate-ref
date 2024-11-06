from ref_core.metrics import Configuration, TriggerInfo
from ref_metrics_example.example import AnnualGlobalMeanTimeseries, calculate_annual_mean_timeseries


def test_annual_mean(esgf_data_dir, test_dataset):
    annual_mean = calculate_annual_mean_timeseries(test_dataset)

    assert annual_mean.time.size == 286


def test_example_metric(tmp_path, test_dataset):
    metric = AnnualGlobalMeanTimeseries()

    configuration = Configuration(
        output_fragment=tmp_path,
    )

    result = metric.run(configuration, trigger=TriggerInfo(dataset=test_dataset))

    assert result.successful
    assert result.output_bundle.exists()
    assert result.output_bundle.is_file()
    assert result.output_bundle.name == "output.json"


def test_example_metric_no_trigger(tmp_path, test_dataset):
    metric = AnnualGlobalMeanTimeseries()

    configuration = Configuration(
        output_fragment=tmp_path,
    )

    result = metric.run(configuration, trigger=None)
    assert result.successful is False
