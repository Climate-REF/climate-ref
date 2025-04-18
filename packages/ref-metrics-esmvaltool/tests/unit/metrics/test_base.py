import json

import cmip_ref_metrics_esmvaltool.metrics.base
import pandas
import pytest
from cmip_ref_metrics_esmvaltool.metrics.base import ESMValToolMetric
from cmip_ref_metrics_esmvaltool.types import Recipe
from ruamel.yaml import YAML

from cmip_ref_core.pycmec.output import OutputCV

yaml = YAML()


@pytest.fixture
def mock_metric():
    class MockMetric(ESMValToolMetric):
        base_recipe = "examples/recipe_python.yml"

        def update_recipe(self, recipe: Recipe, input_files: pandas.DataFrame) -> None:
            pass

    return MockMetric()


@pytest.mark.parametrize("data_dir_exists", [True, False])
def test_build_cmd(mocker, tmp_path, metric_definition, mock_metric, data_dir_exists):
    dataset_registry_manager = mocker.patch.object(
        cmip_ref_metrics_esmvaltool.metrics.base,
        "dataset_registry_manager",
    )
    data_dir = tmp_path / "ESMValTool"
    if data_dir_exists:
        data_dir.mkdir()
    dataset_registry_manager.__getitem__.return_value.abspath = tmp_path
    output_dir = metric_definition.output_directory
    output_dir.mkdir(parents=True)
    cmd = mock_metric.build_cmd(metric_definition)
    config_dir = output_dir / "config"
    recipe = output_dir / "recipe.yml"
    assert cmd == ["esmvaltool", "run", f"--config-dir={config_dir}", f"{recipe}"]
    assert (output_dir / "climate_data").is_dir()
    config = yaml.load((config_dir / "config.yml").read_text(encoding="utf-8"))
    assert len(config["rootpath"]) == 4 if data_dir_exists else 1


def test_build_metric_result(metric_definition, mock_metric):
    results_dir = metric_definition.to_output_path("results") / "recipe_test"

    for subdir in "timeseries", "map":
        metadata = {}
        for dirname in "work", "plots":
            for i in range(2):
                suffix = ".nc" if dirname == "work" else ".png"
                filepath = results_dir / dirname / subdir / "script1" / f"file{i}{suffix}"
                metadata[str(filepath)] = {
                    "caption": f"This is {subdir} test file {i}.",
                }
        metadata_file = results_dir / "run" / subdir / "script1" / "diagnostic_provenance.yml"
        metadata_file.parent.mkdir(parents=True)
        with metadata_file.open("w", encoding="utf-8") as file:
            yaml.dump(metadata, file)

    execution_result = mock_metric.build_metric_result(definition=metric_definition)
    metric_bundle = json.loads(
        execution_result.to_output_path(execution_result.metric_bundle_filename).read_text(encoding="utf-8")
    )
    output_bundle = json.loads(
        execution_result.to_output_path(execution_result.output_bundle_filename).read_text(encoding="utf-8")
    )

    assert isinstance(metric_bundle, dict)
    assert metric_bundle

    assert isinstance(output_bundle, dict)
    assert OutputCV.DATA.value in output_bundle
    assert len(output_bundle[OutputCV.DATA.value]) == 4
    assert OutputCV.PLOTS.value in output_bundle
    plots = output_bundle[OutputCV.PLOTS.value]
    assert len(plots) == 4
    captions = {p["long_name"] for p in plots.values()}
    assert len(captions) == 4
