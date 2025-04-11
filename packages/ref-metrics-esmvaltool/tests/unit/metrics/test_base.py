import json

import pandas
from cmip_ref_metrics_esmvaltool.metrics.base import ESMValToolMetric
from cmip_ref_metrics_esmvaltool.types import Recipe
from ruamel.yaml import YAML

from cmip_ref_core.pycmec.output import OutputCV

yaml = YAML()


def test_build_metric_result(metric_definition):
    class MockMetric(ESMValToolMetric):
        def update_recipe(recipe: Recipe, input_files: pandas.DataFrame) -> None:
            pass

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

    metric = MockMetric()

    execution_result = metric.build_metric_result(definition=metric_definition)
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
