from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import pandas
from ruamel.yaml import YAML

from cmip_ref_core.datasets import SourceDatasetType
from cmip_ref_core.metrics import CommandLineMetric, MetricExecutionDefinition, MetricExecutionResult
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput
from cmip_ref_metrics_esmvaltool.recipe import load_recipe, prepare_climate_data
from cmip_ref_metrics_esmvaltool.types import OutputBundle, Recipe

yaml = YAML()


class ESMValToolMetric(CommandLineMetric):
    """ESMValTool Metric base class."""

    base_recipe: ClassVar[str]

    @staticmethod
    @abstractmethod
    def update_recipe(recipe: Recipe, input_files: pandas.DataFrame) -> None:
        """
        Update the base recipe for the run.

        Parameters
        ----------
        recipe:
            The base recipe to update.
        input_files:
            The dataframe describing the input files.

        """

    @staticmethod
    @abstractmethod
    def format_result(result_dir: Path) -> OutputBundle:
        """
        Create a CMEC output bundle for the results.

        Parameters
        ----------
        result_dir
            Directory containing results from an ESMValTool run.

        Returns
        -------
            A CMEC output bundle.
        """

    def build_cmd(self, definition: MetricExecutionDefinition) -> Iterable[str]:
        """
        Build the command to run an ESMValTool recipe.

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the metric

        Returns
        -------
        :
            The result of running the metric.
        """
        input_files = definition.metric_dataset[SourceDatasetType.CMIP6].datasets
        recipe = load_recipe(self.base_recipe)
        self.update_recipe(recipe, input_files)

        recipe_path = definition.to_output_path("recipe.yml")
        with recipe_path.open("w", encoding="utf-8") as file:
            yaml.dump(recipe, file)

        climate_data = definition.to_output_path("climate_data")

        prepare_climate_data(
            definition.metric_dataset[SourceDatasetType.CMIP6].datasets,
            climate_data_dir=climate_data,
        )

        config = {
            "drs": {
                "CMIP6": "ESGF",
            },
            "output_dir": str(definition.to_output_path("results")),
            "rootpath": {
                "default": str(climate_data),
            },
            "search_esgf": "never",
        }
        config_dir = definition.to_output_path("config")
        config_dir.mkdir()
        with (config_dir / "config.yml").open("w", encoding="utf-8") as file:
            yaml.dump(config, file)

        return [
            "esmvaltool",
            "run",
            f"--config-dir={config_dir}",
            f"{recipe_path}",
        ]

    def build_metric_result(
        self,
        definition: MetricExecutionDefinition,
    ) -> MetricExecutionResult:
        """
        Build the metric result after running an ESMValTool recipe.

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the metric

        Returns
        -------
        :
            The resulting metric.
        """
        result_dir = next(definition.to_output_path("results").glob("*"))

        metric_bundle = self.format_result(result_dir)
        CMECMetric.model_validate(metric_bundle)

        output_bundle = CMECOutput.create_template()
        CMECOutput.model_validate(output_bundle)

        return MetricExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=output_bundle,
            cmec_metric_bundle=metric_bundle,
        )
