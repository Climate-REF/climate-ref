import fnmatch
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import pandas
import xarray as xr
import yaml
from loguru import logger

from climate_ref_core.dataset_registry import dataset_registry_manager
from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import (
    CommandLineDiagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.metric_values.typing import SeriesMetricValue
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput, OutputCV
from climate_ref_esmvaltool.recipe import load_recipe, prepare_climate_data
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class ESMValToolDiagnostic(CommandLineDiagnostic):
    """ESMValTool Diagnostic base class."""

    base_recipe: ClassVar[str]

    @staticmethod
    @abstractmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
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
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """
        Update the arguments needed to create a CMEC diagnostic and output bundle.

        Parameters
        ----------
        result_dir
            Directory containing executions from an ESMValTool run.
        execution_dataset
            The diagnostic dataset used for the diagnostic execution.
        metric_args
            Generic diagnostic bundle arguments.
        output_args
            Generic output bundle arguments.

        Returns
        -------
            The arguments needed to create a CMEC diagnostic and output bundle.
        """
        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)

    def write_recipe(self, definition: ExecutionDefinition) -> Path:
        """
        Update the ESMValTool recipe for the diagnostic and write it to file.

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic

        Returns
        -------
        :
            The path to the written recipe.
        """
        input_files = {
            project: dataset_collection.datasets
            for project, dataset_collection in definition.datasets.items()
        }
        recipe = load_recipe(self.base_recipe)
        self.update_recipe(recipe, input_files)
        recipe_txt = yaml.safe_dump(recipe, sort_keys=False)
        logger.info(f"Using ESMValTool recipe:\n{recipe_txt}")
        recipe_path = definition.to_output_path("recipe.yml")
        with recipe_path.open("w", encoding="utf-8") as file:
            file.write(recipe_txt)
        return recipe_path

    def build_cmd(self, definition: ExecutionDefinition) -> Iterable[str]:
        """
        Build the command to run an ESMValTool recipe.

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic

        Returns
        -------
        :
            The result of running the diagnostic.
        """
        recipe_path = self.write_recipe(definition)
        climate_data = definition.to_output_path("climate_data")

        for metric_dataset in definition.datasets.values():
            prepare_climate_data(
                metric_dataset.datasets,
                climate_data_dir=climate_data,
            )

        config = {
            "drs": {
                "CMIP6": "ESGF",
                "obs4MIPs": "ESGF",
            },
            "output_dir": str(definition.to_output_path("executions")),
            "rootpath": {
                "CMIP6": str(climate_data),
                "obs4MIPs": str(climate_data),
            },
            "search_esgf": "never",
        }

        # Configure the paths to OBS/OBS6/native6 and non-compliant obs4MIPs data
        registry = dataset_registry_manager["esmvaltool"]
        data_dir = registry.abspath / "ESMValTool"  # type: ignore[attr-defined]
        if not data_dir.exists():
            logger.warning(
                "ESMValTool observational and reanalysis data is not available "
                f"in {data_dir}, you may want to run the command "
                "`ref datasets fetch-data --registry esmvaltool`."
            )
        else:
            config["drs"].update(  # type: ignore[attr-defined]
                {
                    "OBS": "default",
                    "OBS6": "default",
                    "native6": "default",
                }
            )
            config["rootpath"].update(  # type: ignore[attr-defined]
                {
                    "OBS": str(data_dir / "OBS"),
                    "OBS6": str(data_dir / "OBS"),
                    "native6": str(data_dir / "native6"),
                }
            )
            config["rootpath"]["obs4MIPs"] = [  # type: ignore[index]
                config["rootpath"]["obs4MIPs"],  # type: ignore[index]
                str(data_dir),
            ]

        config_dir = definition.to_output_path("config")
        config_dir.mkdir()
        config_txt = yaml.safe_dump(config)
        logger.info(f"Using ESMValTool configuration:\n{config_txt}")
        with (config_dir / "config.yml").open("w", encoding="utf-8") as file:
            file.write(config_txt)

        return [
            "esmvaltool",
            "run",
            f"--config-dir={config_dir}",
            f"{recipe_path}",
        ]

    def build_execution_result(
        self,
        definition: ExecutionDefinition,
    ) -> ExecutionResult:
        """
        Build the diagnostic result after running an ESMValTool recipe.

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic

        Returns
        -------
        :
            The resulting diagnostic.
        """
        result_dir = next(definition.to_output_path("executions").glob("*"))

        metric_args = CMECMetric.create_template()
        output_args = CMECOutput.create_template()

        # Input selectors for the datasets used in the diagnostic.
        # TODO: Better handling of multiple source types
        if SourceDatasetType.CMIP6 in definition.datasets:
            input_selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()
        elif SourceDatasetType.obs4MIPs in definition.datasets:
            input_selectors = definition.datasets[SourceDatasetType.obs4MIPs].selector_dict()
        else:
            input_selectors = {}

        # Add the plots and data files
        series = []
        plot_suffixes = {".png", ".jpg", ".pdf", ".ps"}
        for metadata_file in result_dir.glob("run/*/*/diagnostic_provenance.yml"):
            metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
            for filename in metadata:
                caption = metadata[filename].get("caption", "")
                relative_path = definition.as_relative_path(filename)
                for file_def in (*definition.diagnostic.files, *definition.diagnostic.series):
                    if fnmatch.fnmatch(
                        str(relative_path),
                        f"executions/*/{file_def.file_pattern.format(**input_selectors)}",
                    ):
                        dimensions = file_def.dimensions
                        break
                else:
                    dimensions = {}
                if relative_path.suffix in plot_suffixes:
                    key = OutputCV.PLOTS.value
                else:
                    key = OutputCV.DATA.value
                output_args[key][f"{relative_path}"] = {
                    OutputCV.FILENAME.value: f"{relative_path}",
                    OutputCV.LONG_NAME.value: caption,
                    OutputCV.DESCRIPTION.value: "",
                    OutputCV.DIMENSIONS.value: dimensions,
                }
                series.extend(
                    self._extract_series_from_file(
                        definition,
                        filename,
                        relative_path,
                        caption=caption,
                        input_selectors=input_selectors,
                    )
                )

        # Add the index.html file
        index_html = f"{result_dir}/index.html"
        output_args[OutputCV.HTML.value][index_html] = {
            OutputCV.FILENAME.value: index_html,
            OutputCV.LONG_NAME.value: "Results page",
            OutputCV.DESCRIPTION.value: "Page showing the executions of the ESMValTool run.",
        }
        output_args[OutputCV.INDEX.value] = index_html

        # Add the (debug) log file
        output_args[OutputCV.PROVENANCE.value][OutputCV.LOG.value] = f"{result_dir}/run/main_log_debug.txt"

        # Update the diagnostic and output bundle with diagnostic specific executions.
        metric_bundle, output_bundle = self.format_result(
            result_dir=result_dir,
            execution_dataset=definition.datasets,
            metric_args=metric_args,
            output_args=output_args,
        )

        # Add the extra information from the groupby operations
        if len(metric_bundle.DIMENSIONS[MetricCV.JSON_STRUCTURE.value]):
            metric_bundle = metric_bundle.prepend_dimensions(input_selectors)

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=output_bundle,
            cmec_metric_bundle=metric_bundle,
            series=series,
        )

    def _extract_series_from_file(
        self,
        definition: ExecutionDefinition,
        filename: Path,
        relative_path: Path,
        caption: str,
        input_selectors: dict[str, str],
    ) -> list[SeriesMetricValue]:
        """
        Extract series data from a file if it matches any of the series definitions.
        """
        variable_attributes = (
            "long_name",
            "standard_name",
            "units",
        )

        series = []
        for series_def in definition.diagnostic.series:
            if fnmatch.fnmatch(
                str(relative_path),
                f"executions/*/{series_def.file_pattern.format(**input_selectors)}",
            ):
                dataset = xr.open_dataset(filename, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
                dataset = dataset.sel(series_def.sel)
                attributes = {
                    attr: dataset.attrs[attr] for attr in series_def.attributes if attr in dataset.attrs
                }
                attributes["caption"] = caption
                attributes["values_name"] = series_def.values_name
                attributes["index_name"] = series_def.index_name
                for attr in variable_attributes:
                    if attr in dataset[series_def.values_name].attrs:
                        attributes[f"value_{attr}"] = dataset[series_def.values_name].attrs[attr]
                    if attr in dataset[series_def.index_name].attrs:
                        attributes[f"index_{attr}"] = dataset[series_def.index_name].attrs[attr]
                index = dataset[series_def.index_name].values.tolist()
                if hasattr(index[0], "calendar"):
                    attributes["calendar"] = index[0].calendar
                if hasattr(index[0], "isoformat"):
                    # Convert time objects to strings.
                    index = [v.isoformat() for v in index]

                series.append(
                    SeriesMetricValue(
                        dimensions={**input_selectors, **series_def.dimensions},
                        values=dataset[series_def.values_name].values.tolist(),
                        index=index,
                        index_name=series_def.index_name,
                        attributes=attributes,
                    )
                )
        return series
