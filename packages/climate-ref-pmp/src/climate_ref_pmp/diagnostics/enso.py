import json
import os
from collections.abc import Iterable
from typing import Any

from loguru import logger

from climate_ref_core.datasets import DatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    CommandLineDiagnostic,
    DataRequirement,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_pmp.pmp_driver import _get_resource, process_json_result


class ENSO(CommandLineDiagnostic):
    """
    Calculate the ENSO performance metrics for a dataset
    """

    facets = ()

    def __init__(self, metrics_collection: str) -> None:
        self.name = metrics_collection
        self.slug = metrics_collection.lower()
        self.metrics_collection = metrics_collection
        self.parameter_file = "pmp_param_enso.py"

        # TO DO: sftlf and areacell
        # TO DO: Get the path to the files per variable
        def _get_data_requirements(
            metrics_collection: str,
            extra_experiments: str | tuple[str, ...] | list[str] = (),
        ) -> tuple[DataRequirement, DataRequirement]:
            if metrics_collection == "ENSO_perf":
                model_variables: tuple[str, ...] = ("pr", "ts", "taux")
                obs_sources: tuple[str, ...] = ("GPCP", "ERA5")
            elif metrics_collection == "ENSO_tel":
                model_variables = ("pr", "ts")
                obs_sources = ("GPCP", "ERA5")
            elif metrics_collection == "ENSO_proc":
                model_variables = ("ts", "taux", "hfls", "hfss", "rlds", "rlus", "rsds", "rsus")
                obs_sources = ("GPCP", "ERA5", "TropFlux")
            else:
                raise ValueError(
                    f"Unknown metrics collection: {metrics_collection}. "
                    "Valid options are: ENSO_perf, ENSO_tel, ENSO_proc"
                )

            obs_variables = model_variables

            filters = [
                FacetFilter(
                    facets={
                        "frequency": "mon",
                        "experiment_id": ("historical", *extra_experiments),
                        "variable_id": model_variables,
                    }
                )
            ]

            return (
                DataRequirement(
                    source_type=SourceDatasetType.obs4MIPs,
                    filters=(FacetFilter(facets={"source_id": obs_sources, "variable_id": obs_variables}),),
                    group_by=(),
                ),
                DataRequirement(
                    source_type=SourceDatasetType.CMIP6,
                    filters=tuple(filters),
                    group_by=("source_id", "experiment_id", "member_id"),
                ),
            )

        self.data_requirements = _get_data_requirements(metrics_collection)

    def build_cmd(self, definition: ExecutionDefinition) -> Iterable[str]:
        """
        Run the diagnostic on the given configuration.

        Parameters
        ----------
        definition : ExecutionDefinition
            The configuration to run the diagnostic on.

        Returns
        -------
        :
            The result of running the diagnostic.
        """
        mc_name = self.metrics_collection

        # ------------------------------------------------
        # Get the input datasets information for the model
        # ------------------------------------------------
        input_datasets = definition.datasets[SourceDatasetType.CMIP6]
        source_id = input_datasets["source_id"].unique()[0]
        experiment_id = input_datasets["experiment_id"].unique()[0]
        member_id = input_datasets["member_id"].unique()[0]
        variable_ids = input_datasets["variable_id"].unique()
        mod_run = f"{source_id}_{member_id}"

        dict_mod: dict[str, dict[str, Any]] = {}

        def extract_variable(dc: DatasetCollection, variable: str) -> list[str]:
            return dc.datasets[input_datasets["variable_id"] == variable]["path"].to_list()  # type: ignore

        # TO DO: Get the path to the files per variable
        for variable in variable_ids:
            list_files = extract_variable(input_datasets, variable)
            list_areacella = extract_variable(input_datasets, "areacella")
            list_sftlf = extract_variable(input_datasets, "sftlf")

            dict_mod[mod_run][variable] = {
                "path + filename": list_files,
                "varname": variable,
                "path + filename_area": list_areacella,
                "areaname": "areacella",
                "path + filename_landmask": list_sftlf,
                "landmaskname": "sftlf",
            }

        # -------------------------------------------------------
        # Get the input datasets information for the observations
        # -------------------------------------------------------
        reference_dataset = definition.datasets[SourceDatasetType.obs4MIPs]
        reference_dataset_names = reference_dataset["source_id"].unique()

        dict_obs: dict[str, dict[str, Any]] = {}

        # TO DO: Get the path to the files per variable and per source
        for obs_name in reference_dataset_names:
            for variable in variable_ids:
                list_files = input_datasets.datasets[input_datasets["variable_id"] == variable][
                    "path"
                ].to_list()
                dict_obs[obs_name][variable] = {
                    "path + filename": list_files,
                    "varname": variable,
                }

        # Create input directory
        dictDatasets = {
            "model": dict_mod,
            "observations": dict_obs,
            "metricsCollection": mc_name,
            "experiment_id": experiment_id,
        }

        # Create JSON file for dictDatasets
        json_file = os.path.join(
            definition.output_directory, f"input_{mc_name}_{source_id}_{experiment_id}_{member_id}.json"
        )
        with open(json_file, "w") as f:
            json.dump(dictDatasets, f, indent=4)
        logger.debug(f"JSON file created: {json_file}")

        driver_file = _get_resource("climate_ref_pmp.drivers", "enso_driver.py", use_resources=True)
        return [
            "python",
            driver_file,
            "--metrics_collection",
            mc_name,
            "--experiment_id",
            experiment_id,
            "--input_json_path",
            json_file,
            "--output_directory",
            str(definition.output_directory),
        ]

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Build a diagnostic result from the output of the PMP driver

        Parameters
        ----------
        definition
            Definition of the diagnostic execution

        Returns
        -------
            Result of the diagnostic execution
        """
        input_datasets = definition.datasets[SourceDatasetType.CMIP6]
        source_id = input_datasets["source_id"].unique()[0]
        experiment_id = input_datasets["experiment_id"].unique()[0]
        member_id = input_datasets["member_id"].unique()[0]
        mc_name = self.metrics_collection
        pattern = f"{mc_name}_{source_id}_{experiment_id}_{member_id}"

        results_files = list(definition.output_directory.glob(f"{pattern}.json"))
        if len(results_files) != 1:  # pragma: no cover
            logger.warning(f"A single cmec output file not found: {results_files}")
            return ExecutionResult.build_from_failure(definition)

        # Find the other outputs
        png_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.png")]
        data_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.nc")]

        cmec_output, cmec_metric = process_json_result(results_files[0], png_files, data_files)

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output,
            cmec_metric_bundle=cmec_metric,
        )
