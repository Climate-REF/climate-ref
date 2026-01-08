import json
import os
from collections.abc import Collection, Iterable
from typing import Any

from loguru import logger

from climate_ref_core.constraints import AddSupplementaryDataset
from climate_ref_core.datasets import DatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    CommandLineDiagnostic,
    DataRequirement,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.esgf import CMIP6Request, Obs4MIPsRequest
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_pmp.pmp_driver import _get_resource, process_json_result


class ENSO(CommandLineDiagnostic):
    """
    Calculate the ENSO performance metrics for a dataset
    """

    facets = ("source_id", "member_id", "grid_label", "experiment_id", "metric", "reference_datasets")

    def __init__(self, metrics_collection: str, experiments: Collection[str] = ("historical",)) -> None:
        self.name = metrics_collection
        self.slug = metrics_collection.lower()
        self.metrics_collection = metrics_collection
        self.parameter_file = "pmp_param_enso.py"
        self.obs_sources: tuple[str, ...]
        self.model_variables: tuple[str, ...]

        if metrics_collection == "ENSO_perf":  # pragma: no cover
            self.model_variables = ("pr", "ts", "tauu")
            self.obs_sources = ("GPCP-Monthly-3-2", "TropFlux-1-0", "HadISST-1-1")
        elif metrics_collection == "ENSO_tel":
            self.model_variables = ("pr", "ts")
            self.obs_sources = ("GPCP-Monthly-3-2", "TropFlux-1-0", "HadISST-1-1")
        elif metrics_collection == "ENSO_proc":
            self.model_variables = ("ts", "tauu", "hfls", "hfss", "rlds", "rlus", "rsds", "rsus")
            self.obs_sources = (
                "GPCP-Monthly-3-2",
                "TropFlux-1-0",
                "HadISST-1-1",
                "CERES-EBAF-4-2",
            )
        else:
            raise ValueError(
                f"Unknown metrics collection: {metrics_collection}. "
                "Valid options are: ENSO_perf, ENSO_tel, ENSO_proc"
            )

        self.data_requirements = self._get_data_requirements(experiments)

        # Set test data specification based on metrics collection
        if metrics_collection == "ENSO_tel":
            self.test_data_spec = TestDataSpecification(
                test_cases=(
                    TestCase(
                        name="default",
                        description="ENSO teleconnections from ACCESS-ESM1-5 historical",
                        requests=(
                            CMIP6Request(
                                slug="cmip6",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": ("pr", "ts"),
                                    "member_id": "r1i1p1f1",
                                    "table_id": "Amon",
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                            CMIP6Request(
                                slug="supplementary",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": ("areacella", "sftlf"),
                                    "table_id": "fx",
                                },
                            ),
                            Obs4MIPsRequest(
                                slug="obs4mips",
                                facets={
                                    "source_id": ("GPCP-Monthly-3-2", "HadISST-1-1"),
                                    "variable_id": ("pr", "ts"),
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                        ),
                    ),
                ),
            )
        elif metrics_collection == "ENSO_proc":
            self.test_data_spec = TestDataSpecification(
                test_cases=(
                    TestCase(
                        name="default",
                        description="ENSO processes from ACCESS-ESM1-5 historical",
                        requests=(
                            CMIP6Request(
                                slug="cmip6",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": (
                                        "ts",
                                        "tauu",
                                        "hfls",
                                        "hfss",
                                        "rlds",
                                        "rlus",
                                        "rsds",
                                        "rsus",
                                    ),
                                    "member_id": "r1i1p1f1",
                                    "table_id": "Amon",
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                            CMIP6Request(
                                slug="supplementary",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": ("areacella", "sftlf"),
                                    "table_id": "fx",
                                },
                            ),
                            Obs4MIPsRequest(
                                slug="obs4mips",
                                facets={
                                    "source_id": (
                                        "GPCP-Monthly-3-2",
                                        "TropFlux-1-0",
                                        "HadISST-1-1",
                                        "CERES-EBAF-4-2",
                                    ),
                                    "variable_id": (
                                        "ts",
                                        "tauu",
                                        "hfls",
                                        "hfss",
                                        "rlds",
                                        "rlus",
                                        "rsds",
                                        "rsus",
                                    ),
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                        ),
                    ),
                ),
            )

    def _get_data_requirements(
        self,
        experiments: Collection[str] = ("historical",),
    ) -> tuple[DataRequirement, DataRequirement]:
        filters = [
            FacetFilter(
                facets={
                    "frequency": "mon",
                    "experiment_id": tuple(experiments),
                    "variable_id": self.model_variables,
                }
            )
        ]

        return (
            DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(
                    FacetFilter(facets={"source_id": self.obs_sources, "variable_id": self.model_variables}),
                ),
                group_by=("activity_id",),
            ),
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=tuple(filters),
                group_by=("source_id", "experiment_id", "member_id", "grid_label"),
                constraints=(
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                    AddSupplementaryDataset.from_defaults("sftlf", SourceDatasetType.CMIP6),
                ),
            ),
        )

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
        input_selectors = input_datasets.selector_dict()
        source_id = input_selectors["source_id"]
        member_id = input_selectors["member_id"]
        experiment_id = input_selectors["experiment_id"]
        variable_ids = set(input_datasets["variable_id"].unique()) - {"areacella", "sftlf"}
        mod_run = f"{source_id}_{member_id}"

        # We only need one entry for the model run
        dict_mod: dict[str, dict[str, Any]] = {mod_run: {}}

        def extract_variable(dc: DatasetCollection, variable: str) -> list[str]:
            return dc.datasets[input_datasets["variable_id"] == variable]["path"].to_list()  # type: ignore

        # TO DO: Get the path to the files per variable
        for variable in variable_ids:
            list_files = extract_variable(input_datasets, variable)
            list_areacella = extract_variable(input_datasets, "areacella")
            list_sftlf = extract_variable(input_datasets, "sftlf")

            if len(list_files) > 0:
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
            dict_obs[obs_name] = {}
            for variable in variable_ids:
                # Get the list of files for the current variable and observation source
                list_files = reference_dataset.datasets[
                    (reference_dataset["variable_id"] == variable)
                    & (reference_dataset["source_id"] == obs_name)
                ]["path"].to_list()
                # If the list is not empty, add it to the dictionary
                if len(list_files) > 0:
                    dict_obs[obs_name][variable] = {
                        "path + filename": list_files,
                        "varname": variable,
                    }

        # Create input directory
        dict_datasets = {
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
            json.dump(dict_datasets, f, indent=4)
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

        # Find the results files
        results_files = list(definition.output_directory.glob(f"{pattern}_cmec.json"))
        logger.debug(f"Results files: {results_files}")

        if len(results_files) != 1:  # pragma: no cover
            logger.warning(f"A single cmec output file not found: {results_files}")
            return ExecutionResult.build_from_failure(definition)

        # Find the other outputs
        png_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.png")]
        data_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.nc")]

        cmec_output, cmec_metric = process_json_result(results_files[0], png_files, data_files)

        input_selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()
        cmec_metric_bundle = cmec_metric.remove_dimensions(
            [
                "model",
                "realization",
            ],
        ).prepend_dimensions(
            {
                "source_id": input_selectors["source_id"],
                "member_id": input_selectors["member_id"],
                "grid_label": input_selectors["grid_label"],
                "experiment_id": input_selectors["experiment_id"],
            }
        )

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output,
            cmec_metric_bundle=cmec_metric_bundle,
        )
