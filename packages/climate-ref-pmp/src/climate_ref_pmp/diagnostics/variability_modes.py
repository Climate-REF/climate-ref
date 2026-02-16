import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

from loguru import logger

from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    CommandLineDiagnostic,
    DataRequirement,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.esgf import CMIP6Request, CMIP7Request, RegistryRequest
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_pmp.pmp_driver import build_pmp_command, get_model_source_type, process_json_result

# CMIP7 branded variable names (from CMIP7 Data Request)
_BRANDED_VARIABLE_NAMES: dict[str, str] = {
    "ts": "ts_tavg-u-hxy-u",
    "psl": "psl_tavg-u-hxy-u",
}


class ExtratropicalModesOfVariability(CommandLineDiagnostic):
    """
    Calculate the extratropical modes of variability for a given area
    """

    ts_modes = ("PDO", "NPGO", "AMO")
    psl_modes = ("NAO", "NAM", "PNA", "NPO", "SAM")

    facets = (
        "mip_id",
        "source_id",
        "member_id",
        "experiment_id",
        "reference_source_id",
        "mode",
        "season",
        "method",
        "statistic",
    )

    def __init__(self, mode_id: str):
        super().__init__()
        self.mode_id = mode_id.upper()
        self.name = f"Extratropical modes of variability: {mode_id}"
        self.slug = f"extratropical-modes-of-variability-{mode_id.lower()}"

        def _get_data_requirements(
            obs_source: str,
            obs_variable: str,
            model_variable: str,
            extra_experiments: str | tuple[str, ...] | list[str] = (),
        ) -> tuple[tuple[DataRequirement, DataRequirement], ...]:
            cmip6_filters = [
                FacetFilter(
                    facets={
                        "frequency": "mon",
                        "experiment_id": ("historical", "hist-GHG", *extra_experiments),
                        "variable_id": model_variable,
                    }
                )
            ]

            cmip7_filters = [
                FacetFilter(
                    facets={
                        "branded_variable_name": (_BRANDED_VARIABLE_NAMES[model_variable],),
                        "experiment_id": ("historical", "hist-GHG", *extra_experiments),
                        "frequency": "mon",
                        "realm": "atmos",
                        "region": "glb",
                    }
                )
            ]

            obs_requirement = DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(FacetFilter(facets={"source_id": (obs_source,), "variable_id": (obs_variable,)}),),
                group_by=("source_id", "variable_id"),
            )
            cmip6_requirement = DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=tuple(cmip6_filters),
                group_by=("source_id", "experiment_id", "member_id", "grid_label"),
            )
            cmip7_requirement = DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=tuple(cmip7_filters),
                group_by=("source_id", "experiment_id", "variant_label", "grid_label"),
            )

            return (
                (obs_requirement, cmip6_requirement),
                (obs_requirement, cmip7_requirement),
            )

        if self.mode_id in self.ts_modes:
            self.parameter_file = "pmp_param_MoV-ts.py"
            self.data_requirements = _get_data_requirements("HadISST-1-1", "ts", "ts")
            self.test_data_spec = TestDataSpecification(
                test_cases=(
                    TestCase(
                        name="cmip6",
                        description=f"Test {self.mode_id} with CMIP6 ts data and HadISST obs",
                        requests=(
                            RegistryRequest(
                                slug=f"mov-{self.mode_id.lower()}-obs",
                                registry_name="obs4ref",
                                source_type="obs4MIPs",
                                facets={"source_id": "HadISST-1-1", "variable_id": "ts"},
                            ),
                            CMIP6Request(
                                slug=f"mov-{self.mode_id.lower()}-cmip6",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": "ts",
                                    "member_id": "r1i1p1f1",
                                    "frequency": "mon",
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                        ),
                    ),
                    TestCase(
                        name="cmip7",
                        description=f"CMIP7 test case for {self.mode_id}",
                        requests=(
                            RegistryRequest(
                                slug=f"mov-{self.mode_id.lower()}-obs-cmip7",
                                registry_name="obs4ref",
                                source_type="obs4MIPs",
                                facets={"source_id": "HadISST-1-1", "variable_id": "ts"},
                            ),
                            CMIP7Request(
                                slug=f"mov-{self.mode_id.lower()}-cmip7",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": "ts",
                                    "branded_variable_name": "ts_tavg-u-hxy-u",
                                    "variant_label": "r1i1p1f1",
                                    "frequency": "mon",
                                    "region": "glb",
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                        ),
                    ),
                ),
            )
        elif self.mode_id in self.psl_modes:
            self.parameter_file = "pmp_param_MoV-psl.py"
            self.data_requirements = _get_data_requirements("20CR", "psl", "psl", extra_experiments=("amip",))
            self.test_data_spec = TestDataSpecification(
                test_cases=(
                    TestCase(
                        name="cmip6",
                        description=f"Test {self.mode_id} with CMIP6 psl data and 20CR obs",
                        requests=(
                            RegistryRequest(
                                registry_name="obs4ref",
                                source_type="obs4MIPs",
                                slug=f"mov-{self.mode_id.lower()}-obs",
                                facets={"source_id": "20CR", "variable_id": "psl"},
                            ),
                            CMIP6Request(
                                slug=f"mov-{self.mode_id.lower()}-cmip6",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": "psl",
                                    "member_id": "r1i1p1f1",
                                    "frequency": "mon",
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                        ),
                    ),
                    TestCase(
                        name="cmip7",
                        description=f"CMIP7 test case for {self.mode_id}",
                        requests=(
                            RegistryRequest(
                                registry_name="obs4ref",
                                source_type="obs4MIPs",
                                slug=f"mov-{self.mode_id.lower()}-obs-cmip7",
                                facets={"source_id": "20CR", "variable_id": "psl"},
                            ),
                            CMIP7Request(
                                slug=f"mov-{self.mode_id.lower()}-cmip7",
                                facets={
                                    "source_id": "ACCESS-ESM1-5",
                                    "experiment_id": "historical",
                                    "variable_id": "psl",
                                    "branded_variable_name": "psl_tavg-u-hxy-u",
                                    "variant_label": "r1i1p1f1",
                                    "frequency": "mon",
                                    "region": "glb",
                                },
                                time_span=("2000-01", "2014-12"),
                            ),
                        ),
                    ),
                ),
            )
        else:
            raise ValueError(
                f"Unknown mode_id '{self.mode_id}'. Must be one of {self.ts_modes + self.psl_modes}"
            )

    def build_cmd(self, definition: ExecutionDefinition) -> Iterable[str]:
        """
        Build the command to run the diagnostic

        Parameters
        ----------
        definition
            Definition of the diagnostic execution

        Returns
        -------
            Command arguments to execute in the PMP environment
        """
        model_source_type = get_model_source_type(definition)
        input_datasets = definition.datasets[model_source_type]
        source_id = input_datasets["source_id"].unique()[0]
        experiment_id = input_datasets["experiment_id"].unique()[0]
        member_id_col = "variant_label" if model_source_type == SourceDatasetType.CMIP7 else "member_id"
        member_id = input_datasets[member_id_col].unique()[0]

        logger.debug(f"input_datasets: {input_datasets}")
        logger.debug(f"source_id: {source_id}")
        logger.debug(f"experiment_id: {experiment_id}")
        logger.debug(f"member_id: {member_id}")

        reference_dataset = definition.datasets[SourceDatasetType.obs4MIPs]
        reference_dataset_name = reference_dataset["source_id"].unique()[0]
        reference_dataset_path = reference_dataset.datasets.iloc[0]["path"]

        logger.debug(f"reference_dataset: {reference_dataset}")
        logger.debug(f"reference_dataset_name: {reference_dataset_name}")
        logger.debug(f"reference_dataset_path: {reference_dataset_path}")

        model_files = input_datasets.path.to_list()

        if len(model_files) != 1:
            # Have some logic to replace the dates in the filename with a wildcard
            raise NotImplementedError("Only one model file is supported at this time.")

        if isinstance(model_files, list):
            modpath = " ".join([str(p) for p in model_files])
        else:
            modpath = model_files

        if isinstance(reference_dataset_path, list):
            reference_data_path = " ".join([str(p) for p in reference_dataset_path])
        else:
            reference_data_path = reference_dataset_path

        # Build the command to run the PMP driver script
        params: dict[str, str | int | None] = {
            "variability_mode": self.mode_id,
            "modpath": modpath,
            "modpath_lf": "none",
            "mip": model_source_type.value,
            "exp": experiment_id,
            "realization": member_id,
            "modnames": source_id,
            "reference_data_name": reference_dataset_name,
            "reference_data_path": reference_data_path,
            "results_dir": str(definition.output_directory),
            "cmec": None,
            "no_provenance": None,
        }

        # Add conditional parameters
        if self.mode_id in ["SAM"]:  # pragma: no cover
            params["osyear"] = 1950
            params["oeyear"] = 2005

        if self.mode_id in ["NPO", "NPGO"]:
            params["eofn_obs"] = 2
            params["eofn_mod"] = 2
            params["eofn_mod_max"] = 2

        # Pass the parameters using **kwargs
        return build_pmp_command(
            driver_file="variability_modes_driver.py",
            parameter_file=self.parameter_file,
            **params,
        )

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
        model_source_type = get_model_source_type(definition)
        mip = model_source_type.value

        # Use mip-scoped glob to avoid matching files from other MIP runs
        results_files = list(definition.output_directory.glob(f"*_{mip}_*_cmec.json"))
        if len(results_files) != 1:  # pragma: no cover
            logger.warning(f"A single cmec output file not found: {results_files}")
            return ExecutionResult.build_from_failure(definition)

        clean_up_json(results_files[0])

        # Find the other outputs
        png_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.png")]
        data_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.nc")]

        cmec_output_bundle, cmec_metric_bundle = process_json_result(results_files[0], png_files, data_files)
        input_datasets = definition.datasets[model_source_type]
        reference_datasets = definition.datasets[SourceDatasetType.obs4MIPs]
        member_id_col = "variant_label" if model_source_type == SourceDatasetType.CMIP7 else "member_id"
        cmec_metric_bundle = cmec_metric_bundle.remove_dimensions(
            [
                "model",
                "realization",
                "reference",
            ],
        ).prepend_dimensions(
            {
                "mip_id": model_source_type.value,
                "source_id": input_datasets["source_id"].unique()[0],
                "member_id": input_datasets[member_id_col].unique()[0],
                "experiment_id": input_datasets["experiment_id"].unique()[0],
                "reference_source_id": reference_datasets["source_id"].unique()[0],
            }
        )

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output_bundle,
            cmec_metric_bundle=cmec_metric_bundle,
        )


def clean_up_json(json_file: Union[str, Path]) -> None:
    """
    Clean up the JSON file by removing unnecessary fields.

    Parameters
    ----------
    json_file : str or Path
        Path to the JSON file to clean up.
    """
    with open(str(json_file)) as f:
        data = json.load(f)

    # Remove null values from the JSON data
    data = remove_null_values(data)

    with open(str(json_file), "w") as f:
        json.dump(data, f, indent=4)

    # Log the cleanup action
    logger.debug(f"Cleaned up JSON file: {json_file}")
    logger.info("JSON file cleaned up successfully.")


def remove_null_values(data: Union[dict[Any, Any], list[Any], Any]) -> Union[dict[Any, Any], list[Any], Any]:
    """
    Recursively removes keys with null (None) values from a dictionary or list.

    Parameters
    ----------
    data : dict, list, or Any
        The JSON-like data structure to process. It can be a dictionary, a list,
        or any other type of data.

    Returns
    -------
    dict, list, or Any
        A new data structure with null values removed. If the input is a dictionary,
        keys with `None` values are removed. If the input is a list, items are
        recursively processed to remove `None` values. For other types, the input
        is returned unchanged.

    Examples
    --------
    >>> data = {
    ...     "key1": None,
    ...     "key2": {"subkey1": 123, "subkey2": None},
    ...     "key3": [None, 456, {"subkey3": None}],
    ... }
    >>> remove_null_values(data)
    {'key2': {'subkey1': 123}, 'key3': [456, {}]}
    """
    if isinstance(data, dict):
        return {key: remove_null_values(value) for key, value in data.items() if value is not None}
    if isinstance(data, list):
        return [remove_null_values(item) for item in data if item is not None]
    return data
