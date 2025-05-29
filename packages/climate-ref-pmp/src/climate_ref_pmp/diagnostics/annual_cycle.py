import datetime
import json
from pathlib import Path
from typing import Any

from loguru import logger

from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    CommandLineDiagnostic,
    DataRequirement,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.pycmec.metric import remove_dimensions
from climate_ref_pmp.pmp_driver import build_glob_pattern, build_pmp_command, process_json_result

# =================================================================
# PMP diagnostics support functions for the annual cycle diagnostic
# =================================================================


def make_data_requirement(variable_id: str, obs_source: str) -> tuple[DataRequirement, DataRequirement]:
    """
    Create a data requirement for the annual cycle diagnostic.

    Parameters
    ----------
    variable_id : str
        The variable ID to filter the data requirement.
    obs_source : str
        The observation source ID to filter the data requirement.

    Returns
    -------
    DataRequirement
        A DataRequirement object containing the necessary filters and groupings.
    """
    return (
        DataRequirement(
            source_type=SourceDatasetType.PMPClimatology,
            filters=(FacetFilter(facets={"source_id": (obs_source,), "variable_id": (variable_id,)}),),
            group_by=("variable_id", "source_id"),
        ),
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "frequency": "mon",
                        "experiment_id": ("amip", "historical", "hist-GHG", "piControl"),
                        "variable_id": (variable_id,),
                    }
                ),
            ),
            group_by=("variable_id", "source_id", "experiment_id", "member_id", "grid_label"),
        ),
    )


def _transform_results(data: dict[str, Any]) -> dict[str, Any]:
    """
    Transform the executions dictionary to match the expected structure.

    Parameters
    ----------
    data : dict
        The original execution dictionary.

    Returns
    -------
    dict
        The transformed executions dictionary.
    """
    # Remove the model, reference, rip dimensions
    # These are later replaced with a REF-specific naming convention
    data = remove_dimensions(data, ["model", "reference", "rip"])

    # TODO: replace this with the ability to capture series
    # Remove the "CalendarMonths" key from the nested structure
    for region, region_values in data["RESULTS"].items():
        for stat, stat_values in region_values.items():
            if "CalendarMonths" in stat_values:
                stat_values.pop("CalendarMonths")

    # Remove the "CalendarMonths" key from the nested structure in "DIMENSIONS"
    data["DIMENSIONS"]["season"].pop("CalendarMonths")

    return data


def transform_results_files(results_files: list[Any]) -> list[Any]:
    """
    Transform the results files to match the expected structure.

    Parameters
    ----------
    results_files : list
        List of result files to transform.

    Returns
    -------
    list
        List of transformed result files.

    """
    if len(results_files) == 0:
        logger.warning("No results files provided for transformation.")
        return []

    transformed_results_files = []

    for results_file in results_files:
        # Rewrite the CMEC JSON file for compatibility
        with open(results_file) as f:
            results = json.load(f)
            results_transformed = _transform_results(results)

        # Get the stem (filename without extension)
        stem = results_file.stem

        # Create the new filename
        results_file_transformed = results_file.with_name(f"{stem}_transformed.json")

        with open(results_file_transformed, "w") as f:
            # Write the transformed executions back to the file
            json.dump(results_transformed, f, indent=4)
            logger.debug(f"Transformed executions written to {results_file_transformed}")

        transformed_results_files.append(results_file_transformed)

    return transformed_results_files


def _update_top_level_keys(combined_results: dict[str, Any], data: dict[str, Any], levels: list[str]) -> None:
    if "DIMENSIONS" not in data:
        data["DIMENSIONS"] = {}

    top_level_keys = list(data.keys())
    top_level_keys.remove("RESULTS")

    json_structure = data.get("DIMENSIONS", {}).get("json_structure", {})
    json_structure = ["level", *json_structure]

    for key in top_level_keys:
        combined_results[key] = data[key]
        if key == "Variable":
            combined_results[key]["level"] = levels
        elif key == "DIMENSIONS":
            combined_results[key]["json_structure"] = json_structure
            if "level" not in combined_results[key]:
                combined_results[key]["level"] = {}
                for level in levels:
                    combined_results[key]["level"][level] = {}


def combine_results_files(results_files: list[Any], output_directory: str | Path) -> Path:
    """
    Combine multiple results files into a single file.

    Parameters
    ----------
    results_files : list
        List of result files to combine.
    output_directory : str or Path
        Directory where the combined file will be saved.

    Returns
    -------
    Path, list[str]
        The path to the combined results file and a list of levels found in the results files.
    """
    combined_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    combined_results["RESULTS"] = {}
    levels = []

    # Ensure output_directory is a Path object
    if isinstance(output_directory, str):
        output_directory = Path(output_directory)

    last_data = None
    for file in results_files:
        with open(file) as f:
            data = json.load(f)
            last_data = data
            level_key = str(int(data["Variable"]["level"]))
            levels.append(level_key)
            logger.debug(f"Processing file: {file}, level_key: {level_key}")
            # Insert the results into the combined_results dictionary
            if level_key not in combined_results["RESULTS"]:
                combined_results["RESULTS"][level_key] = data.get("RESULTS", {})

    if last_data is not None:
        _update_top_level_keys(combined_results, last_data, levels)

    # Ensure the output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create the combined file path
    combined_file_path = output_directory / "combined_results.json"

    with open(combined_file_path, "w") as f:
        json.dump(combined_results, f, indent=4)

    # return combined_file_path, levels
    return combined_file_path


# ===================================================
# PMP diagnostics main class: annual cycle diagnostic
# ===================================================


def _transform_results(data: dict[str, Any]) -> dict[str, Any]:
    """
    Transform the executions dictionary to match the expected structure.

    Parameters
    ----------
    data : dict
        The original execution dictionary.

    Returns
    -------
    dict
        The transformed executions dictionary.
    """
    # Remove the model, reference, rip dimensions
    # These are later replaced with a REF-specific naming convention
    data = remove_dimensions(data, ["model", "reference", "rip"])

    # TODO: replace this with the ability to capture series
    # Remove the "CalendarMonths" key from the nested structure
    for region, region_values in data["RESULTS"].items():
        for stat, stat_values in region_values.items():
            if "CalendarMonths" in stat_values:
                stat_values.pop("CalendarMonths")

    # Remove the "CalendarMonths" key from the nested structure in "DIMENSIONS"
    data["DIMENSIONS"]["season"].pop("CalendarMonths")

    return data


def transform_results_files(results_files: list[Any]) -> list[Any]:
    """
    Transform the results files to match the expected structure.

    Parameters
    ----------
    results_files : list
        List of result files to transform.

    Returns
    -------
    list
        List of transformed result files.

    """
    if len(results_files) == 0:
        logger.warning("No results files provided for transformation.")
        return []

    transformed_results_files = []

    for results_file in results_files:
        # Rewrite the CMEC JSON file for compatibility
        with open(results_file) as f:
            results = json.load(f)
            results_transformed = _transform_results(results)

        # Get the stem (filename without extension)
        stem = results_file.stem

        # Create the new filename
        results_file_transformed = results_file.with_name(f"{stem}_transformed.json")

        with open(results_file_transformed, "w") as f:
            # Write the transformed executions back to the file
            json.dump(results_transformed, f, indent=4)
            logger.debug(f"Transformed executions written to {results_file_transformed}")

        transformed_results_files.append(results_file_transformed)

    return transformed_results_files


def _update_top_level_keys(combined_results: dict[str, Any], data: dict[str, Any], levels: list[str]) -> None:
    if "DIMENSIONS" not in data:
        data["DIMENSIONS"] = {}

    top_level_keys = list(data.keys())
    top_level_keys.remove("RESULTS")

    json_structure = data.get("DIMENSIONS", {}).get("json_structure", {})
    json_structure = ["level", *json_structure]

    for key in top_level_keys:
        combined_results[key] = data[key]
        if key == "Variable":
            combined_results[key]["level"] = levels
        elif key == "DIMENSIONS":
            combined_results[key]["json_structure"] = json_structure
            if "level" not in combined_results[key]:
                combined_results[key]["level"] = {}
                for level in levels:
                    combined_results[key]["level"][level] = {}


def combine_results_files(results_files: list[Any], output_directory: str | Path) -> Path:
    """
    Combine multiple results files into a single file.

    Parameters
    ----------
    results_files : list
        List of result files to combine.
    output_directory : str or Path
        Directory where the combined file will be saved.

    Returns
    -------
    Path, list[str]
        The path to the combined results file and a list of levels found in the results files.
    """
    combined_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    combined_results["RESULTS"] = {}
    levels = []

    # Ensure output_directory is a Path object
    if isinstance(output_directory, str):
        output_directory = Path(output_directory)

    last_data = None
    for file in results_files:
        with open(file) as f:
            data = json.load(f)
            last_data = data
            level_key = str(int(data["Variable"]["level"]))
            levels.append(level_key)
            logger.debug(f"Processing file: {file}, level_key: {level_key}")
            # Insert the results into the combined_results dictionary
            if level_key not in combined_results["RESULTS"]:
                combined_results["RESULTS"][level_key] = data.get("RESULTS", {})

    if last_data is not None:
        _update_top_level_keys(combined_results, last_data, levels)

    # Ensure the output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create the combined file path
    combined_file_path = output_directory / "combined_results.json"

    with open(combined_file_path, "w") as f:
        json.dump(combined_results, f, indent=4)

    # return combined_file_path, levels
    return combined_file_path


# ===================================================
# PMP diagnostics main class: annual cycle diagnostic
# ===================================================


class AnnualCycle(CommandLineDiagnostic):
    """
    Calculate the annual cycle for a dataset
    """

    name = "Annual Cycle"
    slug = "annual-cycle"
    facets = (
        "source_id",
        "member_id",
        "experiment_id",
        "variable_id",
        "reference_source_id",
        "region",
        "statistic",
        "season",
    )

    data_requirements = (
        # ERA-5 as reference dataset, spatial 2-D variables
        make_data_requirement("ts", "ERA-5"),
        make_data_requirement("uas", "ERA-5"),
        make_data_requirement("vas", "ERA-5"),
        make_data_requirement("psl", "ERA-5"),
        # ERA-5 as reference dataset, spatial 3-D variables
        make_data_requirement("ta", "ERA-5"),
        make_data_requirement("ua", "ERA-5"),
        make_data_requirement("va", "ERA-5"),
        make_data_requirement("zg", "ERA-5"),
        # Other reference datasets, spatial 2-D variables
        make_data_requirement("pr", "GPCP-Monthly-3-2"),
        make_data_requirement("rlds", "CERES-EBAF-4-2"),
        make_data_requirement("rlus", "CERES-EBAF-4-2"),
        make_data_requirement("rlut", "CERES-EBAF-4-2"),
        make_data_requirement("rsds", "CERES-EBAF-4-2"),
        make_data_requirement("rsdt", "CERES-EBAF-4-2"),
        make_data_requirement("rsus", "CERES-EBAF-4-2"),
        make_data_requirement("rsut", "CERES-EBAF-4-2"),
    )

    def __init__(self) -> None:
        self.parameter_file_1 = "pmp_param_annualcycle_1-clims.py"
        self.parameter_file_2 = "pmp_param_annualcycle_2-metrics.py"

    def build_cmds(self, definition: ExecutionDefinition) -> list[list[str]]:
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
        input_datasets = definition.datasets[SourceDatasetType.CMIP6]
        reference_datasets = definition.datasets[SourceDatasetType.PMPClimatology]

        source_id = input_datasets["source_id"].unique()[0]
        experiment_id = input_datasets["experiment_id"].unique()[0]
        member_id = input_datasets["member_id"].unique()[0]
        variable_id = input_datasets["variable_id"].unique()[0]

        model_files_raw = input_datasets.path.to_list()
        if len(model_files_raw) == 1:
            model_files = model_files_raw[0]  # If only one file, use it directly
        elif len(model_files_raw) > 1:
            model_files = build_glob_pattern(model_files_raw)  # If multiple files, build a glob pattern
        else:
            raise ValueError("No model files found")

        logger.debug("build_cmd start")

        logger.debug(f"input_datasets: {input_datasets}")
        logger.debug(f"input_datasets.keys(): {input_datasets.keys()}")

        reference_dataset_name = reference_datasets["source_id"].unique()[0]
        reference_dataset_path = reference_datasets.datasets.iloc[0]["path"]

        logger.debug(f"reference_dataset.datasets: {reference_datasets.datasets}")
        logger.debug(f"reference_dataset_name: {reference_dataset_name}")
        logger.debug(f"reference_dataset_path: {reference_dataset_path}")

        output_directory_path = str(definition.output_directory)

        cmds = []

        # ----------------------------------------------
        # PART 1: Build the command to get climatologies
        # ----------------------------------------------
        # Model
        data_name = f"{source_id}_{experiment_id}_{member_id}"
        data_path = model_files
        params = {
            "vars": variable_id,
            "infile": data_path,
            "outfile": f"{output_directory_path}/{variable_id}_{data_name}_clims.nc",
        }

        cmds.append(
            build_pmp_command(
                driver_file="pcmdi_compute_climatologies.py",
                parameter_file=self.parameter_file_1,
                **params,
            )
        )

        # --------------------------------------------------
        # PART 2: Build the command to calculate diagnostics
        # --------------------------------------------------
        # Reference
        obs_dict = {
            variable_id: {
                reference_dataset_name: {
                    "template": reference_dataset_path,
                },
                "default": reference_dataset_name,
            }
        }

        # Generate a JSON file based on the obs_dict
        with open(f"{output_directory_path}/obs_dict.json", "w") as f:
            json.dump(obs_dict, f)

        date = datetime.datetime.now().strftime("%Y%m%d")

        if variable_id in ["ua", "va", "ta"]:
            levels = ["200", "850"]
        elif variable_id in ["zg"]:
            levels = ["500"]
        else:
            levels = None

        variables = []
        if levels is not None:
            for level in levels:
                variable_id_with_level = f"{variable_id}-{level}"
                variables.append(variable_id_with_level)
        else:
            variables = [variable_id]

        logger.debug(f"variables: {variables}")
        logger.debug(f"levels: {levels}")

        # Build the command for each level
        params = {
            "vars": variables,
            "custom_observations": f"{output_directory_path}/obs_dict.json",
            "test_data_path": output_directory_path,
            "test_data_set": source_id,
            "realization": member_id,
            "filename_template": f"%(variable)_{data_name}_clims.198101-200512.AC.v{date}.nc",
            "metrics_output_path": output_directory_path,
            "cmec": "",
        }

        cmds.append(
            build_pmp_command(
                driver_file="mean_climate_driver.py",
                parameter_file=self.parameter_file_2,
                **params,
            )
        )

        logger.debug("build_cmd end")
        logger.debug(f"cmds: {cmds}")

        return cmds

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
        variable_id = input_datasets["variable_id"].unique()[0]

        if variable_id in ["ua", "va", "ta"]:
            variable_dir_pattern = f"{variable_id}-???"
        else:
            variable_dir_pattern = variable_id

        if variable_id in ["ua", "va", "ta"]:
            variable_dir_pattern = f"{variable_id}-???"
        else:
            variable_dir_pattern = variable_id

        results_directory = definition.output_directory
        png_directory = results_directory / variable_dir_pattern
        data_directory = results_directory / variable_dir_pattern
        png_directory = results_directory / variable_dir_pattern
        data_directory = results_directory / variable_dir_pattern

        logger.debug(f"results_directory: {results_directory}")
        logger.debug(f"png_directory: {png_directory}")
        logger.debug(f"data_directory: {data_directory}")

        # Find the CMEC JSON file(s)
        results_files = transform_results_files(list(results_directory.glob("*_cmec.json")))

        if len(results_files) == 1:
            # If only one file, use it directly
            results_file = results_files[0]
            logger.debug(f"results_file: {results_file}")
        elif len(results_files) > 1:
            logger.error(f"More than one cmec file found: {results_files}")
            results_file = combine_results_files(results_files, definition.output_directory)
        else:
            logger.error("Unexpected case: no cmec file found")
            return ExecutionResult.build_from_failure(definition)

        # Find the other outputs: PNG and NetCDF files
        png_files = list(png_directory.glob("*.png"))
        data_files = list(data_directory.glob("*.nc"))

        # Prepare the output bundles
        cmec_output_bundle, cmec_metric_bundle = process_json_result(results_file, png_files, data_files)

        # Add missing dimensions to the output
        input_selectors = input_datasets.selector_dict()
        reference_selectors = definition.datasets[SourceDatasetType.PMPClimatology].selector_dict()
        cmec_metric_bundle = cmec_metric_bundle.prepend_dimensions(
            {
                "source_id": input_selectors["source_id"],
                "member_id": input_selectors["member_id"],
                "experiment_id": input_selectors["experiment_id"],
                "variable_id": input_selectors["variable_id"],
                "reference_source_id": reference_selectors["source_id"],
            }
        )

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output_bundle,
            cmec_metric_bundle=cmec_metric_bundle,
        )

    def execute(self, definition: ExecutionDefinition) -> None:
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
        cmds = self.build_cmds(definition)

        runs = [self.provider.run(cmd) for cmd in cmds]
        logger.debug(f"runs: {runs}")
