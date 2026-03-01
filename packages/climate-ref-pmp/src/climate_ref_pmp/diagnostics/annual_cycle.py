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
from climate_ref_core.esgf import CMIP6Request, CMIP7Request, RegistryRequest
from climate_ref_core.pycmec.metric import remove_dimensions
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_pmp.pmp_driver import (
    build_glob_pattern,
    build_pmp_command,
    get_model_source_type,
    process_json_result,
)

# =================================================================
# PMP diagnostics support functions for the annual cycle diagnostic
# =================================================================

# CMIP7 branded variable names (from CMIP7 Data Request)
_BRANDED_VARIABLE_NAMES: dict[str, str] = {
    # Surface 2D variables
    "ts": "ts_tavg-u-hxy-u",
    "psl": "psl_tavg-u-hxy-u",
    "pr": "pr_tavg-u-hxy-u",
    "rlds": "rlds_tavg-u-hxy-u",
    "rlus": "rlus_tavg-u-hxy-u",
    "rlut": "rlut_tavg-u-hxy-u",
    "rsds": "rsds_tavg-u-hxy-u",
    "rsdt": "rsdt_tavg-u-hxy-u",
    "rsus": "rsus_tavg-u-hxy-u",
    "rsut": "rsut_tavg-u-hxy-u",
    # Near-surface height variables
    "uas": "uas_tavg-h10m-hxy-u",
    "vas": "vas_tavg-h10m-hxy-u",
    # 3D atmospheric variables on pressure levels
    "ta": "ta_tavg-p19-hxy-air",
    "ua": "ua_tavg-p19-hxy-air",
    "va": "va_tavg-p19-hxy-air",
    "zg": "zg_tavg-p19-hxy-air",
}


def make_data_requirement(
    variable_id: str,
    obs_source: str,
) -> tuple[tuple[DataRequirement, DataRequirement], ...]:
    """
    Create data requirements for the annual cycle diagnostic.

    Returns a pair of (obs, model) DataRequirement tuples for each supported
    source type (CMIP6 and CMIP7).

    Parameters
    ----------
    variable_id : str
        The variable ID to filter the data requirement.
    obs_source : str
        The observation source ID to filter the data requirement.

    Returns
    -------
    tuple[tuple[DataRequirement, DataRequirement], ...]
        A tuple of (obs, model) DataRequirement pairs, one per source type.
    """
    obs_requirement = DataRequirement(
        source_type=SourceDatasetType.PMPClimatology,
        filters=(FacetFilter(facets={"source_id": (obs_source,), "variable_id": (variable_id,)}),),
        group_by=("variable_id", "source_id"),
    )

    cmip6_filters = (
        FacetFilter(
            facets={
                "frequency": "mon",
                "experiment_id": ("amip", "historical", "hist-GHG"),
                "variable_id": (variable_id,),
            }
        ),
    )

    cmip7_filters = (
        FacetFilter(
            facets={
                "branded_variable": (_BRANDED_VARIABLE_NAMES[variable_id],),
                "experiment_id": ("amip", "historical", "hist-GHG"),
                "frequency": "mon",
                "region": "glb",
            }
        ),
    )

    cmip6_requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=cmip6_filters,
        group_by=("variable_id", "source_id", "experiment_id", "member_id", "grid_label"),
    )
    cmip7_requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP7,
        filters=cmip7_filters,
        group_by=("variable_id", "source_id", "experiment_id", "variant_label", "grid_label"),
    )

    return (
        (obs_requirement, cmip6_requirement),
        (obs_requirement, cmip7_requirement),
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


class AnnualCycle(CommandLineDiagnostic):
    """
    Calculate the annual cycle for a dataset
    """

    name = "Annual Cycle"
    slug = "annual-cycle"
    facets = (
        "mip_id",
        "source_id",
        "member_id",
        "experiment_id",
        "variable_id",
        "reference_source_id",
        "region",
        "statistic",
        "season",
    )

    _variable_obs_pairs = (
        # ERA-5 as reference dataset, spatial 2-D variables
        ("ts", "ERA-5"),
        ("uas", "ERA-5"),
        ("vas", "ERA-5"),
        ("psl", "ERA-5"),
        # ERA-5 as reference dataset, spatial 3-D variables
        ("ta", "ERA-5"),
        ("ua", "ERA-5"),
        ("va", "ERA-5"),
        ("zg", "ERA-5"),
        # Other reference datasets, spatial 2-D variables
        ("pr", "GPCP-Monthly-3-2"),
        ("rlds", "CERES-EBAF-4-2"),
        ("rlus", "CERES-EBAF-4-2"),
        ("rlut", "CERES-EBAF-4-2"),
        ("rsds", "CERES-EBAF-4-2"),
        ("rsdt", "CERES-EBAF-4-2"),
        ("rsus", "CERES-EBAF-4-2"),
        ("rsut", "CERES-EBAF-4-2"),
    )

    data_requirements = tuple(
        pair
        for variable_id, obs_source in _variable_obs_pairs
        for pair in make_data_requirement(variable_id, obs_source)
    )

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="cmip6-ts",
                description="Test with CMIP6 ts data and ERA-5 climatology",
                requests=(
                    RegistryRequest(
                        slug="annual-cycle-era5-ts",
                        registry_name="pmp-climatology",
                        facets={"variable_id": "ts", "source_id": "ERA-5"},
                    ),
                    CMIP6Request(
                        slug="annual-cycle-cmip6-ts",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "ts",
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                ),
            ),
            TestCase(
                name="cmip6-pr",
                description="Test with CMIP6 pr data and GPCP-Monthly-3-2 climatology. "
                "Produces double ITCZ pattern in the diagnostics.",
                requests=(
                    RegistryRequest(
                        slug="annual-cycle-gpcp-pr",
                        registry_name="pmp-climatology",
                        facets={"variable_id": "pr", "source_id": "GPCP-Monthly-3-2"},
                    ),
                    CMIP6Request(
                        slug="annual-cycle-cmip6-pr",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "pr",
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                ),
            ),
            TestCase(
                name="cmip7-ts",
                description="CMIP7 test case with converted historical ts from ACCESS-ESM1-5",
                requests=(
                    RegistryRequest(
                        slug="annual-cycle-era5-ts-cmip7",
                        registry_name="pmp-climatology",
                        facets={"variable_id": "ts", "source_id": "ERA-5"},
                    ),
                    CMIP7Request(
                        slug="annual-cycle-cmip7-ts",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "ts",
                            "branded_variable": "ts_tavg-u-hxy-u",
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

    def __init__(self) -> None:
        self.parameter_file_1 = "pmp_param_annualcycle_1-clims.py"
        self.parameter_file_2 = "pmp_param_annualcycle_2-metrics.py"

    def build_cmds(self, definition: ExecutionDefinition) -> list[list[str]]:  # noqa: PLR0915
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
        reference_datasets = definition.datasets[SourceDatasetType.PMPClimatology]

        source_id = input_datasets["source_id"].unique()[0]
        experiment_id = input_datasets["experiment_id"].unique()[0]
        variable_id = input_datasets["variable_id"].unique()[0]
        member_id = input_datasets[
            "variant_label" if model_source_type == SourceDatasetType.CMIP7 else "member_id"
        ].unique()[0]

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

        date_stamp = datetime.datetime.now().strftime("%Y%m%d")

        params = {
            "vars": variable_id,
            "infile": data_path,
            "outfile": f"{output_directory_path}/{variable_id}_{data_name}_clims.nc",
            "version": f"v{date_stamp}",
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
            "filename_template": f"%(variable)_{data_name}_clims.198101-200512.AC.v{date_stamp}.nc",
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
        model_source_type = get_model_source_type(definition)
        input_datasets = definition.datasets[model_source_type]
        variable_id = input_datasets["variable_id"].unique()[0]

        if variable_id in ["ua", "va", "ta"]:
            variable_dir_pattern = f"{variable_id}-???"
        else:
            variable_dir_pattern = variable_id

        results_directory = definition.output_directory
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
            logger.info(f"More than one cmec file found: {results_files}")
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
        member_id_col = "variant_label" if model_source_type == SourceDatasetType.CMIP7 else "member_id"
        reference_datasets = definition.datasets[SourceDatasetType.PMPClimatology]
        cmec_metric_bundle = cmec_metric_bundle.prepend_dimensions(
            {
                "source_id": input_datasets["source_id"].unique()[0],
                "member_id": input_datasets[member_id_col].unique()[0],
                "experiment_id": input_datasets["experiment_id"].unique()[0],
                "variable_id": input_datasets["variable_id"].unique()[0],
                "reference_source_id": reference_datasets["source_id"].unique()[0],
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
