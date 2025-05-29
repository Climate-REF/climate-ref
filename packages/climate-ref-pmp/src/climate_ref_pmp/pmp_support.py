import json
from pathlib import Path
from typing import Any

from loguru import logger

from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
)
from climate_ref_core.pycmec.metric import remove_dimensions

# ================================================================================
# PMP diagnostics support functions, in particular for the annual cycle diagnostic
# ================================================================================


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
