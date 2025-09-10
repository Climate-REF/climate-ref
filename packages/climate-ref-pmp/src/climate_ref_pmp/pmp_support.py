import json
from pathlib import Path
from typing import Any

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


<<<<<<< HEAD
def _transform_results(data: dict[str, Any]) -> dict[str, Any]:
=======
def transform_results(data: dict[str, Any]) -> dict[str, Any]:
>>>>>>> f8ef48a5 (update)
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


<<<<<<< HEAD
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


=======
>>>>>>> f8ef48a5 (update)
def _insert_results(combined_results: dict[str, Any], data: dict[str, Any], level_key: str) -> None:
    for model, model_dict in data.get("RESULTS", {}).items():
        if model not in combined_results["RESULTS"]:
            combined_results["RESULTS"][model] = {}
        for reference, reference_dict in model_dict.items():
            if isinstance(reference_dict, dict):
                if reference not in combined_results["RESULTS"][model]:
                    combined_results["RESULTS"][model][reference] = {}
                for rip, rip_dict in reference_dict.items():
                    if rip not in ["source"]:
                        if rip not in combined_results["RESULTS"][model][reference]:
                            combined_results["RESULTS"][model][reference][rip] = {}
                        combined_results["RESULTS"][model][reference][rip][level_key] = rip_dict


def _update_top_level_keys(combined_results: dict[str, Any], data: dict[str, Any], levels: list[str]) -> None:
    top_level_keys = list(data.keys())
    top_level_keys.remove("RESULTS")
    for key in top_level_keys:
        combined_results[key] = data[key]
        if key == "Variable":
            combined_results[key]["level"] = ", ".join(levels)
        if key == "json_structure":
            combined_results[key] = ["model", "reference", "rip", "level", "region", "statistic", "season"]


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
    Any
        The path to the combined results file.
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
            print("level_key:", level_key)
            _insert_results(combined_results, data, level_key)

    if last_data is not None:
        _update_top_level_keys(combined_results, last_data, levels)

    # Ensure the output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create the combined file path
    combined_file_path = output_directory / "combined_results.json"

    with open(combined_file_path, "w") as f:
        json.dump(combined_results, f, indent=4)

    return combined_file_path
