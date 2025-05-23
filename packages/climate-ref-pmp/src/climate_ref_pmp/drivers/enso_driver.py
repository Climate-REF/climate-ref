"""
ENSO diagnostic driver

This script runs inside the PMP conda environment due to the use of xcdat.
"""

import argparse
import copy
import json
import os
from collections import defaultdict

import xcdat as xc

from pcmdi_metrics import resources  # isort:skip
from pcmdi_metrics.enso.lib import metrics_to_json  # isort:skip
from pcmdi_metrics.io import StringConstructor  # isort:skip
from pcmdi_metrics.utils import create_land_sea_mask

from EnsoMetrics.EnsoCollectionsLib import defCollection  # isort:skip
from EnsoMetrics.EnsoComputeMetricsLib import ComputeCollection  # isort:skip
from EnsoPlots.EnsoMetricPlot import main_plotter  # isort:skip


def main():
    """
    Run the ENSO metrics collection and plotting.

    This script is designed to be run from the command line.
    It takes two command line arguments:
    1. input_json_path: Path to the JSON file containing the datasets.
    2. output_directory: Directory where the output files will be saved.
    """
    print("### PMP ENSO: Compute the metric collection ###\n")

    args = parse_arguments()
    dict_datasets, mod, run, pattern = prepare_datasets(args)
    dict_metric, dict_dive = compute_metrics(args, dict_datasets, mod, run, pattern)
    save_metrics_to_json(args, dict_datasets, dict_metric, dict_dive, pattern)
    plot_results(args, pattern, mod, run)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="A script that takes two inputs and processes them.")
    parser.add_argument("--metrics_collection", type=str, help="metrics collection")
    parser.add_argument("--experiment_id", type=str, help="experiment id")
    parser.add_argument("--input_json_path", type=str, help="JSON file path")
    parser.add_argument("--output_directory", type=str, help="output directory")
    return parser.parse_args()


def prepare_datasets(args):
    """Prepare datasets and update them with land-sea masks."""
    os.makedirs(args.output_directory, exist_ok=True)
    with open(args.input_json_path) as f:
        dict_datasets = json.load(f)
    mod_run = next(iter(dict_datasets["model"].keys()))
    mod, run = mod_run.split("_")
    pattern = f"{args.metrics_collection}_{mod}_{args.experiment_id}_{run}"
    dict_datasets = update_dict_datasets(dict_datasets, os.path.join(args.output_directory, "ref_landmask"))
    # Write a JSON file for dict_datasets
    json_file = os.path.join(args.output_directory, f"input_{pattern}_processed.json")
    with open(json_file, "w") as f:
        json.dump(dict_datasets, f, indent=4)
    print(f"JSON file created: {json_file}")
    return dict_datasets, mod, run, pattern


def compute_metrics(args, dict_datasets, mod, run, pattern):
    """Compute the metric collection."""
    dict_metric = defaultdict(dict)
    dict_dive = defaultdict(dict)
    metrics, dive_results = ComputeCollection(
        args.metrics_collection,
        dict_datasets,
        f"{mod}_{run}",
        netcdf=True,
        netcdf_name=os.path.join(args.output_directory, pattern),
        obs_interpreter=True,
        debug=True,
    )
    dict_metric[mod][run] = metrics
    dict_dive[mod][run] = dive_results
    return dict_metric, dict_dive


def save_metrics_to_json(args, dict_datasets, dict_metric, dict_dive, pattern):
    """Save metrics to a JSON file."""
    egg_pth = resources.resource_path()
    dict_obs = dict_datasets["observations"]
    # pattern = f"{args.metrics_collection}_{mod}_{args.experiment_id}_{run}"
    mod = pattern.split("_")[-3]
    run = pattern.split("_")[-1]
    metrics_to_json(
        args.metrics_collection,
        dict_obs,
        dict_metric,
        dict_dive,
        egg_pth,
        StringConstructor(args.output_directory),
        pattern,
        mod=mod,
        run=run,
    )
    # Write an additional JSON file for the results for CMEC standard
    json_file = os.path.join(args.output_directory, f"{pattern}.json")
    write_CMEC_json(json_file)


def write_CMEC_json(json_file):
    """
    Write the CMEC JSON file.

    Parameters
    ----------
    json_file : str
        Path to the input JSON file.
    """
    # Load the existing JSON file
    with open(json_file) as f:
        dict_data = json.load(f)

    # -----------------------------------------
    # Prepare components for the CMEC structure
    # -----------------------------------------
    metrics_dict = {}
    ref_datasets = []

    mod = next(iter(dict_data["RESULTS"]["model"].keys()))
    run = next(iter(dict_data["RESULTS"]["model"][mod].keys()))

    metrics = list(dict_data["RESULTS"]["model"][mod][run]["value"].keys())
    for metric in metrics:
        metrics_dict[metric] = {}
        ref_datasets.extend(list(dict_data["RESULTS"]["model"][mod][run]["value"][metric]["metric"].keys()))

    ref_datasets = list(set(ref_datasets))  # Remove duplicates

    ref_datasets_dict = {ref: {} for ref in ref_datasets}

    dimensions_dict = {
        "json_structure": ["model", "realization", "metric", "reference_datasets"],
        "model": {mod: {}},
        "realization": {run: {}},
        "metric": metrics_dict,
        "reference_datasets": ref_datasets_dict,
    }

    results_dict = {}
    results_dict[mod] = {}
    results_dict[mod][run] = {}

    for metric in metrics:
        results_dict[mod][run][metric] = {}
        ref_datasets = list(dict_data["RESULTS"]["model"][mod][run]["value"][metric]["metric"].keys())
        for ref_dataset in ref_datasets:
            value = dict_data["RESULTS"]["model"][mod][run]["value"][metric]["metric"][ref_dataset]["value"]
            results_dict[mod][run][metric][ref_dataset] = value

    # Check if the dimensions are in the JSON file
    if "provenance" in dict_data:
        provenance_dict = dict_data["provenance"]
    else:
        provenance_dict = {}

    # Check if the reference datasets are in the JSON file
    if "obs" in dict_data["RESULTS"]:
        ref_dict = dict_data["RESULTS"]["obs"]
    else:
        ref_dict = {}

    # -----------------------------------------------
    # Create a new dictionary with the CMEC structure
    # -----------------------------------------------
    cmec_dict = {
        "RESULTS": results_dict,
        "DIMENSIONS": dimensions_dict,
        "REF": ref_dict,
        "provenance": provenance_dict,
    }

    # ---------------------------------------
    # Write the new dictionary to a JSON file
    # ---------------------------------------
    json_cmec_file = json_file.replace(".json", "_cmec.json")

    with open(json_cmec_file, "w") as f:
        json.dump(cmec_dict, f, indent=4)


def plot_results(args, pattern, mod, run):
    """Plot the results."""
    mod_run = f"{mod}_{run}"
    with open(os.path.join(args.output_directory, f"{pattern}.json")) as ff:
        data_json = json.load(ff)["RESULTS"]["model"][mod][run]
    plot_enso(
        args.metrics_collection,
        mod_run,
        args.experiment_id,
        args.output_directory,
        data_json,
    )


def plot_enso(mc_name, mod_run, exp, path_work_dir, data_json):
    """
    Plot the ENSO metrics collection.

    Parameters
    ----------
    mc_name : str
        Name of the metrics collection.
    mod_run : str
        Model and run name, separated by an underscore.
        e.g., "ACCESS-CM2_r1i1p1f1".
    exp : str
        Experiment name.
    path_work_dir : str
        Path of directory that contains the input NetCDF files and used to save the output PNG files.
    data_json : dict
        Data loaded from the JSON file.
    """
    metrics = sorted(defCollection(mc_name)["metrics_list"].keys(), key=lambda v: v.upper())
    print(f"metrics: {metrics}")

    mod = mod_run.split("_")[0]
    run = mod_run.split("_")[1]

    pattern = "_".join([mc_name, mod, exp, run])

    for met in metrics:
        print(f"met: {met}")
        # get NetCDF file name
        filename_nc = os.path.join(path_work_dir, pattern + "_" + met + ".nc")
        print(f"filename_nc: {filename_nc}")
        if os.path.exists(filename_nc):
            # get diagnostic values for the given model and observations
            if mc_name == "ENSO_tel" and "Map" in met:
                dict_dia = data_json["value"][met + "Corr"]["diagnostic"]
                diagnostic_values = dict((key1, None) for key1 in dict_dia.keys())
                diagnostic_units = ""
            else:
                dict_dia = data_json["value"][met]["diagnostic"]
                diagnostic_values = dict((key1, dict_dia[key1]["value"]) for key1 in dict_dia.keys())
                diagnostic_units = data_json["metadata"]["metrics"][met]["diagnostic"]["units"]
            # get metric values computed with the given model and observations
            if mc_name == "ENSO_tel" and "Map" in met:
                list1, list2 = (
                    [met + "Corr", met + "Rmse"],
                    [
                        "diagnostic",
                        "metric",
                    ],
                )
                dict_met = data_json["value"]
                metric_values = dict(
                    (
                        key1,
                        {mod: [dict_met[su][ty][key1]["value"] for su, ty in zip(list1, list2)]},
                    )
                    for key1 in dict_met[list1[0]]["metric"].keys()
                )
                metric_units = [data_json["metadata"]["metrics"][su]["metric"]["units"] for su in list1]
            else:
                dict_met = data_json["value"][met]["metric"]
                metric_values = dict((key1, {mod: dict_met[key1]["value"]}) for key1 in dict_met.keys())
                metric_units = data_json["metadata"]["metrics"][met]["metric"]["units"]
            # figure name
            figure_name = "_".join([mc_name, mod, exp, run, met])
            print(f"figure_name: {figure_name}")

            main_plotter(
                mc_name,
                met,
                mod,
                exp,
                filename_nc,
                diagnostic_values,
                diagnostic_units,
                metric_values,
                metric_units,
                member=run,
                path_png=path_work_dir,
                name_png=figure_name,
            )

            print("figure plotting done")

        else:
            print(f"file not found: {filename_nc}")


def update_dict_datasets(dict_datasets: dict, output_dir: str = ".") -> dict:
    """
    Update the dictDatasets to include the land-sea mask and remap observation names.

    Parameters
    ----------
    dict_datasets : dict
        Dictionary containing datasets information.
    output_dir : str
        Directory where the land-sea mask will be saved.
        Default is the current directory.

    Returns
    -------
    dict
        Updated dictionary with land-sea mask and remapped observation names.

    Raises
    ------
    FileNotFoundError
        If the input file path is not valid.
    NotImplementedError
        If multiple paths are found for a dataset or if the path is not a string.
    """
    dict_datasets2 = copy.deepcopy(dict_datasets)
    data_types = dict_datasets.keys()  # ["model", "observations"]

    # Select only model and observations datasets
    data_types = [data_type for data_type in data_types if data_type in ["model", "observations"]]

    for data_type in data_types:
        datasets = dict_datasets[data_type].keys()
        for dataset in datasets:
            variables = dict_datasets[data_type][dataset].keys()
            for variable in variables:
                path = dict_datasets[data_type][dataset][variable]["path + filename"]

                # If path is a list and has one element, take it as a string,
                # otherwise raise notImplementedError
                if isinstance(path, list) and len(path) == 1:
                    path = copy.deepcopy(path[0])
                    dict_datasets2[data_type][dataset][variable]["path + filename"] = path
                elif isinstance(path, list) and len(path) > 1:
                    raise NotImplementedError(
                        f"Multiple paths found for {data_type} {dataset} {variable}: {path}"
                    )
                elif not isinstance(path, str):
                    raise NotImplementedError(
                        f"Path is not a string for {data_type} {dataset} {variable}: {path}"
                    )
                else:
                    dict_datasets2[data_type][dataset][variable]["path + filename"] = path

                # Check if the file exists
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")

                # Generate the landmask path regardless data_type is observation or model.
                if (
                    "path + filename_area" not in dict_datasets[data_type][dataset]
                    or "path + filename_landmask" not in dict_datasets[data_type][dataset]
                ):
                    # Generate it per variable as different variables may be on different grids.
                    path_landmask = generate_landmask_path(path, variable, output_dir=output_dir)

                    dict_datasets2[data_type][dataset][variable]["areaname"] = "areacella"
                    dict_datasets2[data_type][dataset][variable]["landmaskname"] = "sftlf"
                    dict_datasets2[data_type][dataset][variable]["path + filename_area"] = path_landmask
                    dict_datasets2[data_type][dataset][variable]["path + filename_landmask"] = path_landmask

                # Map variable names to ENSO package recognized names
                var_name_mapping = {"ts": "sst", "tauu": "taux"}
                var_name_key = var_name_mapping.get(variable, variable)

                # Update the variable name
                dict_datasets2[data_type][dataset][var_name_key] = dict_datasets2[data_type][dataset].pop(
                    variable
                )

            if data_type == "observations":
                # Mapping of old observation names to new ones recognized by the ENSO package
                observation_name_mapping = {
                    "GPCP-2-3": "GPCPv2.3",
                    "ERA-INT": "ERA-Interim",
                    "ERA-5": "ERA5",
                    "AVISO-1-0": "AVISO",
                    "TropFlux-1-0": "Tropflux",
                    "HadISST-1-1": "HadISST",
                }
                # Get the new name if it exists in the mapping, otherwise keep the original name
                dataset_name_key = observation_name_mapping.get(dataset, dataset)
                # Update the dictDatasets with the new name
                dict_datasets2[data_type][dataset_name_key] = dict_datasets2[data_type].pop(dataset)

    return dict_datasets2


def generate_landmask_path(file_path, var_name, output_dir=".", output_filename=None):
    """
    Generate the landmask path based on the given file path.

    Parameters
    ----------
    file_path : str
        Path to the input NetCDF file.
    var_name : str
        Variable name to be used for creating the land-sea mask.
    output_dir : str
        Directory where the land-sea mask will be saved.
        Default is the current directory.
    output_filename : str
        Name of the output land-sea mask file.
        If not provided, it will be generated based on the input file name.
        Default is None.

    Returns
    -------
    str
        Path to the generated land-sea mask file.

    Raises
    ------
    FileNotFoundError
        If the input file path is not valid.
    ValueError
        If the variable name is not valid.
    """
    # If file_path is a list, take the first element
    if isinstance(file_path, list):
        file_path = file_path[0]

    # Check if the file path is valid
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if the variable name is valid
    if not isinstance(var_name, str):
        raise ValueError(f"Invalid variable name: {var_name}")

    # Open the dataset using xcdat and create the land-sea mask
    ds = xc.open_dataset(file_path)
    mask = create_land_sea_mask(ds[var_name])

    # Name mask variable as 'sftlf'
    mask.name = "sftlf"

    # Check if the output directory exists, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the land-sea mask to a new NetCDF file
    if output_filename:
        landmask_filename = output_filename
    else:
        # Generate the filename based on the input file name
        landmask_filename = os.path.basename(file_path).replace(".nc", "_landmask.nc")

    landmask_path = os.path.join(output_dir, landmask_filename)
    mask.to_netcdf(landmask_path)

    return os.path.abspath(landmask_path)


if __name__ == "__main__":
    main()
