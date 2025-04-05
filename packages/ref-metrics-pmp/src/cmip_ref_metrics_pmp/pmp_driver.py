import importlib.metadata
import importlib.resources
import json
import pathlib
from typing import Any

from loguru import logger
from rich.pretty import pretty_repr

from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


def _remove_nested_key(data: dict[str, Any], key: str) -> dict[str, Any]:
    """
    Remove a nested key from a dictionary

    Parameters
    ----------
    data
        Dictionary to remove the key from
    key
        Key to remove

    Returns
    -------
        The dictionary with the key removed
    """
    if key in data:
        data.pop(key)
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = _remove_nested_key(v, key)
    return data


def process_json_result(
    json_filename: pathlib.Path, png_files: list[pathlib.Path], data_files: list[pathlib.Path]
) -> tuple[CMECOutput, CMECMetric]:
    """
    Process a PMP JSON result into the appropriate CMEC bundles

    Parameters
    ----------
    json_filename
        Filename of the JSON file that is written out by PMP
    png_files
        List of PNG files to be included in the output
    data_files
        List of data files to be included in the output

    Returns
    -------
        tuple of CMEC output and metric bundles
    """
    with open(json_filename) as fh:
        json_result = json.load(fh)

    cmec_output = CMECOutput.create_template()
    cmec_output["provenance"] = {**cmec_output["provenance"], **json_result["provenance"]}

    # Add the plots and data files
    for fname in png_files:
        cmec_output["plots"][fname.name] = {
            "filename": str(fname),
            "long_name": "Plot",
            "description": "Plot produced by the metric",
        }
    for fname in data_files:
        cmec_output["data"][fname.name] = {
            "filename": str(fname),
            "long_name": "Output data",
            "description": "Data produced by the metric",
        }

    cmec_metric = CMECMetric.create_template()
    cmec_metric["DIMENSIONS"] = {}
    dimensions = json_result["DIMENSIONS"]

    if "dimensions" in dimensions:  # pragma: no branch
        # Merge the contents of inner "dimensions" into the parent "DIMENSIONS"
        dimensions.update(dimensions["dimensions"])
        del dimensions["dimensions"]

    if "statistic" in dimensions["json_structure"]:  # pragma: no branch
        dimensions["json_structure"].remove("statistic")
        dimensions.pop("statistic")

    # Remove the "attributes" key from the RESULTS
    # This isn't standard CMEC output, but it is what PMP produces
    results = json_result["RESULTS"]

    cmec_metric["RESULTS"] = results
    cmec_metric["DIMENSIONS"] = dimensions

    if "provenance" in json_result:  # pragma: no branch
        cmec_metric["provenance"] = json_result["provenance"]

    logger.info(f"cmec_output: {pretty_repr(cmec_output)}")
    logger.info(f"cmec_metric: {pretty_repr(cmec_metric)}")

    return CMECOutput(**cmec_output), CMECMetric(**cmec_metric)


def _get_resource(package: str, resource_name: str | pathlib.Path, use_resources: bool) -> str:
    """
    Get the path to a resource within the pcmdi_metric package without importing.

    Parameters
    ----------
    package: str
        Python package name if use_resources is True, otherwise the distribution name
        (the pypi package name).
    resource_name : str
        The resource path relative to the package.
    use_resources : bool
        If True, use the importlib.resources API, otherwise use importlib.metadata.

        importlib.resources is the preferred way to access resources because it handles
        packages which have been editably installed,
        but it implictly imports the package.

        Whereas `importlib.metadata` uses the package metadata to resolve the location of a resource.

    Returns
    -------
        The full path to the target resource.
    """
    if use_resources:
        resource_path = str(importlib.resources.files(package) / str(resource_name))
    else:
        distribution = importlib.metadata.distribution(package)
        resource_path = str(distribution.locate_file(pathlib.Path(package) / resource_name))
    if not pathlib.Path(resource_path).exists():
        raise FileNotFoundError(f"Resource {resource_name} not found in {package} package.")
    return str(resource_path)


def build_pmp_command(  # noqa: PLR0913
    driver_file: str,
    parameter_file: str,
    model_files: list[str] | str,
    reference_name: str,
    reference_paths: list[str] | str,
    source_id: str,
    experiment_id: str,
    member_id: str,
    output_directory_path: str,
    **kwargs: dict[str, str | int | float | list[str]],
) -> list[str]:
    """
    Run a PMP driver script via a conda environment

    This function runs a PMP driver script using a specific conda environment.
    The driver script is responsible for running the PMP metrics and producing output.
    The output consists of a JSON file that contains the results of the PMP metrics,
    and a set of PNG and data files that are produced by the metrics.

    Parameters
    ----------
    driver_file
        Filename of the PMP driver script to run

        Relative to the pcmdi_metrics package.
    parameter_file
        Filename of the parameter file to use

        Relative to the cmip_ref_metrics_pmp.params package.
    model_files
        A list of model files to evaluate

        Currently, we only support a single model file.
    reference_name
        Name of the reference dataset to use
    reference_paths
        Path to the reference dataset
    source_id
        Source ID of the model data
    member_id
        Member ID of the model data
    output_directory_path
        Output data where all the results are stored
    kwargs
        Additional arguments to pass to the driver script
    """
    # Note this uses the driver script from the REF env *not* the PMP conda env
    _driver_script = _get_resource("pcmdi_metrics", driver_file, use_resources=False)
    _parameter_file = _get_resource("cmip_ref_metrics_pmp.params", parameter_file, use_resources=True)

    if isinstance(model_files, list):
        modpath = " ".join([str(p) for p in model_files])
    else:
        modpath = model_files

    if isinstance(reference_paths, list):
        reference_data_path = " ".join([str(p) for p in reference_paths])
    else:
        reference_data_path = reference_paths

    if len(model_files) != 1:
        # Have some logic to replace the dates in the filename with a wildcard
        raise NotImplementedError("Only one model file is supported at this time.")

    # Run the driver script inside the PMP conda environment
    cmd = [
        "python",
        _driver_script,
        "-p",
        _parameter_file,
        "--modnames",
        source_id,
        "--exp",
        experiment_id,
        "--realization",
        member_id,
        "--modpath",
        modpath,
        "--reference_data_path",
        reference_data_path,
        "--reference_data_name",
        reference_name,
        "--results_dir",
        output_directory_path,
        "--cmec",
        "--no_provenance",
    ]

    # Loop through additional arguments if they exist
    if kwargs:  # pragma: no cover
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

    logger.info(f"pmp command to run: {cmd}")

    return cmd
