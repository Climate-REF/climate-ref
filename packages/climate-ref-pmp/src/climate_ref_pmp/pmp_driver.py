import importlib.metadata
import importlib.resources
import json
import os
import pathlib
from typing import Any

from loguru import logger
from rich.pretty import pretty_repr

from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


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
        tuple of CMEC output and diagnostic bundles
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
            "description": "Plot produced by the diagnostic",
        }
    for fname in data_files:
        cmec_output["data"][fname.name] = {
            "filename": str(fname),
            "long_name": "Output data",
            "description": "Data produced by the diagnostic",
        }

    cmec_metric = CMECMetric.create_template()
    cmec_metric["DIMENSIONS"] = {}
    dimensions = json_result["DIMENSIONS"]

    if "dimensions" in dimensions:  # pragma: no branch
        # Merge the contents of inner "dimensions" into the parent "DIMENSIONS"
        dimensions.update(dimensions["dimensions"])
        del dimensions["dimensions"]

    results = json_result["RESULTS"]

    cmec_metric["RESULTS"] = results
    cmec_metric["DIMENSIONS"] = dimensions

    if "provenance" in json_result:  # pragma: no branch
        cmec_metric["PROVENANCE"] = json_result["provenance"]

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


def build_pmp_command(
    driver_file: str,
    parameter_file: str,
    **kwargs: str | int | float | list[str] | None,
) -> list[str]:
    """
    Run a PMP driver script via a conda environment

    This function runs a PMP driver script using a specific conda environment.
    The driver script is responsible for running the PMP diagnostics and producing output.
    The output consists of a JSON file that contains the executions of the PMP diagnostics,
    and a set of PNG and data files that are produced by the diagnostics.

    The PMP driver scripts are installed in the PMP conda environment,
    but absolute paths should be used for non-PMP scripts.

    Parameters
    ----------
    driver_file
        Filename of the PMP driver script to run
    parameter_file
        Filename of the parameter file to use
    kwargs
        Additional arguments to pass to the driver script
    """
    # Note this uses the driver script from the REF env *not* the PMP conda env
    _parameter_file = _get_resource("climate_ref_pmp.params", parameter_file, use_resources=True)

    # Run the driver script inside the PMP conda environment
    cmd = [
        driver_file,
        "-p",
        _parameter_file,
    ]

    # Loop through additional arguments if they exist
    if kwargs:  # pragma: no cover
        for key, value in kwargs.items():
            if isinstance(value, list):
                cmd.extend([f"--{key}"] + [str(v) for v in value])
            elif value:
                cmd.extend([f"--{key}", str(value)])
            else:
                cmd.extend([f"--{key}"])

    logger.info(f"PMP Command: {cmd}")

    return cmd


def build_glob_pattern(paths: list[str]) -> str:
    """
    Generate a glob pattern that matches files based on common path, prefix, and suffix.

    Generate a glob pattern that matches all files in the given list of paths,
    based on their common directory, filename prefix, and suffix.

    Parameters
    ----------
    paths : list of str
        A list of full file paths. The paths should point to actual files,
        and should have enough similarity in their structure and naming
        to extract common patterns.

    Returns
    -------
    str
        A glob pattern string that can be used with `glob.glob(pattern, recursive=True)`
        to match all the provided files and others with the same structural pattern.

    Examples
    --------
    >>> paths = [
    ...     "/home/user/data/folder1/file1.txt",
    ...     "/home/user/data/folder1/file2.txt",
    ...     "/home/user/data/folder2/file3.txt",
    ... ]
    >>> pattern = build_glob_pattern(paths)
    >>> print(pattern)
    /home/user/data/**/file*.txt
    """
    if not paths:
        raise ValueError("The path list is empty.")

    # Find the common directory path
    common_path = os.path.commonpath(paths)

    # Extract filenames and parent directories
    filenames = [os.path.basename(path) for path in paths]
    dirnames = [os.path.dirname(path) for path in paths]
    same_dir = all(d == dirnames[0] for d in dirnames)

    # Helper to find common prefix
    def common_prefix(strings: list[str]) -> str:
        if not strings:
            return ""
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    break
        return prefix

    # Helper to find common suffix
    def common_suffix(strings: list[str]) -> str:
        reversed_strings = [s[::-1] for s in strings]
        reversed_suffix = common_prefix(reversed_strings)
        return reversed_suffix[::-1]

    prefix = common_prefix(filenames)
    suffix = common_suffix(filenames)

    # Use simpler pattern if all files are in the same directory
    if same_dir:
        pattern = os.path.join(dirnames[0], f"{prefix}*{suffix}")
    else:
        pattern = os.path.join(common_path, "**", f"{prefix}*{suffix}")

    return pattern
