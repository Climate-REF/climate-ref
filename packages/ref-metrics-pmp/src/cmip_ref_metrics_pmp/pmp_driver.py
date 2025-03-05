import importlib.metadata
import importlib.resources
import pathlib
import subprocess

DEFAULT_CONDA_ENV = "ref-metrics-pmp"


def get_resource_filename(package: str, resource_name: str | pathlib.Path, use_resources: bool) -> str:
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
        resource_path = str(importlib.resources.path(package, resource_name))
    else:
        distribution = importlib.metadata.distribution(package)
        resource_path = str(distribution.locate_file(pathlib.Path(package) / resource_name))
    if not pathlib.Path(resource_path).exists():
        raise FileNotFoundError(f"Resource {resource_name} not found in {package} package.")
    return str(resource_path)


def execute_pmp_driver(  # noqa: PLR0913
    driver_file: str,
    parameter_file: str,
    model_files: list[str],
    reference_name: str,
    reference_paths: list[str],
    source_id: str,
    member_id: str,
    output_directory_path: str,
    conda_env_name: str = DEFAULT_CONDA_ENV,
) -> None:
    """
    Run a PMP driver script via a conda environment

    Parameters
    ----------
    driver_file
    parameter_file
    model_files
    reference_name
    reference_paths
    source_id
    member_id
    output_directory_path
    conda_env_name
        Name of the conda environment to use to execute.

        This conda env must have pcmdi_metrics installed.

    """
    # Note this uses the driver script from the REF env *not* the PMP conda env
    _driver_script = get_resource_filename("pcmdi_metrics", driver_file, use_resources=False)
    _parameter_file = get_resource_filename("cmip_ref_metrics_pmp.params", parameter_file, use_resources=True)

    if len(model_files) != 1:
        # Have some logic to replace the dates in the filename with a wildcard
        raise NotImplementedError("Only one model file is supported at this time.")

    # Run the driver script inside the PMP conda environment
    cmd = [
        "conda",
        "run",
        "--name",
        conda_env_name,
        "python",
        _driver_script,
        "-p",
        _parameter_file,
        "--modnames",
        source_id,
        "--realization",
        member_id,
        "--modpath",
        *[str(p) for p in model_files],
        "--reference_data_path",
        *[str(p) for p in reference_paths],
        "--reference_data_name",
        reference_name,
        "--results_dir",
        output_directory_path,
    ]

    # Run the command and capture the output
    proc = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    # Print the command output
    print("Output:\n", proc.stdout)
    # Print any errors
    if proc.stderr:
        print("Error:\n", proc.stderr)

    # TODO: Not sure what you want to return here?
    # Maybe a boolean indicating success + the log output?
