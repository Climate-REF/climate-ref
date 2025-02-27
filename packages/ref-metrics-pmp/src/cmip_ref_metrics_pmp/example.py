import os
import subprocess
from pathlib import Path
from typing import Any

import xarray as xr
from pcmdi_metrics.io.base import download_sample_data_files

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import DataRequirement, Metric, MetricExecutionDefinition, MetricResult
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


def load_ref_data():
    # Load the reference data from a file
    # This is a placeholder function, replace with actual data loading logic

    # Define the content to be written to the file
    content = """https://pcmdiweb.llnl.gov/pss/pmpdata/
99c8691e0f615dc4d79b4fb5e926cc76  obs4MIPs_PCMDI_monthly/MOHC/HadISST-1-1/mon/ts/gn/v20210727/ts_mon_HadISST-1-1_PCMDI_gn_187001-201907.nc"""

    # Specify the file name
    file_name = "download.txt"

    # Write the content to the file
    with open(file_name, "w") as file:
        file.write(content)

    print(f"Content has been written to {file_name}")

    # This is where you will be downloading the sample_data
    ref_data_directory = "ref_data"

    # Let's download the files
    try:
        download_sample_data_files("download.txt", ref_data_directory)
        print("All files downloaded")
        return os.path.join(
            ref_data_directory,
            "obs4MIPs_PCMDI_monthly/MOHC/HadISST-1-1/mon/ts/gn/v20210727/ts_mon_HadISST-1-1_PCMDI_gn_187001-201907.nc",
        ), "HadISST-1-1"
    except Exception:
        print("Download failed")
        raise Exception("Download failed")


def calculate_annual_cycle(input_files: list[Path]) -> xr.Dataset:
    """
    Calculate the annual mean timeseries for a dataset.

    While this function is implemented here,
    in most cases the metric calculation will be in the underlying benchmarking package.
    How the metric is calculated is up to the provider.

    Parameters
    ----------
    input_files
        List of input files to calculate the annual mean timeseries.

        This dataset may consist of multiple data files.

    Returns
    -------
    :
        The annual mean timeseries of the dataset
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    xr_ds = xr.open_mfdataset(input_files, combine="by_coords", chunks=None, decode_times=time_coder)

    annual_mean = xr_ds.resample(time="YS").mean()
    return annual_mean.mean(dim=["lat", "lon"], keep_attrs=True)


def format_cmec_output_bundle(dataset: xr.Dataset) -> dict[str, Any]:
    """
    Create a simple CMEC output bundle for the dataset.

    Parameters
    ----------
    dataset
        Processed dataset

    Returns
    -------
        A CMEC output bundle ready to be written to disk
    """
    # TODO: Check how timeseries data are generally serialised
    cmec_output = {
        "DIMENSIONS": {
            "model": {dataset.attrs["source_id"]: {}},
            "region": {"global": {}},
            "metric": {"tas": {}},
            "json_structure": [
                "model",
                "region",
                "metric",
            ],
        },
        # Is the schema tracked?
        "SCHEMA": {
            "name": "CMEC-REF",
            "package": "example",
            "version": "v1",
        },
        "RESULTS": {
            dataset.attrs["source_id"]: {"global": {"tas": 0}},
        },
    }

    return cmec_output


class ExtratropicalModesOfVariability_PDO(Metric):
    """
    Calculate the annual cycle for a dataset
    """

    name = "PMP Extratropical modes of variability PDO"
    slug = "pmp-extratropical-modes-of-variability-pdo"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={"frequency": "mon", "experiment_id": ("historical"), "variable_id": "ts"}
                ),
                # Ignore some experiments because they are not relevant
                FacetFilter(facets={"experiment_id": ("amip")}, keep=False),
            ),
            # Add cell areas to the groups
            # constraints=(AddCellAreas(),),
            # Run the metric on each unique combination of model, variable, experiment, and variant
            group_by=("source_id", "variable_id", "experiment_id", "variant_label", "member_id"),
        ),
    )

    def run(self, definition: MetricExecutionDefinition) -> MetricResult:
        """
        Run a metric

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the metric

        Returns
        -------
        :
            The result of running the metric.
        """
        # This is where one would hook into however they want to run
        # their benchmarking packages.
        # cmec-driver, python calls, subprocess calls all would work

        input_datasets = definition.metric_dataset[SourceDatasetType.CMIP6]

        # NEED HELP: Retrive info
        # QUESTION: reference data?
        source_id = input_datasets["source_id"].unique()[0]
        member_id = input_datasets["member_id"].unique()[0]
        path_to_model_nc_file = input_datasets.path.to_list()  # this is going to be list of strings
        path_to_ref_nc_file, reference_data_name = load_ref_data()
        output_directory_path = definition.output_directory

        cmd = [
            "variability_modes.py",
            "-p",
            "param/pmp_param_MoV-PDO.py",
            "--modnames",
            source_id,
            "--realization",
            member_id,
            "--modpath",
            *path_to_model_nc_file,
            "--reference_data_path",
            path_to_ref_nc_file,
            "--reference_data_name",
            reference_data_name,
            "--results_dir",
            output_directory_path,
        ]

        # Run the command and capture the output
        proc = subprocess.run(
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

        # annual_mean_global_mean_timeseries = calculate_annual_cycle(input_files=input_datasets.path.to_list())

        # Expected outcome from the run: a JSON file and some PNG files
        # QUESTION: Do we expect the metric to return the output, or save it somewhere should be fine?

        # Load json as dict and return it
        # About png files --- talk to Min

        result_dict = SOMETHING()

        # the format function actually returns the metric bundle
        cmec_metric = format_cmec_output_bundle(annual_mean_global_mean_timeseries)
        CMECMetric.model_validate(cmec_metric)

        # create a empty CMEC output dictionary
        cmec_output = CMECOutput.create_template()
        CMECOutput.model_validate(cmec_output)

        # the cmec_output_bundle and cmec_metric_bundle are required keywords, cannot be omitted
        return MetricResult.build_from_output_bundle(
            definition,
            result_dict,
        )
