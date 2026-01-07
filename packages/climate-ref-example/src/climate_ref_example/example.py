from pathlib import Path
from typing import Any

import xarray as xr

from climate_ref_core.constraints import AddSupplementaryDataset, RequireContiguousTimerange
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


def calculate_annual_mean_timeseries(input_files: list[Path]) -> xr.Dataset:
    """
    Calculate the annual mean timeseries for a dataset.

    While this function is implemented here,
    in most cases the diagnostic calculation will be in the underlying benchmarking package.
    How the diagnostic is calculated is up to the provider.

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

    return annual_mean.weighted(xr_ds.areacella.fillna(0)).mean(dim=["lat", "lon"], keep_attrs=True)


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
    # All keys listed in the sample are the CMEC keywords.
    # The value of diagnostics is the json file following the
    # CMEC diagnostic bundle standard. Only provenance is required,
    # others are optional

    # cmec_output = {
    #    "index": "index.html",
    #    "provenance": {
    #        "environment": {},
    #        "modeldata": [],
    #        "obsdata": {},
    #        "log": "cmec_output.log",
    #    },
    #    "data": {},
    #    "html": {},
    #    "metrics": {},
    #    "plots": {},
    # }
    # create_template will generate the same above dictionary
    cmec_output = CMECOutput.create_template()

    CMECOutput.model_validate(cmec_output)

    return cmec_output


def format_cmec_metric_bundle(dataset: xr.Dataset) -> dict[str, Any]:
    """
    Create a simple CMEC diagnostic bundle for the dataset.

    Parameters
    ----------
    dataset
        Processed dataset

    Returns
    -------
        A CMEC diagnostic bundle ready to be written to disk
    """
    # TODO: Check how timeseries data are generally serialised
    #
    # Only DIMENSIONS, json_structure, and RESULTS are the keywords,
    # other keys are derived from dimensions in json_structure and
    # the values of dictionaries in DIMENSIONS with the dimension
    # names as their keys. The order of keys of RESULTS shall
    # the order of their dimensions in the json_structure

    cmec_metric = {
        "DIMENSIONS": {
            "json_structure": [
                "region",
                "metric",
                "statistic",
            ],
            "region": {"global": {}},
            "metric": {"tas": {}, "pr": {}},
            "statistic": {"rmse": {}, "mb": {}},
        },
        "RESULTS": {
            "global": {"tas": {"rmse": 0, "mb": 0}, "pr": {"rmse": 0, "mb": 0}},
        },
    }

    CMECMetric.model_validate(cmec_metric)

    return cmec_metric


class GlobalMeanTimeseries(Diagnostic):
    """
    Calculate the annual mean global mean timeseries for a dataset
    """

    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": ("tas", "rsut")}),),
            # Run the diagnostic on each unique combination of model, variable, experiment, and variant
            group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
            constraints=(
                # Add cell areas to the groups
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                RequireContiguousTimerange(group_by=("instance_id",)),
            ),
        ),
    )
    facets = ("source_id", "variable_id", "experiment_id", "variant_label", "region", "metric", "statistic")

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Run a diagnostic

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic

        Returns
        -------
        :
            The result of running the diagnostic.
        """
        # This is where one would hook into however they want to run
        # their benchmarking packages.
        # cmec-driver, python calls, subprocess calls all would work

        input_datasets = definition.datasets[SourceDatasetType.CMIP6]

        result = calculate_annual_mean_timeseries(input_files=input_datasets.path.to_list())
        # Drop time_bnds to avoid xarray cftime bounds encoding regression (xarray >= 2025.11.0)
        if "time_bnds" in result:
            result = result.drop_vars("time_bnds")
        result.to_netcdf(definition.output_directory / "annual_mean_global_mean_timeseries.nc")

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Create a result object from the output of the diagnostic
        """
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(
            definition.output_directory / "annual_mean_global_mean_timeseries.nc", decode_times=time_coder
        )

        input_selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()

        cmec_metric_bundle = CMECMetric(**format_cmec_metric_bundle(ds)).prepend_dimensions(
            {
                "source_id": input_selectors["source_id"],
                "variable_id": input_selectors["variable_id"],
                "experiment_id": input_selectors["experiment_id"],
                "variant_label": input_selectors["variant_label"],
            }
        )

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=format_cmec_output_bundle(ds),
            cmec_metric_bundle=cmec_metric_bundle,
        )
