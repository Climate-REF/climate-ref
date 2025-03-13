from pathlib import Path
from typing import Any

import xarray as xr

from cmip_ref_core.constraints import AddSupplementaryDataset
from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import (
    DataRequirement,
    Metric,
    MetricExecutionDefinition,
    MetricExecutionResult,
)
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


def calculate_annual_mean_timeseries(input_files: list[Path]) -> xr.Dataset:
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
    return annual_mean.weighted(xr_ds.areacella).mean(dim=["lat", "lon"], keep_attrs=True)


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
    # The value of metrics is the json file following the
    # CMEC metric bundle standard. Only provenance is required,
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
    Create a simple CMEC metric bundle for the dataset.

    Parameters
    ----------
    dataset
        Processed dataset

    Returns
    -------
        A CMEC metric bundle ready to be written to disk
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
                "model",
                "region",
                "metric",
            ],
            "model": {dataset.attrs["source_id"]: {}},
            "region": {"global": {}},
            "metric": {"tas": {}, "pr": {}},
        },
        "RESULTS": {
            dataset.attrs["source_id"]: {"global": {"tas": 0, "pr": {"rmse": 0, "mb": 0}}},
        },
    }

    CMECMetric.model_validate(cmec_metric)

    return cmec_metric


class GlobalMeanTimeseries(Metric):
    """
    Calculate the annual mean global mean timeseries for a dataset
    """

    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(facets={"variable_id": ("tas", "rsut")}),
                # Ignore some experiments because they are not relevant
                FacetFilter(facets={"experiment_id": ("1pctCO2-*", "hist-*")}, keep=False),
            ),
            # Run the metric on each unique combination of model, variable, experiment, and variant
            group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
            constraints=(
                # Add cell areas to the groups
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )

    def run(self, definition: MetricExecutionDefinition) -> MetricExecutionResult:
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

        annual_mean_global_mean_timeseries = calculate_annual_mean_timeseries(
            input_files=input_datasets.path.to_list()
        )

        return MetricExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=format_cmec_output_bundle(annual_mean_global_mean_timeseries),
            cmec_metric_bundle=format_cmec_metric_bundle(annual_mean_global_mean_timeseries),
        )
