from pathlib import Path
from typing import Any

import xarray as xr
from ilamb3.dataset import integrate_space  # type: ignore

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import DataRequirement, Metric, MetricExecutionDefinition, MetricResult


def calculate_global_mean_timeseries(input_files: list[Path]) -> xr.Dataset:
    """
    Calculate the global mean timeseries for a dataset.

    Parameters
    ----------
    input_files
        List of input files to calculate the annual mean timeseries.

    Returns
    -------
    :
        The annual mean timeseries of the dataset
    """
    ds = xr.open_mfdataset(input_files, combine="by_coords", chunks=None, use_cftime=True)
    mean: xr.Dataset = integrate_space(ds, "tas", mean=True).to_dataset()
    return mean


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
    cmec_output = {
        "DIMENSIONS": {
            "dimensions": {
                "source_id": {dataset.attrs["source_id"]: {}},
                "region": {"global": {}},
                "variable": {"tas": {}},
            },
            "json_structure": [
                "model",
                "region",
                "statistic",
            ],
        },
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


class GlobalMeanTimeseries(Metric):
    """
    Calculate the global mean timeseries for a dataset
    """

    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(facets={"variable_id": ("tas", "rsut")}),
                FacetFilter(facets={"experiment_id": ("1pctCO2-*", "hist-*")}, keep=False),
            ),
            group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
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
        input_datasets = definition.metric_dataset[SourceDatasetType.CMIP6]
        global_mean_timeseries = calculate_global_mean_timeseries(input_files=input_datasets.path.to_list())

        return MetricResult.build_from_output_bundle(
            definition, format_cmec_output_bundle(global_mean_timeseries)
        )
