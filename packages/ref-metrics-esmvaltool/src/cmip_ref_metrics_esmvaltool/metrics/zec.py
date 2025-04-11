from pathlib import Path

import pandas
import xarray

from cmip_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
)
from cmip_ref_core.datasets import FacetFilter, MetricDataset, SourceDatasetType
from cmip_ref_core.metrics import DataRequirement
from cmip_ref_core.pycmec.metric import MetricCV
from cmip_ref_metrics_esmvaltool.metrics.base import ESMValToolMetric
from cmip_ref_metrics_esmvaltool.recipe import dataframe_to_recipe
from cmip_ref_metrics_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class ZeroEmissionCommitment(ESMValToolMetric):
    """
    Calculate the global mean Zero Emission Commitment (ZEC) temperature.
    """

    name = "Zero Emission Commitment"
    slug = "esmvaltool-zero-emission-commitment"
    base_recipe = "recipe_zec.yml"

    experiments = (
        "1pctCO2",
        "esm-1pct-brch-1000PgC",
    )
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": ("tas",),
                        "experiment_id": experiments,
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireFacets("experiment_id", experiments),
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: pandas.DataFrame) -> None:
        """Update the recipe."""
        # Prepare updated datasets section in recipe. It contains two
        # datasets, one for the "esm-1pct-brch-1000PgC" and one for the "piControl"
        # experiment.
        datasets = dataframe_to_recipe(input_files)["tas"]["additional_datasets"]
        base_dataset = next(ds for ds in datasets if ds["exp"] == "1pctCO2")
        dataset = next(ds for ds in datasets if ds["exp"] == "esm-1pct-brch-1000PgC")
        start = dataset["timerange"].split("/")[0]
        base_start = f"{int(start[:4]) - 10:04d}{start[4:]}"
        base_end = f"{int(start[:4]) + 10:04d}{start[4:]}"
        base_dataset["timerange"] = f"{base_start}/{base_end}"
        variables = recipe["diagnostics"]["zec"]["variables"]
        variables["tas_base"] = {
            "short_name": "tas",
            "preprocessor": "anomaly_base",
            "additional_datasets": [base_dataset],
        }
        variables["tas"] = {
            "preprocessor": "spatial_mean",
            "additional_datasets": [dataset],
        }

    @staticmethod
    def format_result(
        result_dir: Path,
        metric_dataset: MetricDataset,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[MetricBundleArgs, OutputBundleArgs]:
        """Format the result."""
        input_files = next(c.datasets for _, c in metric_dataset.items())
        source_id = input_files.iloc[0].source_id

        zec_ds = xarray.open_dataset(result_dir / "work" / "zec" / "zec" / "zec_50.nc")
        zec = float(zec_ds["zec"].values[0])

        # Update the metric bundle arguments with the computed metrics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "source_id",
                "region",
                "metric",
            ],
            "source_id": {source_id: {}},
            "region": {"global": {}},
            "metric": {"zec": {}},
        }
        metric_args[MetricCV.RESULTS.value] = {
            source_id: {
                "global": {
                    "zec": zec,
                },
            },
        }

        return metric_args, output_args
