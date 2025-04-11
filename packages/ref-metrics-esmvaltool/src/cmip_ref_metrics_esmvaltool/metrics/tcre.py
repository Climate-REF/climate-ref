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


class TransientClimateResponseEmissions(ESMValToolMetric):
    """
    Calculate the global mean Transient Climate Response to Cumulative CO2 Emissions.
    """

    name = "Transient Climate Response to Cumulative CO2 Emissions"
    slug = "esmvaltool-transient-climate-response-emissions"
    base_recipe = "recipe_tcre.yml"

    experiments = (
        "esm-1pctCO2",
        "esm-piControl",
    )
    variables = (
        "tas",
        "fco2antt",
    )
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": variables,
                        "frequency": "mon",
                        "experiment_id": experiments,
                    },
                ),
                FacetFilter(
                    facets={
                        "variable_id": "fco2antt",
                        "experiment_id": "esm-piControl",
                    },
                    keep=False,
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireFacets("experiment_id", experiments),
                RequireFacets("variable_id", variables),
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: pandas.DataFrame) -> None:
        """Update the recipe."""
        # Prepare updated datasets section in recipe. It contains three
        # datasets, "tas" and "fco2antt" for the "esm-1pctCO2" and just "tas"
        # for the "esm-piControl" experiment.
        recipe_variables = dataframe_to_recipe(input_files)
        tas_esm_1pctCO2 = next(
            ds for ds in recipe_variables["tas"]["additional_datasets"] if ds["exp"] == "esm-1pctCO2"
        )
        fco2antt_esm_1pctCO2 = next(
            ds for ds in recipe_variables["fco2antt"]["additional_datasets"] if ds["exp"] == "esm-1pctCO2"
        )
        tas_esm_piControl = next(
            ds for ds in recipe_variables["tas"]["additional_datasets"] if ds["exp"] == "esm-piControl"
        )
        tas_esm_piControl["timerange"] = tas_esm_1pctCO2["timerange"]

        recipe["diagnostics"]["tcre"]["variables"] = {
            "tas_esm-1pctCO2": {
                "short_name": "tas",
                "preprocessor": "global_annual_mean_anomaly",
                "additional_datasets": [tas_esm_1pctCO2],
            },
            "tas_esm-piControl": {
                "short_name": "tas",
                "preprocessor": "global_annual_mean_anomaly",
                "additional_datasets": [tas_esm_piControl],
            },
            "fco2antt": {
                "preprocessor": "global_cumulative_sum",
                "additional_datasets": [fco2antt_esm_1pctCO2],
            },
        }
        recipe["diagnostics"].pop("barplot")

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

        tcre_ds = xarray.open_dataset(result_dir / "work" / "tcre" / "calculate_tcre" / "tcre.nc")
        tcre = float(tcre_ds["tcre"].values[0])

        # Update the metric bundle arguments with the computed metrics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "source_id",
                "region",
                "metric",
            ],
            "source_id": {source_id: {}},
            "region": {"global": {}},
            "metric": {"tcre": {}},
        }
        metric_args[MetricCV.RESULTS.value] = {
            source_id: {
                "global": {
                    "tcre": tcre,
                },
            },
        }
        return metric_args, output_args
