from pathlib import Path

import pandas
import pandas as pd

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class ENSOBasicClimatology(ESMValToolDiagnostic):
    """
    Calculate the ENSO CLIVAR metrics - background climatology.
    """

    name = "ENSO Basic Climatology"
    slug = "enso-basic-climatology"
    base_recipe = "ref/recipe_enso_basicclimatology.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": ("pr", "tauu"),
                        "experiment_id": "historical",
                        "table_id": "Amon",
                    },
                )
                | FacetFilter(
                    facets={
                        "variable_id": "tos",
                        "experiment_id": "historical",
                        "table_id": "Omon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireFacets(
                    "variable_id",
                    (
                        "pr",
                        "tos",
                        "tauu",
                    ),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
            ),
        ),
    )
    facets = ()

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        recipe.pop("datasets")
        for diagnostic in recipe["diagnostics"].values():
            for variable in diagnostic["variables"].values():
                variable["additional_datasets"].extend(
                    recipe_variables[variable["short_name"]]["additional_datasets"]
                )


class ENSOCharacteristics(ESMValToolDiagnostic):
    """
    Calculate the ENSO CLIVAR metrics - basic ENSO characteristics.
    """

    name = "ENSO Characteristics"
    slug = "enso-characteristics"
    base_recipe = "ref/recipe_enso_characteristics.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "tos",
                        "experiment_id": "historical",
                        "table_id": "Omon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP6),
                RequireFacets("variable_id", ("tos", "areacello")),
            ),
        ),
    )
    facets = ("grid_label", "member_id", "source_id", "region", "metric")

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        recipe["datasets"] = recipe_variables["tos"]["additional_datasets"]
        # TODO: update the observational data requirement once available on ESGF.
        # Observations - use only one per run
        recipe["datasets"].append(
            # {
            #     "dataset": "NOAA-ERSSTv5",
            #     "version": "v5",
            #     "project": "OBS6",
            #     "type": "reanaly",
            #     "tier": 2,
            # }
            {
                "dataset": "TROPFLUX",
                "version": "v1",
                "project": "OBS6",
                "type": "reanaly",
                "tier": 2,
            }
        )

    @staticmethod
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        metrics = pd.read_csv(
            result_dir / "work" / "diagnostic_metrics" / "plot_script" / "matrix.csv",
            names=["dataset", "metric_name", "metric_value"],
        )

        # Update the diagnostic bundle arguments with the computed diagnostics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "region",
                "metric",
            ],
            "region": {"global": {}},
            "metric": {metric: {} for metric in metrics.metric_name},
        }
        metric_args[MetricCV.RESULTS.value] = {
            "global": {
                metric_name: metric_value
                for metric_name, metric_value in zip(
                    metrics.metric_name,
                    metrics.metric_value,
                )
            },
        }

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
