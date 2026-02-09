from pathlib import Path

import pandas
import xarray

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_esmvaltool.diagnostics.base import (
    ESMValToolDiagnostic,
    fillvalues_to_nan,
    get_cmip_source_type,
)
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class EquilibriumClimateSensitivity(ESMValToolDiagnostic):
    """
    Calculate the global mean equilibrium climate sensitivity for a dataset.
    """

    name = "Equilibrium Climate Sensitivity"
    slug = "equilibrium-climate-sensitivity"
    base_recipe = "recipe_ecs.yml"

    variables = (
        "rlut",
        "rsdt",
        "rsut",
        "tas",
    )
    experiments = (
        "abrupt-4xCO2",
        "piControl",
    )
    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": variables,
                            "experiment_id": experiments,
                            "table_id": "Amon",
                        },
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets(
                        "variable_id",
                        required_facets=variables,
                        group_by=("source_id", "member_id", "grid_label", "experiment_id"),
                    ),
                    RequireFacets(
                        "experiment_id",
                        required_facets=experiments,
                        group_by=("source_id", "member_id", "grid_label", "variable_id"),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                ),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": variables,
                            "experiment_id": experiments,
                            "frequency": "mon",
                        },
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets(
                        "variable_id",
                        required_facets=variables,
                        group_by=("source_id", "variant_label", "grid_label", "experiment_id"),
                    ),
                    RequireFacets(
                        "experiment_id",
                        required_facets=experiments,
                        group_by=("source_id", "variant_label", "grid_label", "variable_id"),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
    )
    facets = ("grid_label", "member_id", "source_id", "region", "metric")
    series = (
        SeriesDefinition(
            file_pattern="ecs/calculate/ecs_regression_*.nc",
            dimensions={
                "statistic": ("global annual mean anomaly of rtnt vs tas"),
            },
            values_name="rtnt_anomaly",
            index_name="tas_anomaly",
            attributes=[],
        ),
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Only run the diagnostic that computes ECS for a single model.
        recipe["diagnostics"] = {
            "ecs": {
                "description": "Calculate ECS.",
                "variables": {
                    "tas": {
                        "preprocessor": "spatial_mean",
                    },
                    "rtnt": {
                        "preprocessor": "spatial_mean",
                        "derive": True,
                    },
                },
                "scripts": {
                    "calculate": {
                        "script": "climate_metrics/ecs.py",
                        "calculate_mmm": False,
                    },
                },
            },
        }

        # Prepare updated datasets section in recipe. It contains two
        # datasets, one for the "abrupt-4xCO2" and one for the "piControl"
        # experiment.
        cmip_source = get_cmip_source_type(input_files)
        recipe_variables = dataframe_to_recipe(
            input_files[cmip_source],
            equalize_timerange=True,
        )
        recipe["datasets"] = recipe_variables["tas"]["additional_datasets"]

        # Remove keys from the recipe that are only used for YAML anchors
        keys_to_remove = [
            "CMIP5_RTMT",
            "CMIP6_RTMT",
            "CMIP5_RTNT",
            "CMIP6_RTNT",
            "ECS_SCRIPT",
            "SCATTERPLOT",
        ]
        for key in keys_to_remove:
            recipe.pop(key, None)

    @staticmethod
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        ecs_ds = xarray.open_dataset(result_dir / "work" / "ecs" / "calculate" / "ecs.nc")
        ecs = float(fillvalues_to_nan(ecs_ds["ecs"].values)[0])
        lambda_ds = xarray.open_dataset(result_dir / "work" / "ecs" / "calculate" / "lambda.nc")
        lambda_ = float(fillvalues_to_nan(lambda_ds["lambda"].values)[0])

        # Update the diagnostic bundle arguments with the computed diagnostics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            MetricCV.JSON_STRUCTURE.value: [
                "region",
                "metric",
            ],
            "region": {"global": {}},
            "metric": {"ecs": {}, "lambda": {}},
        }
        metric_args[MetricCV.RESULTS.value] = {
            "global": {
                "ecs": ecs,
                "lambda": lambda_,
            },
        }

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
