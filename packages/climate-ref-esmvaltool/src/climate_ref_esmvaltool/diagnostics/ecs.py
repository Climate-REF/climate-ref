from pathlib import Path

import pandas
import xarray

from climate_ref_core.constraints import (
    AddParentDataset,
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import (
    ESMValToolDiagnostic,
    fillvalues_to_nan,
    get_cmip_source_type,
)
from climate_ref_esmvaltool.recipe import dataframe_to_recipe, get_child_and_parent_dataset
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
                            "experiment_id": "abrupt-4xCO2",
                            "table_id": "Amon",
                        },
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    AddParentDataset.from_defaults(SourceDatasetType.CMIP6),
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireFacets(
                        "variable_id",
                        required_facets=variables,
                        group_by=("source_id", "member_id", "grid_label", "experiment_id"),
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
                            "branded_variable_name": (
                                "rlut_tavg-u-hxy-u",
                                "rsdt_tavg-u-hxy-u",
                                "rsut_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                            ),
                            "experiment_id": experiments,
                            "frequency": "mon",
                            "region": "glb",
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

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="cmip6",
                description="Test with CMIP6 data.",
                requests=(
                    CMIP6Request(
                        slug="cmip6",
                        facets={
                            "experiment_id": ["abrupt-4xCO2", "piControl"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", "rlut", "rsdt", "rsut", "tas"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                    ),
                ),
            ),
            TestCase(
                name="cmip7",
                description="Test with CMIP7 data.",
                requests=(
                    CMIP7Request(
                        slug="cmip7",
                        facets={
                            "experiment_id": ["abrupt-4xCO2", "piControl"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", "rlut", "rsdt", "rsut", "tas"],
                            "branded_variable_name": [
                                "areacella_ti-u-hxy-u",
                                "rlut_tavg-u-hxy-u",
                                "rsdt_tavg-u-hxy-u",
                                "rsut_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                    ),
                ),
            ),
        )
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
        if cmip_source == SourceDatasetType.CMIP6:
            df = input_files[SourceDatasetType.CMIP6]
            recipe["datasets"] = get_child_and_parent_dataset(
                df[df.variable_id == "tas"],
                parent_experiment="piControl",
                child_duration_in_years=150,
                parent_offset_in_years=0,
                parent_duration_in_years=150,
            )
        else:
            # CMIP7: use per-variable additional_datasets to preserve correct branding_suffix
            recipe_variables = dataframe_to_recipe(
                input_files[cmip_source],
                equalize_timerange=True,
            )
            recipe["datasets"] = []
            for var_name, var_settings in recipe["diagnostics"]["ecs"]["variables"].items():
                short_name = var_settings.get("short_name", var_name)
                if short_name in recipe_variables:
                    var_settings["additional_datasets"] = recipe_variables[short_name]["additional_datasets"]
                elif var_name == "rtnt":
                    # rtnt is derived from rlut, rsdt, rsut - use rlut's dataset
                    var_settings["additional_datasets"] = recipe_variables["rlut"]["additional_datasets"]

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
