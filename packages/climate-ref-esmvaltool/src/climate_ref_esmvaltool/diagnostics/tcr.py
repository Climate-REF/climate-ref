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


class TransientClimateResponse(ESMValToolDiagnostic):
    """
    Calculate the global mean transient climate response for a dataset.
    """

    name = "Transient Climate Response"
    slug = "transient-climate-response"
    base_recipe = "recipe_tcr.yml"

    experiments = (
        "1pctCO2",
        "piControl",
    )
    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": ("tas",),
                            "experiment_id": experiments,
                            "table_id": "Amon",
                        },
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets("experiment_id", experiments),
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
                            "variable_id": ("tas",),
                            "experiment_id": experiments,
                            "frequency": "mon",
                        },
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets("experiment_id", experiments),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
    )
    facets = ("grid_label", "member_id", "source_id", "region", "metric")
    series = (
        SeriesDefinition(
            file_pattern="tcr/calculate/{source_id}*.nc",
            dimensions={
                "statistic": "global annual mean tas anomaly relative to linear fit of piControl run",
            },
            values_name="tas_anomaly",
            index_name="time",
            attributes=[],
        ),
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Only run the diagnostic that computes TCR for a single model.
        recipe["diagnostics"] = {
            "tcr": {
                "description": "Calculate TCR.",
                "variables": {
                    "tas": {
                        "preprocessor": "spatial_mean",
                    },
                },
                "scripts": {
                    "calculate": {
                        "script": "climate_metrics/tcr.py",
                        "calculate_mmm": False,
                    },
                },
            },
        }

        # Prepare updated datasets section in recipe. It contains two
        # datasets, one for the "1pctCO2" and one for the "piControl"
        # experiment.
        cmip_source = get_cmip_source_type(input_files)
        recipe_variables = dataframe_to_recipe(
            input_files[cmip_source],
            equalize_timerange=True,
        )
        recipe["datasets"] = recipe_variables["tas"]["additional_datasets"]

        # Remove keys from the recipe that are only used for YAML anchors
        keys_to_remove = [
            "TCR",
            "SCATTERPLOT",
            "VAR_SETTING",
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
        tcr_ds = xarray.open_dataset(result_dir / "work" / "tcr" / "calculate" / "tcr.nc")
        tcr = float(fillvalues_to_nan(tcr_ds["tcr"].values)[0])

        # Update the diagnostic bundle arguments with the computed diagnostics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "region",
                "metric",
            ],
            "region": {"global": {}},
            "metric": {"tcr": {}},
        }
        metric_args[MetricCV.RESULTS.value] = {
            "global": {
                "tcr": tcr,
            },
        }

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
