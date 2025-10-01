from pathlib import Path

import pandas
import xarray as xr

from climate_ref_core.constraints import (
    AddParentDataset,
    AddSupplementaryDataset,
    RequireContiguousTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, fillvalues_to_nan
from climate_ref_esmvaltool.recipe import get_child_and_parent_dataset
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class TransientClimateResponse(ESMValToolDiagnostic):
    """
    Calculate the global mean transient climate response for a dataset.
    """

    name = "Transient Climate Response"
    slug = "transient-climate-response"
    base_recipe = "recipe_tcr.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": ("tas",),
                        "experiment_id": "1pctCO2",
                        "table_id": "Amon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                AddParentDataset(),
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
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
        df = input_files[SourceDatasetType.CMIP6]
        recipe["datasets"] = get_child_and_parent_dataset(
            df[df.variable_id == "tas"],
            parent_experiment="piControl",
            child_duration_in_years=140,
            parent_offset_in_years=0,
            parent_duration_in_years=140,
        )

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
        tcr_ds = xr.open_dataset(result_dir / "work" / "tcr" / "calculate" / "tcr.nc")
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
