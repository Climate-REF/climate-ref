from pathlib import Path

import pandas as pd

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.metric_values.typing import FileDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class ClimateAtGlobalWarmingLevels(ESMValToolDiagnostic):
    """
    Calculate climate variables at global warming levels.
    """

    name = "Climate variables at global warming levels"
    slug = "climate-at-global-warming-levels"
    base_recipe = "recipe_calculate_gwl_exceedance_stats.yml"

    variables = (
        "pr",
        "tas",
    )

    matching_facets = (
        "source_id",
        "member_id",
        "grid_label",
        "table_id",
        "variable_id",
    )

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": variables,
                        "experiment_id": (
                            "ssp126",
                            "ssp245",
                            "ssp370",
                            "ssp585",
                        ),
                        "table_id": "Amon",
                    },
                ),
            ),
            group_by=("experiment_id",),
            constraints=(
                AddSupplementaryDataset(
                    supplementary_facets={"experiment_id": "historical"},
                    matching_facets=matching_facets,
                    optional_matching_facets=tuple(),
                ),
                RequireTimerange(
                    group_by=matching_facets,
                    start=PartialDateTime(year=1850, month=1),
                    end=PartialDateTime(year=2100, month=12),
                ),
                RequireFacets(
                    "experiment_id",
                    required_facets=("historical",),
                    group_by=matching_facets,
                ),
                RequireFacets(
                    "variable_id",
                    required_facets=variables,
                    group_by=("experiment_id", "source_id", "member_id", "grid_label", "table_id"),
                ),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )
    facets = ("experiment_id", "global warming level", "metric")

    files = tuple(
        FileDefinition(
            file_pattern=f"plots/gwl_mean_plots_{var_name}/plot_gwl_stats/*.png",
            dimensions={
                "statistic": "mean",
                "variable_id": var_name,
            },
        )
        for var_name in variables
    ) + tuple(
        FileDefinition(
            file_pattern=f"work/gwl_mean_plots_{var_name}/plot_gwl_stats/*.nc",
            dimensions={
                "statistic": "mean",
                "variable_id": var_name,
            },
        )
        for var_name in variables
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pd.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Set up the datasets
        diagnostics = recipe["diagnostics"]
        for diagnostic in diagnostics.values():
            diagnostic.pop("additional_datasets")
        recipe_variables = dataframe_to_recipe(
            input_files[SourceDatasetType.CMIP6],
            group_by=(
                "source_id",
                "member_id",
                "grid_label",
                "table_id",
                "variable_id",
            ),
        )
        datasets = recipe_variables["tas"]["additional_datasets"]
        datasets = [ds for ds in datasets if ds["exp"] != "historical"]
        for dataset in datasets:
            dataset.pop("timerange")
        recipe["datasets"] = datasets

        # Specify the timeranges
        diagnostics["calculate_gwl_exceedance_years"]["variables"]["tas_anomaly"] = {
            "short_name": "tas",
            "preprocessor": "calculate_anomalies",
            "timerange": "1850/2100",
        }

        diagnostics["gwl_mean_plots_tas"]["variables"]["tas"] = {
            "short_name": "tas",
            "preprocessor": "multi_model_gwl_stats",
            "timerange": "2000/2100",
        }

        diagnostics["gwl_mean_plots_pr"]["variables"]["pr"] = {
            "short_name": "pr",
            "preprocessor": "multi_model_gwl_stats",
            "timerange": "2000/2100",
        }

    @staticmethod
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "global warming level",
                "metric",
            ],
            "global warming level": {},
            "metric": {"exceedance_year": {}},
        }

        df = pd.read_csv(
            result_dir
            / "work"
            / "calculate_gwl_exceedance_years"
            / "gwl_exceedance_calculation"
            / "GWL_exceedance_years.csv"
        )
        for row in df.itertuples(index=False):
            gwl = str(row.GWL)
            if gwl not in metric_args[MetricCV.DIMENSIONS.value]["global warming level"]:
                metric_args[MetricCV.DIMENSIONS.value]["global warming level"][gwl] = {}
            metric_args[MetricCV.RESULTS.value][gwl] = {
                "exceedance_year": int(str(row.Exceedance_Year)),
            }

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
