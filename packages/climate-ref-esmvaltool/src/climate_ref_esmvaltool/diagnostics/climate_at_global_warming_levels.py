import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


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
                    },
                ),
            ),
            group_by=("experiment_id",),
            constraints=(
                RequireFacets("variable_id", variables),
                AddSupplementaryDataset(
                    supplementary_facets={"experiment_id": "historical"},
                    matching_facets=(
                        "source_id",
                        "member_id",
                        "grid_label",
                        "table_id",
                        "variable_id",
                    ),
                    optional_matching_facets=tuple(),
                ),
                RequireFacets("experiment_id", ("historical",)),
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
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
        # Set up the datasets
        diagnostics = recipe["diagnostics"]
        for diagnostic in diagnostics.values():
            diagnostic.pop("additional_datasets")
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        datasets = recipe_variables["tas"]["additional_datasets"]
        datasets = [ds for ds in datasets if ds["exp"] != "historical"]
        for dataset in datasets:
            dataset.pop("timerange")
            dataset["activity"] = ["CMIP", dataset["activity"]]
            dataset["exp"] = ["historical", dataset["exp"]]
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
