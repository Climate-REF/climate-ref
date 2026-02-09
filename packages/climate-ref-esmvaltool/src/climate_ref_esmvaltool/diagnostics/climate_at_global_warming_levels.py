import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
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

    matching_facets = (
        "source_id",
        "member_id",
        "grid_label",
        "table_id",
        "variable_id",
    )

    cmip7_matching_facets = (
        "source_id",
        "variant_label",
        "grid_label",
        "variable_id",
    )

    data_requirements = (
        (
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
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": variables,
                            "experiment_id": (
                                # TODO: Redetermine the scenario naming for CMIP7 and update these accordingly
                            ),
                            "frequency": "mon",
                        },
                    ),
                ),
                group_by=("experiment_id",),
                constraints=(
                    AddSupplementaryDataset(
                        supplementary_facets={"experiment_id": "historical"},
                        matching_facets=cmip7_matching_facets,
                        optional_matching_facets=tuple(),
                    ),
                    RequireTimerange(
                        group_by=cmip7_matching_facets,
                        start=PartialDateTime(year=1850, month=1),
                        end=PartialDateTime(year=2100, month=12),
                    ),
                    RequireFacets(
                        "experiment_id",
                        required_facets=("historical",),
                        group_by=cmip7_matching_facets,
                    ),
                    RequireFacets(
                        "variable_id",
                        required_facets=variables,
                        group_by=("experiment_id", "source_id", "variant_label", "grid_label"),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
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
        cmip_source = get_cmip_source_type(input_files)
        diagnostics = recipe["diagnostics"]
        for diagnostic in diagnostics.values():
            diagnostic.pop("additional_datasets")
        group_by: tuple[str, ...]
        if cmip_source == SourceDatasetType.CMIP7:
            group_by = (
                "source_id",
                "variant_label",
                "grid_label",
                "variable_id",
            )
        else:
            group_by = (
                "source_id",
                "member_id",
                "grid_label",
                "table_id",
                "variable_id",
            )
        recipe_variables = dataframe_to_recipe(
            input_files[cmip_source],
            group_by=group_by,
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
