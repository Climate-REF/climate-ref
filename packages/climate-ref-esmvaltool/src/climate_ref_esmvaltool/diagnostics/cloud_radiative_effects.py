import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireOverlappingTimerange,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


class CloudRadiativeEffects(ESMValToolDiagnostic):
    """
    Plot climatologies and zonal mean profiles of cloud radiative effects (sw + lw) for a dataset.
    """

    name = "Climatologies and zonal mean profiles of cloud radiative effects"
    slug = "cloud-radiative-effects"
    base_recipe = "ref/recipe_ref_cre.yml"

    facets = ()

    variables = (
        "rlut",
        "rlutcs",
        "rsut",
        "rsutcs",
    )
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": variables,
                        "experiment_id": ("historical",),
                    }
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireFacets("variable_id", variables),
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
        # TODO: Use CERES-EBAF, ESACCI-CLOUD, and ISCCP-FH from obs4MIPs once available.
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: dict[SourceDatasetType, pandas.DataFrame]) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}

        datasets = recipe_variables["rsut"]["additional_datasets"]
        for dataset in datasets:
            dataset.pop("timerange")
        recipe["datasets"] = datasets
