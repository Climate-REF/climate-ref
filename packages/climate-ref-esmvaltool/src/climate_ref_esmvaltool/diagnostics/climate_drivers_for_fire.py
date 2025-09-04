import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireFacets,
    RequireOverlappingTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


class ClimateDriversForFire(ESMValToolDiagnostic):
    """
    Calculate diagnostics regarding climate drivers for fire.
    """

    name = "Climate drivers for fire"
    slug = "climate-drivers-for-fire"
    base_recipe = "ref/recipe_ref_fire.yml"

    variables = (
        "cVeg",
        "hurs",
        "pr",
        "tas",
        "tasmax",
        "treeFrac",
        "vegFrac",
    )
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": variables,
                        "frequency": "mon",
                        "experiment_id": "historical",
                    }
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireFacets("variable_id", variables),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("sftlf", SourceDatasetType.CMIP6),
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
        dataset = recipe_variables["cVeg"]["additional_datasets"][0]
        dataset.pop("mip")
        dataset.pop("timerange")
        dataset["start_year"] = 2013
        dataset["end_year"] = 2014
        recipe["datasets"] = [dataset]
        recipe["diagnostics"]["fire_evaluation"]["scripts"]["fire_evaluation"]["remove_confire_files"] = True
