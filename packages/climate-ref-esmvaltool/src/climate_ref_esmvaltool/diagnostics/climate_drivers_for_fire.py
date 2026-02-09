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


class ClimateDriversForFire(ESMValToolDiagnostic):
    """
    Calculate diagnostics regarding climate drivers for fire.
    """

    name = "Climate drivers for fire"
    slug = "climate-drivers-for-fire"
    base_recipe = "ref/recipe_ref_fire.yml"

    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        {
                            "variable_id": ("hurs", "pr", "tas", "tasmax"),
                            "experiment_id": "historical",
                            "table_id": "Amon",
                        }
                    ),
                    FacetFilter(
                        {
                            "variable_id": ("cVeg", "treeFrac"),
                            "experiment_id": "historical",
                            "table_id": "Lmon",
                        }
                    ),
                    FacetFilter(
                        {
                            "variable_id": "vegFrac",
                            "experiment_id": "historical",
                            "table_id": "Emon",
                        }
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(2013, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    AddSupplementaryDataset.from_defaults("sftlf", SourceDatasetType.CMIP6),
                    RequireFacets(
                        "variable_id",
                        (
                            "cVeg",
                            "hurs",
                            "pr",
                            "tas",
                            "tasmax",
                            "sftlf",
                            "treeFrac",
                            "vegFrac",
                        ),
                    ),
                ),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        {
                            "variable_id": ("hurs", "pr", "tas", "tasmax"),
                            "experiment_id": "historical",
                            "frequency": "mon",
                        }
                    ),
                    FacetFilter(
                        {
                            "variable_id": ("cVeg", "treeFrac"),
                            "experiment_id": "historical",
                            "frequency": "mon",
                        }
                    ),
                    FacetFilter(
                        {
                            "variable_id": "vegFrac",
                            "experiment_id": "historical",
                            "frequency": "mon",
                        }
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(2013, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    AddSupplementaryDataset.from_defaults("sftlf", SourceDatasetType.CMIP7),
                    RequireFacets(
                        "variable_id",
                        (
                            "cVeg",
                            "hurs",
                            "pr",
                            "tas",
                            "tasmax",
                            "sftlf",
                            "treeFrac",
                            "vegFrac",
                        ),
                    ),
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
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
        dataset = recipe_variables["cVeg"]["additional_datasets"][0]
        dataset.pop("mip")
        dataset.pop("timerange")
        dataset["start_year"] = 2013
        dataset["end_year"] = 2014
        recipe["datasets"] = [dataset]
        recipe["diagnostics"]["fire_evaluation"]["scripts"]["fire_evaluation"]["remove_confire_files"] = True
