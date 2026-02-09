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
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


class CloudRadiativeEffects(ESMValToolDiagnostic):
    """
    Plot climatologies and zonal mean profiles of cloud radiative effects (sw + lw) for a dataset.
    """

    name = "Climatologies and zonal mean profiles of cloud radiative effects"
    slug = "cloud-radiative-effects"
    base_recipe = "ref/recipe_ref_cre.yml"

    variables = (
        "rlut",
        "rlutcs",
        "rsut",
        "rsutcs",
    )
    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": variables,
                            "experiment_id": "historical",
                            "table_id": "Amon",
                        }
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1996, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets("variable_id", variables),
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
                            "experiment_id": "historical",
                            "frequency": "mon",
                        }
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1996, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets("variable_id", variables),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
        # TODO: Use CERES-EBAF, ESACCI-CLOUD, and ISCCP-FH from obs4MIPs once available.
    )

    facets = ()
    series = tuple(
        SeriesDefinition(
            file_pattern=f"plot_profiles/plot/variable_vs_lat_{var_name}_*.nc",
            sel={"dim0": 0},  # Select the model.
            dimensions={"variable_id": var_name, "statistic": "zonal mean"},
            values_name=var_name,
            index_name="lat",
            attributes=[],
        )
        for var_name in ["lwcre", "swcre"]
    ) + tuple(
        SeriesDefinition(
            file_pattern=f"plot_profiles/plot/variable_vs_lat_{var_name}_*.nc",
            sel={"dim0": i},  # Select the observation.
            dimensions={"variable_id": var_name, "statistic": "zonal mean", "reference_source_id": source_id},
            values_name=var_name,
            index_name="lat",
            attributes=[],
        )
        for var_name in ["lwcre", "swcre"]
        for i, source_id in enumerate(
            ["CERES-EBAF-Ed4.2", "ESACCI-CLOUD-AVHRR-AMPM-fv3.0", "ISCCP-FH"], start=1
        )
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: dict[SourceDatasetType, pandas.DataFrame]) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
        recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}

        datasets = recipe_variables["rsut"]["additional_datasets"]
        for dataset in datasets:
            dataset.pop("timerange")
        recipe["datasets"] = datasets
