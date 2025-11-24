import pandas

from climate_ref_core.constraints import (
    PartialDateTime,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement

# from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


class O3LatTimeMapplot(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - zonal mean total column ozone vs. time.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-lat-time"
    base_recipe = "ref/recipe_ozone.yml"


class O3PolarCapTimeseriesSH(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - October SH polar mean (60S-85S) time series.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-sh-oct"
    base_recipe = "ref/recipe_ozone.yml"


class O3PolarCapTimeseriesNH(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - March NH polar mean (60N-85N) time series.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-nh-mar"
    base_recipe = "ref/recipe_ozone.yml"


class O3ZonalMeanProfiles(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - stratospheric zonal mean profiles.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-zonal"
    base_recipe = "ref/recipe_ozone.yml"


class O3LatMonthMapplot(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - zonal mean total column ozone vs. annual cycle plot.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-annual-cycle"
    base_recipe = "ref/recipe_ozone.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": ("o3", "ps"),
                        "experiment_id": "historical",
                        "table_id": "AERmon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                RequireFacets(
                    "variable_id",
                    (
                        "o3",
                        "ps",
                    ),
                ),
            ),
        ),
    )
    facets = ()

    #    series = (
    #        tuple(
    #            SeriesDefinition(
    #                file_pattern=f"lat_month_mapplot/plot/*.nc",
    #                dimensions=(
    #                    {
    #                        "statistic": (
    #                            f"Hovmoeller plot of the annual cycle of total "
    #                            f"column ozone (month vs latitude)"
    #                        ),
    #                    }
    #                    | ({} if source_id == "{source_id}" else {"reference_source_id": source_id})
    #                ),
    #                values_name="toz",
    #                index_name="lat",
    #                attributes=[],
    #            )
    #        )
    #    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        #        breakpoint()
        recipe.pop("datasets")
        recipe["diagnostics"].pop("polar_cap_time_series_SH")
        recipe["diagnostics"].pop("polar_cap_time_series_NH")
        recipe["diagnostics"].pop("lat_time_mapplot")
        recipe["diagnostics"].pop("zonal_mean_profiles")
        for diagnostic in recipe["diagnostics"].values():
            for variable in diagnostic["variables"].values():
                variable["additional_datasets"].extend(
                    recipe_variables[variable["short_name"]]["additional_datasets"]
                )
