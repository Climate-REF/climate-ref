import pandas

from climate_ref_core.constraints import (
    PartialDateTime,
    RequireContiguousTimerange,
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
    base_recipe = "ref/recipe_ref_ozone.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
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
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "source_id": "C3S-GTO-ECV-9-0",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id",),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2014, 12),
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
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        dataset = recipe_variables["toz"]["additional_datasets"][0]
        # set time range of model (CMIP6) dataset (should match observational period)
        dataset["timerange"] = "1996/2014"
        recipe["datasets"] = [dataset]
        diagnostic = "lat_time_mapplot"
        recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}
        recipe["diagnostics"][diagnostic]["variables"]["toz"]["timerange"] = "1996/2014"


class O3PolarCapTimeseriesSH(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - October SH polar mean (60S-85S) time series.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-sh-oct"
    base_recipe = "ref/recipe_ref_ozone.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "experiment_id": "historical",
                        "table_id": "AERmon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1950, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "source_id": "C3S-GTO-ECV-9-0",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id",),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2021, 12),
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
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        dataset = recipe_variables["toz"]["additional_datasets"][0]
        # set model (CMIP6) time range to 1950...2014
        dataset["timerange"] = "1950/2014"
        recipe["datasets"] = [dataset]
        diagnostic = "polar_cap_time_series_SH"
        recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}


class O3PolarCapTimeseriesNH(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - March NH polar mean (60N-85N) time series.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-nh-mar"
    base_recipe = "ref/recipe_ref_ozone.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "experiment_id": "historical",
                        "table_id": "AERmon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1950, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "source_id": "C3S-GTO-ECV-9-0",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id",),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2021, 12),
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
        # Make sure only grid cells south of 80N are considered as there are no
        # measurements north of 80N in March. Specifying 85N as northern boundary
        # in the orignal 'recipe_ref_ozone.yml' is a bug!
        recipe["preprocessors"]["create_time_series_NH"]["extract_region"]["end_latitude"] = 80
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        dataset = recipe_variables["toz"]["additional_datasets"][0]
        # set model (CMIP6) time range to 1950...2014
        dataset["timerange"] = "1950/2014"
        recipe["datasets"] = [dataset]
        diagnostic = "polar_cap_time_series_NH"
        # adjust plot title to reflect bug fix regarding northern boundary (see above)
        recipe["diagnostics"][diagnostic]["scripts"]["plot"]["plots"]["timeseries"]["pyplot_kwargs"][
            "title"
        ] = "Total Column Ozone, 60-80N, March"
        recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}


class O3ZonalMeanProfiles(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - stratospheric zonal mean profiles.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-zonal"
    base_recipe = "ref/recipe_ref_ozone.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "o3",
                        "experiment_id": "historical",
                        "table_id": "Amon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(2005, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
            ),
        ),
        #        DataRequirement(
        #            source_type=SourceDatasetType.obs4REF,
        #            filters=(
        #                FacetFilter(
        #                    facets={
        #                        "variable_id": "o3",
        #                        "source_id": "ESACCI-OZONE",
        #                        "frequency": "mon",
        #                    },
        #                ),
        #            ),
        #            group_by=("source_id",),
        #            constraints=(
        #                RequireTimerange(
        #                    group_by=("instance_id",),
        #                    start=PartialDateTime(2005, 1),
        #                    end=PartialDateTime(2014, 12),
        #                ),
        #            ),
        #        ),
    )
    facets = ()

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        dataset = recipe_variables["o3"]["additional_datasets"][0]
        # set model (CMIP6) time range to 2005...2014
        dataset["timerange"] = "2005/2014"
        recipe["datasets"] = [dataset]
        diagnostic = "zonal_mean_profiles"
        # adjust plot title to actual time range
        recipe["diagnostics"][diagnostic]["scripts"]["plot"]["plots"]["zonal_mean_profile"]["pyplot_kwargs"][
            "suptitle"
        ] = "{long_name} (2005-2014 mean)"
        recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}
        recipe["diagnostics"][diagnostic]["variables"]["o3"]["timerange"] = "2005/2014"


class O3LatMonthMapplot(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - zonal mean total column ozone vs. annual cycle plot.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-annual-cycle"
    base_recipe = "ref/recipe_ref_ozone.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "experiment_id": "historical",
                        "table_id": "AERmon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(2005, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "source_id": "C3S-GTO-ECV-9-0",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id",),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(2005, 1),
                    end=PartialDateTime(2014, 12),
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
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        dataset = recipe_variables["toz"]["additional_datasets"][0]
        # set model (CMIP6) time range to 2005...2014
        dataset["timerange"] = "2005/2014"
        recipe["datasets"] = [dataset]
        diagnostic = "lat_month_mapplot"
        recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}
        recipe["diagnostics"][diagnostic]["variables"]["toz"]["timerange"] = "2005/2014"
