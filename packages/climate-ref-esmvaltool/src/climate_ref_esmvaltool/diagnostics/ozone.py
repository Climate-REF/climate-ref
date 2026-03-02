import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireContiguousTimerange,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf.cmip6 import CMIP6Request
from climate_ref_core.esgf.cmip7 import CMIP7Request
from climate_ref_core.esgf.obs4mips import Obs4MIPsRequest
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe

ozone_obs_filter = FacetFilter(
    facets={
        "variable_id": "toz",
        "source_id": "C3S-GTO-ECV-9-0",
        "frequency": "mon",
    },
)

toz_data_requirement = (
    (
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
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(ozone_obs_filter,),
            group_by=("source_id",),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2014, 12),
                ),
            ),
        ),
    ),
    (
        DataRequirement(
            source_type=SourceDatasetType.CMIP7,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "toz",
                        "experiment_id": "historical",
                        "branded_variable": "toz_tavg-u-hxy-u",
                        "frequency": "mon",
                        "region": "glb",
                    },
                ),
            ),
            group_by=("source_id", "variant_label", "grid_label"),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(ozone_obs_filter,),
            group_by=("source_id",),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2014, 12),
                ),
            ),
        ),
    ),
)
toz_test_spec = TestDataSpecification(
    test_cases=(
        TestCase(
            name="cmip6",
            description="Test with CMIP6 data.",
            requests=(
                CMIP6Request(
                    slug="cmip6",
                    facets={
                        "experiment_id": "historical",
                        "frequency": "mon",
                        "source_id": "GFDL-ESM4",
                        "variable_id": "toz",
                    },
                    remove_ensembles=True,
                    time_span=("1996", "2015"),
                ),
                Obs4MIPsRequest(
                    slug="obs4mips",
                    facets=ozone_obs_filter.facets,
                    remove_ensembles=False,
                    time_span=("1980", "2009"),
                ),
            ),
        ),
        TestCase(
            name="cmip7",
            description="Test with CMIP7 data.",
            requests=(
                CMIP7Request(
                    slug="cmip7",
                    facets={
                        "experiment_id": ["historical"],
                        "source_id": "GFDL-ESM4",
                        "variable_id": "toz",
                        "branded_variable": [
                            "toz_tavg-u-hxy-u",
                        ],
                        "variant_label": "r1i1p1f1",
                        "frequency": ["fx", "mon"],
                        "region": "glb",
                    },
                    remove_ensembles=True,
                    time_span=("1980", "2009"),
                ),
                Obs4MIPsRequest(
                    slug="obs4mips",
                    facets=ozone_obs_filter.facets,
                    remove_ensembles=False,
                    time_span=("1980", "2009"),
                ),
            ),
        ),
    ),
)


class O3LatTimeMapplot(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - zonal mean total column ozone vs. time.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-lat-time"
    base_recipe = "ref/recipe_ref_ozone.yml"

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
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

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
        dataset = recipe_variables["toz"]["additional_datasets"][0]
        # set model (CMIP6) time range to 1950...2014
        dataset["timerange"] = "1950/2014"
        recipe["datasets"] = [dataset]
        diagnostic = "polar_cap_time_series_SH"
        recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}


class O3PolarCapTimeseriesNH(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - March NH polar mean (60N-80N) time series.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-nh-mar"
    base_recipe = "ref/recipe_ref_ozone.yml"

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec

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
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
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
        # TODO: Use ESACCI-OZONE (SAGE-OMPS, variable o3) from obs4MIPs once available.
    )
    facets = ()
    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="cmip6",
                description="Test with CMIP6 data.",
                requests=(
                    CMIP6Request(
                        slug="cmip6",
                        facets={
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "source_id": "GFDL-ESM4",
                            "variable_id": "o3",
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2015"),
                    ),
                ),
            ),
            TestCase(
                name="cmip7",
                description="Test with CMIP7 data.",
                requests=(
                    CMIP7Request(
                        slug="cmip7",
                        facets={
                            "experiment_id": ["historical"],
                            "source_id": "GFDL-ESM4",
                            "variable_id": "o3",
                            "branded_variable": [
                                "o3_tavg-al-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1980", "2009"),
                    ),
                ),
            ),
        ),
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
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

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
        dataset = recipe_variables["toz"]["additional_datasets"][0]
        # set model (CMIP6) time range to 2005...2014
        dataset["timerange"] = "2005/2014"
        recipe["datasets"] = [dataset]
        diagnostic = "lat_month_mapplot"
        recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}
        recipe["diagnostics"][diagnostic]["variables"]["toz"]["timerange"] = "2005/2014"
