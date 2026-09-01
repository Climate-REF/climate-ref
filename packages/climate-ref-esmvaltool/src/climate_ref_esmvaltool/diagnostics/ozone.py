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
from climate_ref_core.metric_values.typing import FileDefinition, SeriesDefinition
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import (
    ESMValToolDiagnostic,
    get_cmip_source_type,
)
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
                    end=PartialDateTime(2021, 12),
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
                # The recipe plots the observations through 2021, so require that coverage.
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2021, 12),
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
                        "variable_id": "toz",
                        "branded_variable": [
                            "toz_tavg-u-hxy-u",
                        ],
                        "variant_label": "r1i1p1f1",
                        "frequency": ["fx", "mon"],
                        "region": "glb",
                    },
                    remove_ensembles=True,
                    time_span=("1950", "2015"),
                    # Fabricate the CMIP7 historical series to extend to 2021-12
                    extend_historical_to=(2021, 12),
                ),
                Obs4MIPsRequest(
                    slug="obs4mips",
                    facets=ozone_obs_filter.facets,
                    remove_ensembles=False,
                    time_span=("1996", "2021"),
                ),
            ),
        ),
    ),
)


def _prepare_recipe(
    recipe: Recipe,
    input_files: dict[SourceDatasetType, pandas.DataFrame],
    diagnostic: str,
    variable: str,
    cmip7_timerange: str | None = None,
) -> None:
    """
    Prune the recipe to one diagnostic and insert the solved model dataset.
    """
    cmip_source = get_cmip_source_type(input_files)
    recipe_variables = dataframe_to_recipe(input_files[cmip_source])

    dataset = recipe_variables[variable]["additional_datasets"][0]
    dataset.pop("timerange", None)
    recipe["datasets"] = [dataset]
    recipe["diagnostics"] = {diagnostic: recipe["diagnostics"][diagnostic]}

    # The recipe includes CMIP6 time ranges
    if cmip_source == SourceDatasetType.CMIP7 and cmip7_timerange is not None:
        recipe["diagnostics"][diagnostic]["variables"][variable]["timerange"] = cmip7_timerange


class O3LatTimeMapplot(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - zonal mean total column ozone vs. time.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-lat-time"
    base_recipe = "ref/recipe_ref_ozone.yml"
    version = 3

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec
    files = (
        FileDefinition(
            file_pattern="plots/lat_time_mapplot/plot/*.png",
            dimensions={"variable_id": "toz", "statistic": "zonal mean vs time"},
        ),
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        _prepare_recipe(recipe, input_files, "lat_time_mapplot", "toz", cmip7_timerange="1997/2021")


class O3PolarCapTimeseriesSH(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - October SH polar mean (60S-85S) time series.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-sh-oct"
    base_recipe = "ref/recipe_ref_ozone.yml"
    version = 3

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec
    files = (
        FileDefinition(
            file_pattern="plots/polar_cap_time_series_SH/plot/timeseries_toz_SH_Oct.png",
            dimensions={
                "variable_id": "toz",
                "statistic": "Southern Hemisphere October polar mean",
            },
        ),
    )
    # dim0=0 is the model, dim0=1 contains the observational reference data.
    series = (
        SeriesDefinition(
            file_pattern="work/polar_cap_time_series_SH/plot/timeseries_toz_SH_Oct.nc",
            sel={"dim0": 0},
            dimensions={
                "variable_id": "toz",
                "statistic": "Southern Hemisphere October polar mean",
            },
            values_name="toz",
            index_name="time",
            attributes=[],
        ),
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        _prepare_recipe(
            recipe,
            input_files,
            "polar_cap_time_series_SH",
            "toz",
            cmip7_timerange="1950/2021",
        )


class O3PolarCapTimeseriesNH(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - March NH polar mean (60N-85N) time series.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-nh-mar"
    base_recipe = "ref/recipe_ref_ozone.yml"
    version = 3

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec
    files = (
        FileDefinition(
            file_pattern="plots/polar_cap_time_series_NH/plot/timeseries_toz_NH_MAR.png",
            dimensions={
                "variable_id": "toz",
                "statistic": "Northern Hemisphere March polar mean",
            },
        ),
    )
    # dim0=0 is the model, dim0=1 contains the observational reference data.
    series = (
        SeriesDefinition(
            file_pattern="work/polar_cap_time_series_NH/plot/timeseries_toz_NH_MAR.nc",
            sel={"dim0": 0},
            dimensions={
                "variable_id": "toz",
                "statistic": "Northern Hemisphere March polar mean",
            },
            values_name="toz",
            index_name="time",
            attributes=[],
        ),
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        _prepare_recipe(
            recipe,
            input_files,
            "polar_cap_time_series_NH",
            "toz",
            cmip7_timerange="1950/2021",
        )


class O3ZonalMeanProfiles(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - stratospheric zonal mean profiles.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-zonal"
    base_recipe = "ref/recipe_ref_ozone.yml"
    version = 4

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
                    start=PartialDateTime(1990, 1),
                    end=PartialDateTime(2000, 12),
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
                        time_span=("1990", "2001"),
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
                        time_span=("1990", "2001"),
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
        _prepare_recipe(recipe, input_files, "zonal_mean_profiles", "o3")


class O3LatMonthMapplot(ESMValToolDiagnostic):
    """
    Calculate the ozone diagnostics - zonal mean total column ozone vs. annual cycle plot.
    """

    name = "Ozone Diagnostics"
    slug = "ozone-annual-cycle"
    base_recipe = "ref/recipe_ref_ozone.yml"
    version = 3

    data_requirements = toz_data_requirement
    facets = ()
    test_data_spec = toz_test_spec
    files = (
        FileDefinition(
            file_pattern="plots/lat_month_mapplot/plot/*.png",
            dimensions={"variable_id": "toz", "statistic": "zonal mean annual cycle"},
        ),
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        _prepare_recipe(recipe, input_files, "lat_month_mapplot", "toz", cmip7_timerange="1997/2021")
