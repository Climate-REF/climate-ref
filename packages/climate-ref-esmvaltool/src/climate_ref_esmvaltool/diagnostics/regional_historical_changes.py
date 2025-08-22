import copy

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


class RegionalHistoricalAnnualCycle(ESMValToolDiagnostic):
    """
    Plot regional historical annual cycle of climate variables.
    """

    name = "Regional historical annual cycle of climate variables"
    slug = "regional-historical-annual-cycle"
    base_recipe = "ref/recipe_ref_annual_cycle_region.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": (
                            "hus",
                            "pr",
                            "psl",
                            "tas",
                            "ua",
                        ),
                        "experiment_id": "historical",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireFacets(
                    "variable_id",
                    (
                        "hus",
                        "pr",
                        "psl",
                        "tas",
                        "ua",
                    ),
                ),
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": (
                            "psl",
                            "ua",
                        ),
                        "source_id": "ERA-5",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id",),
            constraints=(RequireContiguousTimerange(group_by=("instance_id",)),),
            # TODO: Add obs4MIPs datasets once available and working:
            #
            # obs4MIPs dataset that cannot be ingested (https://github.com/Climate-REF/climate-ref/issues/260):
            # - GPCP-V2.3: pr
            #
            # Not yet available on obs4MIPs:
            # - ERA5: hus
            # - HadCRUT5_ground_5.0.1.0-analysis: tas
        ),
    )
    facets = ("source_id", "member_id", "grid_label")

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        regions = [
            "Arabian-Peninsula",
            "Arabian-Sea",
            "Arctic-Ocean",
            "Bay-of-Bengal",
            "C.Australia",
            "C.North-America",
            "Caribbean",
            "Central-Africa",
            "E.Antarctica",
            "E.Asia",
            "E.Australia",
            "E.C.Asia",
            "E.Europe",
            "E.North-America",
            "E.Siberia",
            "E.Southern-Africa",
            "Equatorial.Atlantic-Ocean",
            "Equatorial.Indic-Ocean",
            "Equatorial.Pacific-Ocean",
            "Greenland/Iceland",
            "Madagascar",
            "Mediterranean",
            "N.Atlantic-Ocean",
            "N.Australia",
            "N.Central-America",
            "N.E.North-America",
            "N.E.South-America",
            "N.Eastern-Africa",
            "N.Europe",
            "N.Pacific-Ocean",
            "N.South-America",
            "N.W.North-America",
            "N.W.South-America",
            "New-Zealand",
            "Russian-Arctic",
            "Russian-Far-East",
            "S.Asia",
            "S.Atlantic-Ocean",
            "S.Australia",
            "S.Central-America",
            "S.E.Asia",
            "S.E.South-America",
            "S.Eastern-Africa",
            "S.Indic-Ocean",
            "S.Pacific-Ocean",
            "S.South-America",
            "S.W.South-America",
            "Sahara",
            "South-American-Monsoon",
            "Southern-Ocean",
            "Tibetan-Plateau",
            "W.Antarctica",
            "W.C.Asia",
            "W.North-America",
            "W.Siberia",
            "W.Southern-Africa",
            "West&Central-Europe",
            "Western-Africa",
        ]

        # Update the dataset.
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        dataset = recipe_variables["hus"]["additional_datasets"][0]
        dataset.pop("timerange")
        dataset["benchmark_dataset"] = True
        dataset["plot_label"] = "{dataset}.{ensemble}.{grid}".format(**dataset)
        recipe["datasets"] = [dataset]

        # Generate diagnostics for each region.
        diagnostics = {}
        for region in regions:
            for diagnostic_name, orig_diagnostic in recipe["diagnostics"].items():
                # Create the diagnostic for the region.
                diagnostic = copy.deepcopy(orig_diagnostic)
                diagnostics[f"{diagnostic_name}-{region}"] = diagnostic

                for variable in diagnostic["variables"].values():
                    # Remove unwanted facets that are part of the dataset.
                    for facet in ("project", "exp", "ensemble", "grid"):
                        variable.pop(facet, None)
                    # Update the preprocessor so it extracts the region.
                    preprocessor_name = variable["preprocessor"]
                    preprocessor = copy.deepcopy(recipe["preprocessors"][preprocessor_name])
                    preprocessor["extract_shape"]["ids"] = {"Name": [region]}
                    variable["preprocessor"] = f"{preprocessor_name}-{region}"
                    recipe["preprocessors"][variable["preprocessor"]] = preprocessor

                # Update plot titles with region name.
                for script in diagnostic["scripts"].values():
                    for plot in script["plots"].values():
                        plot["pyplot_kwargs"] = {"title": f"{{long_name}} {region}"}
        recipe["diagnostics"] = diagnostics


class RegionalHistoricalTimeSeries(RegionalHistoricalAnnualCycle):
    """
    Plot regional historical mean and anomaly of climate variables.
    """

    name = "Regional historical mean and anomaly of climate variables"
    slug = "regional-historical-timeseries"
    base_recipe = "ref/recipe_ref_timeseries_region.yml"


class RegionalHistoricalTrend(ESMValToolDiagnostic):
    """
    Plot regional historical trend of climate variables.
    """

    name = "Regional historical trend of climate variables"
    slug = "regional-historical-trend"
    base_recipe = "ref/recipe_ref_trend_regions.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": (
                            "hus",
                            "pr",
                            "psl",
                            "tas",
                            "ua",
                        ),
                        "experiment_id": "historical",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": (
                            "psl",
                            "tas",
                            "ua",
                        ),
                        "source_id": "ERA-5",
                        "frequency": "mon",
                    },
                ),
            ),
            group_by=("source_id",),
            constraints=(RequireContiguousTimerange(group_by=("instance_id",)),),
            # TODO: Add obs4MIPs datasets once available and working:
            #
            # obs4MIPs dataset that cannot be ingested (https://github.com/Climate-REF/climate-ref/issues/260):
            # - GPCP-V2.3: pr
            #
            # Not yet available on obs4MIPs:
            # - ERA5: hus
            # - HadCRUT5_ground_5.0.1.0-analysis: tas
        ),
    )
    facets = ("source_id", "member_id", "grid_label")

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe["datasets"] = []
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        diagnostics = {}
        for diagnostic_name, diagnostic in recipe["diagnostics"].items():
            for variable_name, variable in diagnostic["variables"].items():
                if variable_name not in recipe_variables:
                    continue
                dataset = recipe_variables[variable_name]["additional_datasets"][0]
                dataset.pop("timerange")
                variable["additional_datasets"].append(dataset)
                diagnostics[diagnostic_name] = diagnostic
        recipe["diagnostics"] = diagnostics
