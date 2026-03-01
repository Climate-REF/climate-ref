import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.metric_values.typing import FileDefinition
from climate_ref_core.testing import TestCase, TestDataSpecification
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
                            "branded_variable": (
                                "hurs_tavg-h2m-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "tas_tmaxavg-h2m-hxy-u",
                            ),
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                        }
                    ),
                    FacetFilter(
                        {
                            "branded_variable": (
                                "cVeg_tavg-u-hxy-lnd",
                                "treeFrac_tavg-u-hxy-u",
                                "vegFrac_tavg-u-hxy-u",
                            ),
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
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
                        "branded_variable",
                        (
                            "cVeg_tavg-u-hxy-lnd",
                            "hurs_tavg-h2m-hxy-u",
                            "pr_tavg-u-hxy-u",
                            "sftlf_ti-u-hxy-u",
                            "tas_tavg-h2m-hxy-u",
                            "tas_tmaxavg-h2m-hxy-u",
                            "treeFrac_tavg-u-hxy-u",
                            "vegFrac_tavg-u-hxy-u",
                        ),
                    ),
                ),
            ),
        ),
    )
    facets = ()
    files = (
        FileDefinition(
            file_pattern="plots/fire_evaluation/fire_evaluation/burnt_fraction_*.png",
            dimensions={"statistic": "burnt fraction"},
        ),
        FileDefinition(
            file_pattern="plots/fire_evaluation/fire_evaluation/fire_weather_control_*.png",
            dimensions={"statistic": "fire weather control"},
        ),
        FileDefinition(
            file_pattern="plots/fire_evaluation/fire_evaluation/fuel_load_continuity_control_*.png",
            dimensions={"statistic": "fuel load continuity control"},
        ),
    )

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
                            "source_id": "CanESM5",
                            "variable_id": [
                                "cVeg",
                                "hurs",
                                "pr",
                                "sftlf",
                                "tas",
                                "tasmax",
                                "treeFrac",
                                "vegFrac",
                            ],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("2013", "2014"),
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
                            "experiment_id": "historical",
                            "source_id": "CanESM5",
                            "variable_id": [
                                "cVeg",
                                "hurs",
                                "pr",
                                "sftlf",
                                "tas",
                                "tasmax",
                                "treeFrac",
                                "vegFrac",
                            ],
                            "branded_variable": [
                                "cVeg_tavg-u-hxy-lnd",
                                "hurs_tavg-h2m-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "sftlf_ti-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "tas_tmaxavg-h2m-hxy-u",
                                "treeFrac_tavg-u-hxy-u",
                                "vegFrac_tavg-u-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("2013", "2014"),
                    ),
                ),
            ),
        )
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        cmip_source = get_cmip_source_type(input_files)
        recipe_variables = dataframe_to_recipe(input_files[cmip_source])
        recipe.pop("datasets")
        for diagnostic in recipe["diagnostics"].values():
            for variable_group, variable in diagnostic.get("variables", {}).items():
                cmip6_short_name = variable.get("short_name", variable_group)
                if cmip_source == SourceDatasetType.CMIP7 and cmip6_short_name == "tasmax":
                    short_name = "tas"
                else:
                    short_name = cmip6_short_name
                variable["short_name"] = short_name
                variable["start_year"] = 2013
                variable["end_year"] = 2014
                datasets = recipe_variables[short_name]["additional_datasets"]
                for dataset in datasets:
                    dataset.pop("timerange", None)
                if cmip_source == SourceDatasetType.CMIP7 and short_name == "tas":
                    # Separate the two "tas" datasets into "tas" and "tasmax".
                    if cmip6_short_name == "tasmax":
                        datasets = [d for d in datasets if d["branding_suffix"] == "tmaxavg-h2m-hxy-u"]
                    else:
                        datasets = [d for d in datasets if d["branding_suffix"] == "tavg-h2m-hxy-u"]

                variable["additional_datasets"] = datasets

        recipe["diagnostics"]["fire_evaluation"]["scripts"]["fire_evaluation"]["remove_confire_files"] = True
