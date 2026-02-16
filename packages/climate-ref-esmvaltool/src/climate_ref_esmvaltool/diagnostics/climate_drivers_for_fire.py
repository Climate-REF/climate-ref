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
                            "realm": "atmos",
                        }
                    ),
                    FacetFilter(
                        {
                            "branded_variable": (
                                "cVeg_tavg-u-hxy-lnd",
                                "treeFrac_tavg-u-hxy-u",
                            ),
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                            "realm": "land",
                        }
                    ),
                    FacetFilter(
                        {
                            "branded_variable": "vegFrac_tavg-u-hxy-u",
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                            "realm": "land",
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

        if cmip_source == SourceDatasetType.CMIP7:
            # CMIP7: use per-variable additional_datasets to preserve correct branding_suffix
            recipe["datasets"] = []
            for diagnostic in recipe["diagnostics"].values():
                for var_name, variable in diagnostic.get("variables", {}).items():
                    short_name = variable.get("short_name", var_name)
                    if short_name in recipe_variables:
                        datasets = recipe_variables[short_name]["additional_datasets"]
                        for ds in datasets:
                            ds.pop("mip", None)
                            ds.pop("timerange", None)
                            ds["start_year"] = 2013
                            ds["end_year"] = 2014
                        variable["additional_datasets"] = datasets
        else:
            dataset = recipe_variables["cVeg"]["additional_datasets"][0]
            dataset.pop("mip")
            dataset.pop("timerange")
            dataset["start_year"] = 2013
            dataset["end_year"] = 2014
            recipe["datasets"] = [dataset]

        recipe["diagnostics"]["fire_evaluation"]["scripts"]["fire_evaluation"]["remove_confire_files"] = True
