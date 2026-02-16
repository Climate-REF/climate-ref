import pandas

from climate_ref_core.constraints import AddSupplementaryDataset, RequireContiguousTimerange
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.metric_values.typing import FileDefinition, SeriesDefinition
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


class GlobalMeanTimeseries(ESMValToolDiagnostic):
    """
    Calculate the annual mean global mean timeseries for a dataset.
    """

    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"
    base_recipe = "examples/recipe_python.yml"

    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": ("tas",)}),),
                group_by=("source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
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
                            "branded_variable": "tas_tavg-h2m-hxy-u",
                            "region": "glb",
                        }
                    ),
                ),
                group_by=(
                    "source_id",
                    "experiment_id",
                    "variant_label",
                    "variable_id",
                    "grid_label",
                ),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
    )

    facets = ()
    series = (
        SeriesDefinition(
            file_pattern="work/timeseries/script1/*.nc",
            dimensions={
                "statistic": "annual mean",
                "variable_id": "tas",
                "region": "global",
            },
            values_name="tas",
            index_name="time",
            attributes=[],
        ),
    )
    files = (
        FileDefinition(
            file_pattern="plots/timeseries/script1/png/*.png",
            dimensions={
                "statistic": "annual mean",
                "variable_id": "tas",
                "region": "global",
            },
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
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", "tas"],
                            "frequency": ["fx", "mon"],
                            "experiment_id": "historical",
                        },
                        remove_ensembles=True,
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
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", "tas"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "experiment_id": "historical",
                            "region": "glb",
                        },
                        remove_ensembles=True,
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
        # Clear unwanted elements from the recipe.
        recipe["datasets"].clear()
        recipe["diagnostics"].pop("map")
        variables = recipe["diagnostics"]["timeseries"]["variables"]
        variables.clear()

        # Prepare updated variables section in recipe.
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
        recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}
        for variable in recipe_variables.values():
            variable["preprocessor"] = "annual_mean_global"
            variable["caption"] = "Annual global mean {long_name} according to {dataset}."

        # Populate recipe with new variables/datasets.
        variables.update(recipe_variables)
