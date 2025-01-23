from pathlib import Path

import pandas
import xarray

from cmip_ref_core.constraints import AddSupplementaryDataset, RequireContiguousTimerange
from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import DataRequirement
from cmip_ref_metrics_esmvaltool._version import __version__
from cmip_ref_metrics_esmvaltool.metrics.base import ESMValToolMetric
from cmip_ref_metrics_esmvaltool.recipe import dataframe_to_recipe
from cmip_ref_metrics_esmvaltool.types import OutputBundle, Recipe


class GlobalMeanTimeseries(ESMValToolMetric):
    """
    Calculate the annual mean global mean timeseries for a dataset.
    """

    name = "Global Mean Timeseries"
    slug = "esmvaltool-global-mean-timeseries"
    base_recipe = "examples/recipe_python.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": ("tas",)}),),
            group_by=("instance_id",),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: pandas.DataFrame) -> None:
        """Update the recipe."""
        # Clear unwanted elements from the recipe.
        recipe["datasets"].clear()
        recipe["diagnostics"].pop("map")
        variables = recipe["diagnostics"]["timeseries"]["variables"]
        variables.clear()

        # Prepare updated variables section in recipe.
        recipe_variables = dataframe_to_recipe(input_files)
        recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}
        for variable in recipe_variables.values():
            variable["preprocessor"] = "annual_mean_global"
            variable["caption"] = "Annual global mean {long_name} according to {dataset}."

        # Populate recipe with new variables/datasets.
        variables.update(recipe_variables)

    @staticmethod
    def format_result(result_dir: Path) -> OutputBundle:
        """Format the result."""
        result = next(result_dir.glob("work/timeseries/script1/*.nc"))
        dataset = xarray.open_dataset(result)

        # TODO: Check how timeseries data are generally serialised
        cmec_output = {
            "DIMENSIONS": {
                "dimensions": {
                    "source_id": {dataset.attrs["source_id"]: {}},
                    "region": {"global": {}},
                    "variable": {"tas": {}},
                },
                "json_structure": [
                    "model",
                    "region",
                    "statistic",
                ],
            },
            # Is the schema tracked?
            "SCHEMA": {
                "name": "CMEC-REF",
                "package": "cmip_ref_metrics_esmvaltool",
                "version": __version__,
            },
            "RESULTS": {
                dataset.attrs["source_id"]: {"global": {"tas": 0}},
            },
        }

        return cmec_output
