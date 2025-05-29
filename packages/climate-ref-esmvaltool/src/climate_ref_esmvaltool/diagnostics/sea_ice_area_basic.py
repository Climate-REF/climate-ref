import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


class SeaIceAreaBasic(ESMValToolDiagnostic):
    """
    Calculate seasonal cycle and time series of NH and SH sea ice area.
    """

    name = "Sea ice area basic"
    slug = "sea-ice-area-basic"
    base_recipe = "ref/recipe_ref_sea_ice_area_basic.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "siconc",
                        "experiment_id": "historical",
                    },
                ),
            ),
            group_by=("instance_id",),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP6),
            ),
        ),
        # TODO: Use OSI-450-nh and OSI-450-sh from obs4MIPs once available.
    )
    facets = ()

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Update datasets
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        recipe["datasets"] = recipe_variables["siconc"]["additional_datasets"]

        # Use the timerange from the recipe, as defined in the variable.
        for dataset in recipe["datasets"]:
            dataset.pop("timerange")

        # Update observational datasets
        nh_obs = {
            "dataset": "OSI-450-nh",
            "mip": "OImon",
            "project": "OBS",
            "supplementary_variables": [
                {
                    "short_name": "areacello",
                    "mip": "fx",
                },
            ],
            "tier": 2,
            "type": "reanaly",
            "version": "v3",
        }
        sh_obs = nh_obs.copy()
        sh_obs["dataset"] = "OSI-450-sh"
        diagnostics = recipe["diagnostics"]
        diagnostics["siarea_min"]["variables"]["sea_ice_area_nh_sep"]["additional_datasets"] = [nh_obs]
        diagnostics["siarea_min"]["variables"]["sea_ice_area_sh_feb"]["additional_datasets"] = [sh_obs]
        diagnostics["siarea_seas"]["variables"]["sea_ice_area_nh"]["additional_datasets"] = [nh_obs]
        diagnostics["siarea_seas"]["variables"]["sea_ice_area_sh"]["additional_datasets"] = [sh_obs]

        # Update the captions.
        dataset = "{dataset}.{ensemble}.{grid}".format(**recipe["datasets"][0])
        for diagnostic in diagnostics.values():
            for script_settings in diagnostic["scripts"].values():
                for plot_settings in script_settings["plots"].values():
                    plot_settings["caption"] = plot_settings["caption"].replace("[dataset]", dataset)
