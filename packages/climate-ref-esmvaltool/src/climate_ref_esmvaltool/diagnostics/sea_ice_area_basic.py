from collections.abc import Mapping
from types import MappingProxyType

import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request, RegistryRequest
from climate_ref_core.metric_values.typing import FileDefinition, SeriesDefinition
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import (
    _DATASETS_REGISTRY_NAME,
    ESMValToolDiagnostic,
    get_cmip_source_type,
)
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.reference_registry import parse_registry_key
from climate_ref_esmvaltool.types import Recipe

REGIONS = {
    "nh": "Northern Hemisphere",
    "sh": "Southern Hemisphere",
}

MONTHS = {
    "nh": "September",
    "sh": "February",
}

_REFERENCE_SOURCE_IDS = tuple(f"OSI-450-{region}" for region in REGIONS)
_REFERENCE_VARIABLES = ("sic", "areacello")

_REFERENCE_FACETS: Mapping[str, str | tuple[str, ...]] = MappingProxyType(
    {
        "project": "OBS",
        "source_id": _REFERENCE_SOURCE_IDS,
        "variable_id": _REFERENCE_VARIABLES,
    }
)
"""The OSI-450 observations the recipe compares each model against.

Solving and fetching share this, so a test case cannot fetch a different set of files
from the one the solver goes on to select.
"""

_REFERENCE_REQUIREMENT = DataRequirement(
    source_type=SourceDatasetType.ESMValToolReference,
    filters=(FacetFilter(facets=_REFERENCE_FACETS),),
    # A single execution plots both hemispheres, so grouping cannot be by `source_id`.
    # The filter pins the project, so this is one group holding every selected file.
    group_by=("project",),
    constraints=(
        RequireFacets("source_id", _REFERENCE_SOURCE_IDS),
        RequireFacets("variable_id", _REFERENCE_VARIABLES),
    ),
)
"""Declaring the observations means the run sees only these files rather than the whole registry,
and the executions that used them are recorded in the database.
"""

_REFERENCE_REQUEST = RegistryRequest(
    slug="osi-450",
    registry_name=_DATASETS_REGISTRY_NAME,
    # A request names a source type by its enum name. A name the test case catalog builder
    # does not recognise is skipped, leaving the case with nothing to solve.
    source_type=SourceDatasetType.ESMValToolReference.name,
    facets=_REFERENCE_FACETS,
    key_parser=parse_registry_key,
)
"""Fetches the reference data a test case needs.

`parse_registry_key` reads a registry key the way ingest reads the downloaded file,
so a facet cannot be spelled one way here and another way in the catalog.
"""


class SeaIceAreaBasic(ESMValToolDiagnostic):
    """
    Calculate seasonal cycle and time series of NH and SH sea ice area.
    """

    name = "Sea ice area basic"
    slug = "sea-ice-area-basic"
    base_recipe = "ref/recipe_ref_sea_ice_area_basic.yml"
    version = 2

    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": "siconc",
                            "experiment_id": "historical",
                            "table_id": "SImon",
                        },
                    ),
                ),
                group_by=("source_id", "experiment_id", "member_id", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1979, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP6),
                    RequireFacets("variable_id", ("siconc", "areacello")),
                ),
            ),
            _REFERENCE_REQUIREMENT,
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets={
                            "branded_variable": "siconc_tavg-u-hxy-u",
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                        },
                    ),
                ),
                group_by=("source_id", "experiment_id", "variant_label", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1979, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP7),
                    RequireFacets("variable_id", ("siconc", "areacello")),
                ),
            ),
            _REFERENCE_REQUIREMENT,
        ),
        # TODO: Use OSI-450-nh and OSI-450-sh from obs4MIPs once available.
    )
    facets = ()
    files = tuple(
        FileDefinition(
            file_pattern=f"plots/siarea_min/allplots/timeseries_sea_ice_area_{region}_*.png",
            dimensions={
                "region": REGIONS[region],
                "statistic": f"{MONTHS[region]} sea ice area",
            },
        )
        for region in REGIONS
    ) + tuple(
        FileDefinition(
            file_pattern=f"plots/siarea_seas/allplots/annual_cycle_sea_ice_area_{region}_*.png",
            dimensions={
                "region": REGIONS[region],
                "statistic": "20-year average seasonal cycle of the sea ice area",
            },
        )
        for region in REGIONS
    )
    series = tuple(
        SeriesDefinition(
            file_pattern=f"siarea_min/allplots/timeseries_sea_ice_area_{region}_*.nc",
            sel={"dim0": i},
            dimensions=(
                {
                    "region": REGIONS[region],
                    "statistic": f"{MONTHS[region]} sea ice area",
                }
                | ({} if i == 0 else {"reference_source_id": f"OSI-450-{region}"})
            ),
            values_name="siconc",
            index_name="time",
            attributes=[],
        )
        for region in REGIONS
        for i in range(2)
    ) + tuple(
        SeriesDefinition(
            file_pattern=f"siarea_seas/allplots/annual_cycle_sea_ice_area_{region}_*.nc",
            sel={"dim0": i},
            dimensions=(
                {
                    "region": REGIONS[region],
                    "statistic": "20-year average seasonal cycle of the sea ice area",
                }
                | ({} if i == 0 else {"reference_source_id": f"OSI-450-{region}"})
            ),
            values_name="siconc",
            index_name="month_number",
            attributes=[],
        )
        for region in REGIONS
        for i in range(2)
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
                            "variable_id": ["areacello", "siconc"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("1979", "2014"),
                    ),
                    _REFERENCE_REQUEST,
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
                            "variable_id": ["areacello", "siconc"],
                            "branded_variable": [
                                "areacello_ti-u-hxy-u",
                                "siconc_tavg-u-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1979", "2014"),
                    ),
                    _REFERENCE_REQUEST,
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
        # Update datasets
        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
        recipe["datasets"] = recipe_variables["siconc"]["additional_datasets"]

        # Use the timerange from the recipe, as defined in the variable.
        for dataset in recipe["datasets"]:
            dataset.pop("timerange")

        # Update observational datasets.
        # These name the files `_REFERENCE_REQUIREMENT` selects,
        # so a change here needs the same change there.
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
