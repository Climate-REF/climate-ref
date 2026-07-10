import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireOverlappingTimerange,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request, Obs4MIPsRequest
from climate_ref_core.metric_values.typing import FileDefinition, SeriesDefinition
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe

_REFERENCE_SOURCE_ID = "CERES-EBAF-4-2-1"

# CMIP6 `historical` ends in 2014, whereas CMIP7 runs through 2021.
_TIMERANGES = {
    SourceDatasetType.CMIP6: (2002, 2014),
    SourceDatasetType.CMIP7: (2002, 2021),
}


def _reference_data_requirement(start_year: int, end_year: int) -> DataRequirement:
    return DataRequirement(
        source_type=SourceDatasetType.obs4MIPs,
        filters=(
            FacetFilter(
                facets={
                    "frequency": "mon",
                    "source_id": _REFERENCE_SOURCE_ID,
                    "variable_id": (
                        "rlut",
                        "rlutcs",
                        "rsut",
                        "rsutcs",
                    ),
                }
            ),
        ),
        group_by=("source_id",),
        constraints=(
            RequireTimerange(
                group_by=("instance_id",),
                start=PartialDateTime(start_year, 1),
                end=PartialDateTime(end_year, 12),
            ),
        ),
    )


class CloudRadiativeEffects(ESMValToolDiagnostic):
    """
    Plot climatologies and zonal mean profiles of cloud radiative effects (sw + lw) for a dataset.
    """

    name = "Climatologies and zonal mean profiles of cloud radiative effects"
    slug = "cloud-radiative-effects"
    base_recipe = "ref/recipe_ref_cre_cmip7.yml"
    version = 2

    variables = (
        "rlut",
        "rlutcs",
        "rsut",
        "rsutcs",
    )
    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": variables,
                            "experiment_id": "historical",
                            "table_id": "Amon",
                        }
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(_TIMERANGES[SourceDatasetType.CMIP6][0], 1),
                        end=PartialDateTime(_TIMERANGES[SourceDatasetType.CMIP6][1], 12),
                    ),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets("variable_id", variables),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                ),
            ),
            _reference_data_requirement(*_TIMERANGES[SourceDatasetType.CMIP6]),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets={
                            "branded_variable": (
                                "rlut_tavg-u-hxy-u",
                                "rlutcs_tavg-u-hxy-u",
                                "rsut_tavg-u-hxy-u",
                                "rsutcs_tavg-u-hxy-u",
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
                        start=PartialDateTime(_TIMERANGES[SourceDatasetType.CMIP7][0], 1),
                        end=PartialDateTime(_TIMERANGES[SourceDatasetType.CMIP7][1], 12),
                    ),
                    RequireOverlappingTimerange(group_by=("instance_id",)),
                    RequireFacets("variable_id", variables),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
            _reference_data_requirement(*_TIMERANGES[SourceDatasetType.CMIP7]),
        ),
    )

    facets = ()
    files = (
        tuple(
            FileDefinition(
                file_pattern=f"plots/plot_profiles/plot/variable_vs_lat_{var_name}_*.png",
                dimensions={"variable_id": var_name, "statistic": "zonal mean"},
            )
            for var_name in ["lwcre", "swcre"]
        )
        + tuple(
            FileDefinition(
                file_pattern=f"plots/plot_maps/plot/map_{var_name}_*.png",
                dimensions={"variable_id": var_name, "statistic": "climatology map"},
            )
            for var_name in ["lwcre", "swcre"]
        )
        + tuple(
            FileDefinition(
                file_pattern=f"work/plot_maps/plot/map_{var_name}_*.nc",
                dimensions={"variable_id": var_name, "statistic": "climatology map"},
            )
            for var_name in ["lwcre", "swcre"]
        )
    )
    series = tuple(
        SeriesDefinition(
            file_pattern=f"plot_profiles/plot/variable_vs_lat_{var_name}_*.nc",
            sel={"dim0": 0},  # Select the model.
            dimensions={"variable_id": var_name, "statistic": "zonal mean"},
            values_name=var_name,
            index_name="lat",
            attributes=[],
        )
        for var_name in ["lwcre", "swcre"]
    ) + tuple(
        SeriesDefinition(
            file_pattern=f"plot_profiles/plot/variable_vs_lat_{var_name}_*.nc",
            sel={"dim0": 1},  # Select the observation.
            dimensions={
                "variable_id": var_name,
                "statistic": "zonal mean",
                "reference_source_id": _REFERENCE_SOURCE_ID,
            },
            values_name=var_name,
            index_name="lat",
            attributes=[],
        )
        for var_name in ["lwcre", "swcre"]
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
                            "variable_id": ["areacella", "rlut", "rlutcs", "rsut", "rsutcs"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("2002", "2014"),
                    ),
                    Obs4MIPsRequest(
                        slug="obs4mips",
                        facets={
                            "frequency": "mon",
                            "source_id": _REFERENCE_SOURCE_ID,
                            "variable_id": ["rlut", "rlutcs", "rsut", "rsutcs"],
                        },
                        time_span=("2002", "2014"),
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
                            "variable_id": ["areacella", "rlut", "rlutcs", "rsut", "rsutcs"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "rlut_tavg-u-hxy-u",
                                "rlutcs_tavg-u-hxy-u",
                                "rsut_tavg-u-hxy-u",
                                "rsutcs_tavg-u-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1995", "2014"),
                        # We fabricate the CMIP7 `historical` series to extend to 2021-12
                        extend_historical_to=(2021, 12),
                    ),
                    Obs4MIPsRequest(
                        slug="obs4mips",
                        facets={
                            "frequency": "mon",
                            "source_id": _REFERENCE_SOURCE_ID,
                            "variable_id": ["rlut", "rlutcs", "rsut", "rsutcs"],
                        },
                        time_span=("2002", "2021"),
                    ),
                ),
            ),
        )
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: dict[SourceDatasetType, pandas.DataFrame]) -> None:
        """Update the recipe."""
        cmip_source = get_cmip_source_type(input_files)
        start_year, end_year = _TIMERANGES[cmip_source]

        recipe_variables = dataframe_to_recipe(input_files[cmip_source])
        recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}

        datasets = recipe_variables["rsut"]["additional_datasets"]
        for dataset in datasets:
            dataset.pop("timerange")
        recipe["datasets"] = datasets

        # The base recipe is written for CMIP7, which runs a decade later than CMIP6.
        recipe["timerange_for_models"]["timerange"] = f"{start_year}/{end_year}"
        for diagnostic in recipe["diagnostics"].values():
            for variable in diagnostic["variables"].values():
                variable["timerange"] = f"{start_year}/{end_year}"
            for dataset in diagnostic["additional_datasets"]:
                dataset["start_year"] = start_year
                dataset["end_year"] = end_year
