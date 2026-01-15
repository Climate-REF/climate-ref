from functools import partial

import pandas

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, Obs4MIPsRequest
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


def get_cmip6_data_requirements(variables: tuple[str, ...]) -> tuple[DataRequirement, ...]:
    """Create a data requirement for CMIP6 data."""
    return (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": variables,
                        "experiment_id": "historical",
                        "table_id": "Amon",
                    },
                ),
            ),
            group_by=("source_id", "experiment_id", "member_id", "frequency", "grid_label"),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(1996, 1),
                    end=PartialDateTime(2014, 12),
                ),
                RequireFacets("variable_id", variables),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )


def update_recipe(
    recipe: Recipe,
    input_files: dict[SourceDatasetType, pandas.DataFrame],
    var_x: str,
    var_y: str,
) -> None:
    """Update the recipe."""
    recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
    diagnostics = recipe["diagnostics"]
    diagnostic_name = f"plot_joint_{var_x}_{var_y}_model"
    diagnostic = diagnostics.pop(diagnostic_name)
    diagnostics.clear()
    diagnostics[diagnostic_name] = diagnostic
    recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}
    datasets = next(iter(recipe_variables.values()))["additional_datasets"]
    for dataset in datasets:
        dataset["timerange"] = "1996/2014"
    diagnostic["additional_datasets"] = datasets
    suptitle = "CMIP6 {dataset} {ensemble} {grid} {timerange}".format(**datasets[0])
    diagnostic["scripts"]["plot"]["suptitle"] = suptitle
    diagnostic["scripts"]["plot"]["plot_filename"] = (
        f"jointplot_{var_x}_{var_y}_{suptitle.replace(' ', '_').replace('/', '-')}"
    )


class CloudScatterplotCltSwcre(ESMValToolDiagnostic):
    """
    Scatterplot of clt vs swcre.
    """

    name = "Scatterplots of two cloud-relevant variables (clt vs swcre)"
    slug = "cloud-scatterplots-clt-swcre"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = get_cmip6_data_requirements(("clt", "rsut", "rsutcs"))
    update_recipe = partial(update_recipe, var_x="clt", var_y="swcre")

    test_data_spec = TestDataSpecification(
        test_cases=(
            # TODO: Select a model that has chunked output so we don't need to download a 10GB file
            # TestCase(
            #     name="default",
            #     description="Cloud scatterplot clt vs swcre from CESM2 historical",
            #     requests=(
            #         CMIP6Request(
            #             slug="cloud",
            #             facets={
            #                 "source_id": "CESM2",
            #                 "experiment_id": "historical",
            #                 "variable_id": ("clt", "rsut", "rsutcs"),
            #                 "member_id": "r1i1p1f1",
            #                 "table_id": "Amon",
            #             },
            #             time_span=("1996-01", "2014-12"),
            #         ),
            #         CMIP6Request(
            #             slug="areacella",
            #             facets={
            #                 "source_id": "CESM2",
            #                 "experiment_id": "historical",
            #                 "variable_id": "areacella",
            #                 "table_id": "fx",
            #                 "member_id": "r1i1p1f1",
            #             },
            #         ),
            #     ),
            # ),
        ),
    )


class CloudScatterplotClwviPr(ESMValToolDiagnostic):
    """
    Scatterplot of clwvi vs pr.
    """

    name = "Scatterplots of two cloud-relevant variables (clwvi vs pr)"
    slug = "cloud-scatterplots-clwvi-pr"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = get_cmip6_data_requirements(("clwvi", "pr"))
    update_recipe = partial(update_recipe, var_x="clwvi", var_y="pr")

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Cloud scatterplot clwvi vs pr from CESM2 historical",
                requests=(
                    CMIP6Request(
                        slug="cloud",
                        facets={
                            "source_id": "CESM2",
                            "experiment_id": "historical",
                            "variable_id": ("clwvi", "pr"),
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("1996-01", "2014-12"),
                    ),
                    CMIP6Request(
                        slug="areacella",
                        facets={
                            "source_id": "CESM2",
                            "experiment_id": "historical",
                            "variable_id": "areacella",
                            "table_id": "fx",
                        },
                    ),
                ),
            ),
        ),
    )


class CloudScatterplotCliviLwcre(ESMValToolDiagnostic):
    """
    Scatterplot of clivi vs lwcre.
    """

    name = "Scatterplots of two cloud-relevant variables (clivi vs lwcre)"
    slug = "cloud-scatterplots-clivi-lwcre"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = get_cmip6_data_requirements(("clivi", "rlut", "rlutcs"))
    update_recipe = partial(update_recipe, var_x="clivi", var_y="lwcre")

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Cloud scatterplot clivi vs lwcre from CESM2 historical",
                requests=(
                    CMIP6Request(
                        slug="cloud",
                        facets={
                            "source_id": "CESM2",
                            "experiment_id": "historical",
                            "variable_id": ("clivi", "rlut", "rlutcs"),
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("1996-01", "2014-12"),
                    ),
                    CMIP6Request(
                        slug="areacella",
                        facets={
                            "source_id": "CESM2",
                            "experiment_id": "historical",
                            "variable_id": "areacella",
                            "table_id": "fx",
                        },
                    ),
                ),
            ),
        ),
    )


class CloudScatterplotCliTa(ESMValToolDiagnostic):
    """
    Scatterplot of cli vs ta.
    """

    name = "Scatterplots of two cloud-relevant variables (cli vs ta)"
    slug = "cloud-scatterplots-cli-ta"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = get_cmip6_data_requirements(("cli", "ta"))
    update_recipe = partial(update_recipe, var_x="cli", var_y="ta")

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Cloud scatterplot cli vs ta from CESM2 historical",
                requests=(
                    CMIP6Request(
                        slug="cloud",
                        facets={
                            "source_id": "CESM2",
                            "experiment_id": "historical",
                            "variable_id": ("cli", "ta"),
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("1996-01", "2014-12"),
                    ),
                    CMIP6Request(
                        slug="areacella",
                        facets={
                            "source_id": "CESM2",
                            "experiment_id": "historical",
                            "variable_id": "areacella",
                            "table_id": "fx",
                        },
                    ),
                ),
            ),
        ),
    )


class CloudScatterplotsReference(ESMValToolDiagnostic):
    """
    Reference scatterplots of two cloud-relevant variables.
    """

    name = "Reference scatterplots of two cloud-relevant variables"
    slug = "cloud-scatterplots-reference"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(
                FacetFilter(
                    facets={
                        "source_id": ("ERA-5",),
                        "variable_id": ("ta",),
                    },
                ),
            ),
            group_by=("instance_id",),
            constraints=(
                RequireTimerange(
                    group_by=("instance_id",),
                    start=PartialDateTime(2007, 1),
                    end=PartialDateTime(2014, 12),
                ),
            ),
            # TODO: Add obs4MIPs datasets once available and working:
            #
            # obs4MIPs datasets with issues:
            # - GPCP-V2.3: pr
            # - CERES-EBAF-4-2: rlut, rlutcs, rsut, rsutcs
            #
            # Unsure if available on obs4MIPs:
            # - AVHRR-AMPM-fv3.0: clivi, clwvi
            # - ESACCI-CLOUD: clt
            # - CALIPSO-ICECLOUD: cli
            #
            # Related issues:
            # - https://github.com/Climate-REF/climate-ref/issues/260
            # - https://github.com/esMValGroup/esMValCore/issues/2712
            # - https://github.com/esMValGroup/esMValCore/issues/2711
            # - https://github.com/sciTools/iris/issues/6411
        ),
    )

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Cloud scatterplot reference from ERA-5 obs4MIPs",
                requests=(
                    Obs4MIPsRequest(
                        slug="era5-ta",
                        facets={
                            "source_id": "ERA-5",
                            "variable_id": "ta",
                        },
                        time_span=("2007-01", "2015-12"),
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
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.obs4MIPs])
        recipe["diagnostics"] = {k: v for k, v in recipe["diagnostics"].items() if k.endswith("_ref")}

        era5_dataset = recipe_variables["ta"]["additional_datasets"][0]
        era5_dataset["timerange"] = "2007/2015"  # Use the same timerange as for the other variable.
        era5_dataset["alias"] = era5_dataset["dataset"]
        diagnostic = recipe["diagnostics"]["plot_joint_cli_ta_ref"]
        diagnostic["variables"]["ta"]["additional_datasets"] = [era5_dataset]
        suptitle = "CALIPSO-ICECLOUD / {dataset} {timerange}".format(**era5_dataset)
        diagnostic["scripts"]["plot"]["suptitle"] = suptitle
        diagnostic["scripts"]["plot"]["plot_filename"] = (
            f"jointplot_cli_ta_{suptitle.replace(' ', '_').replace('/', '-')}"
        )

        # Use the correct obs4MIPs dataset name for dataset that cannot be ingested
        # https://github.com/Climate-REF/climate-ref/issues/260.
        diagnostic = recipe["diagnostics"]["plot_joint_clwvi_pr_ref"]
        diagnostic["variables"]["pr"]["additional_datasets"] = [
            {
                "dataset": "GPCP-V2.3",
                "project": "obs4MIPs",
                "alias": "GPCP-SG",
                "timerange": "1992/2016",
            }
        ]
