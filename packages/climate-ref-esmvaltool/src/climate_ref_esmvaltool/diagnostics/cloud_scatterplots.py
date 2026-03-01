from collections.abc import Collection
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
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.metric_values.typing import FileDefinition
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import Recipe


def get_cmip_data_requirements(
    variables: tuple[str, ...],
    branded_variables: tuple[str, ...] | None = None,
) -> tuple[tuple[DataRequirement, ...], ...]:
    """Create data requirements for CMIP6 and CMIP7 data."""
    cmip7_facets: dict[str, str | Collection[str]] = {
        "experiment_id": "historical",
        "frequency": "mon",
        "region": "glb",
    }
    if branded_variables is not None:
        cmip7_facets["branded_variable"] = branded_variables
    return (
        (
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
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets=cmip7_facets,
                    ),
                ),
                group_by=("source_id", "experiment_id", "variant_label", "frequency", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1996, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    RequireFacets("variable_id", variables),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
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
    cmip_source = get_cmip_source_type(input_files)
    recipe_variables = dataframe_to_recipe(input_files[cmip_source])
    diagnostics = recipe["diagnostics"]
    diagnostic_name = f"plot_joint_{var_x}_{var_y}_model"
    diagnostic = diagnostics.pop(diagnostic_name)
    diagnostics.clear()
    diagnostics[diagnostic_name] = diagnostic
    recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}
    datasets = next(iter(recipe_variables.values()))["additional_datasets"]
    for var_data in recipe_variables.values():
        for ds in var_data["additional_datasets"]:
            ds["timerange"] = "1996/2014"

    if cmip_source == SourceDatasetType.CMIP7:
        # CMIP7: use per-variable additional_datasets to preserve correct branding_suffix
        for var_name, var_settings in diagnostic.get("variables", {}).items():
            short_name = var_settings.get("short_name", var_name)
            if short_name in recipe_variables:
                var_settings["additional_datasets"] = recipe_variables[short_name]["additional_datasets"]
            else:
                # For derived variables, use the first available dataset
                var_settings["additional_datasets"] = datasets
        # Remove diagnostic-level additional_datasets to avoid CMIP6 reference datasets
        # being merged with CMIP7 per-variable datasets
        diagnostic.pop("additional_datasets", None)
    else:
        diagnostic["additional_datasets"] = datasets

    project = datasets[0]["project"]
    suptitle = f"{project} {{dataset}} {{ensemble}} {{grid}} {{timerange}}".format(**datasets[0])
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
    data_requirements = get_cmip_data_requirements(
        ("clt", "rsut", "rsutcs"),
        branded_variables=("clt_tavg-u-hxy-u", "rsut_tavg-u-hxy-u", "rsutcs_tavg-u-hxy-u"),
    )
    update_recipe = partial(update_recipe, var_x="clt", var_y="swcre")
    files = (
        FileDefinition(
            file_pattern="plots/plot_joint_clt_swcre_model/plot/png/*.png",
            dimensions={"statistic": "joint histogram of clt vs swcre"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_clt_swcre_model/plot/*.nc",
            dimensions={"statistic": "joint histogram of clt vs swcre"},
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
                            "variable_id": ["areacella", "clt", "rsut", "rsutcs"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
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
                            "variable_id": ["areacella", "clt", "rsut", "rsutcs"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "clt_tavg-u-hxy-u",
                                "rsut_tavg-u-hxy-u",
                                "rsutcs_tavg-u-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
                    ),
                ),
            ),
        )
    )


class CloudScatterplotClwviPr(ESMValToolDiagnostic):
    """
    Scatterplot of clwvi vs pr.
    """

    name = "Scatterplots of two cloud-relevant variables (clwvi vs pr)"
    slug = "cloud-scatterplots-clwvi-pr"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = get_cmip_data_requirements(
        ("clwvi", "pr"),
        branded_variables=("clwvi_tavg-u-hxy-u", "pr_tavg-u-hxy-u"),
    )
    update_recipe = partial(update_recipe, var_x="clwvi", var_y="pr")
    files = (
        FileDefinition(
            file_pattern="plots/plot_joint_clwvi_pr_model/plot/png/*.png",
            dimensions={"statistic": "joint histogram of clwvi vs pr"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_clwvi_pr_model/plot/*.nc",
            dimensions={"statistic": "joint histogram of clwvi vs pr"},
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
                            "variable_id": ["areacella", "clwvi", "pr"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
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
                            "variable_id": ["areacella", "clwvi", "pr"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "clwvi_tavg-u-hxy-u",
                                "pr_tavg-u-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
                    ),
                ),
            ),
        )
    )


class CloudScatterplotCliviLwcre(ESMValToolDiagnostic):
    """
    Scatterplot of clivi vs lwcre.
    """

    name = "Scatterplots of two cloud-relevant variables (clivi vs lwcre)"
    slug = "cloud-scatterplots-clivi-lwcre"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = get_cmip_data_requirements(
        ("clivi", "rlut", "rlutcs"),
        branded_variables=("clivi_tavg-u-hxy-u", "rlut_tavg-u-hxy-u", "rlutcs_tavg-u-hxy-u"),
    )
    update_recipe = partial(update_recipe, var_x="clivi", var_y="lwcre")
    files = (
        FileDefinition(
            file_pattern="plots/plot_joint_clivi_lwcre_model/plot/png/*.png",
            dimensions={"statistic": "joint histogram of clivi vs lwcre"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_clivi_lwcre_model/plot/*.nc",
            dimensions={"statistic": "joint histogram of clivi vs lwcre"},
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
                            "variable_id": ["areacella", "clivi", "rlut", "rlutcs"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
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
                            "variable_id": ["areacella", "clivi", "rlut", "rlutcs"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "clivi_tavg-u-hxy-u",
                                "rlut_tavg-u-hxy-u",
                                "rlutcs_tavg-u-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
                    ),
                ),
            ),
        )
    )


class CloudScatterplotCliTa(ESMValToolDiagnostic):
    """
    Scatterplot of cli vs ta.
    """

    name = "Scatterplots of two cloud-relevant variables (cli vs ta)"
    slug = "cloud-scatterplots-cli-ta"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    data_requirements = get_cmip_data_requirements(
        ("cli", "ta"),
        branded_variables=("cli_tavg-al-hxy-u", "ta_tavg-p19-hxy-air"),
    )
    update_recipe = partial(update_recipe, var_x="cli", var_y="ta")
    files = (
        FileDefinition(
            file_pattern="plots/plot_joint_cli_ta_model/plot/png/*.png",
            dimensions={"statistic": "joint histogram of cli vs ta"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_cli_ta_model/plot/*.nc",
            dimensions={"statistic": "joint histogram of cli vs ta"},
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
                            "source_id": "CESM2",
                            "variable_id": ["areacella", "cli", "ta"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
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
                            "source_id": "CESM2",
                            "variable_id": ["areacella", "cli", "ta"],
                            "table_id": ["fx", "Amon"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "cli_tavg-al-hxy-u",
                                "ta_tavg-p19-hxy-air",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1996", "2014"),
                    ),
                ),
            ),
        )
    )


class CloudScatterplotsReference(ESMValToolDiagnostic):
    """
    Reference scatterplots of two cloud-relevant variables.
    """

    name = "Reference scatterplots of two cloud-relevant variables"
    slug = "cloud-scatterplots-reference"
    base_recipe = "ref/recipe_ref_scatterplot.yml"
    facets = ()
    files = (
        FileDefinition(
            file_pattern="plots/plot_joint_cli_ta_ref/plot/png/*.png",
            dimensions={"statistic": "joint histogram of cli vs ta"},
        ),
        FileDefinition(
            file_pattern="plots/plot_joint_clivi_lwcre_ref/plot/png/*.png",
            dimensions={"statistic": "joint histogram of clivi vs lwcre"},
        ),
        FileDefinition(
            file_pattern="plots/plot_joint_clt_swcre_ref/plot/png/*.png",
            dimensions={"statistic": "joint histogram of clt vs swcre"},
        ),
        FileDefinition(
            file_pattern="plots/plot_joint_clwvi_pr_ref/plot/png/*.png",
            dimensions={"statistic": "joint histogram of clwvi vs pr"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_cli_ta_ref/plot/*.nc",
            dimensions={"statistic": "joint histogram of cli vs ta"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_clivi_lwcre_ref/plot/*.nc",
            dimensions={"statistic": "joint histogram of clivi vs lwcre"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_clt_swcre_ref/plot/*.nc",
            dimensions={"statistic": "joint histogram of clt vs swcre"},
        ),
        FileDefinition(
            file_pattern="work/plot_joint_clwvi_pr_ref/plot/*.nc",
            dimensions={"statistic": "joint histogram of clwvi vs pr"},
        ),
    )
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
