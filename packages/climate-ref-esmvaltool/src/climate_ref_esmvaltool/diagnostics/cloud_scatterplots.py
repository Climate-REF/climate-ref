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
from climate_ref_core.metric_values.typing import FileDefinition
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
