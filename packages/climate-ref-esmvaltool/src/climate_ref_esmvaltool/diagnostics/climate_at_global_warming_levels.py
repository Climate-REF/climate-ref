from pathlib import Path

import pandas as pd

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request
from climate_ref_core.metric_values.typing import FileDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe

VARIABLES = ("pr", "tas")
CMIP7_BRANDED_VARIABLES = ("pr_tavg-u-hxy-u", "tas_tavg-h2m-hxy-u")
CMIP7_MATCHING_FACETS = ("source_id", "variant_label", "grid_label", "variable_id")
CONCENTRATION_DRIVEN_SCENARIOS = tuple(f"scen7-{s}" for s in ("h", "hl", "m", "ml", "l", "ln", "vl"))
EMISSIONS_DRIVEN_SCENARIOS = tuple(f"esm-{s}" for s in CONCENTRATION_DRIVEN_SCENARIOS)
PARENT_EXPERIMENTS = ("historical", "esm-hist")


def _cmip7_requirement(scenarios: tuple[str, ...], parent_experiment: str) -> tuple[DataRequirement]:
    """Build a CMIP7 requirement that pairs each scenario with its own parent historical run."""
    return (
        DataRequirement(
            source_type=SourceDatasetType.CMIP7,
            filters=(
                FacetFilter(
                    facets={
                        "branded_variable": CMIP7_BRANDED_VARIABLES,
                        "experiment_id": scenarios,
                        "frequency": "mon",
                        "region": "glb",
                    },
                ),
            ),
            group_by=("experiment_id",),
            constraints=(
                AddSupplementaryDataset(
                    supplementary_facets={"experiment_id": parent_experiment},
                    matching_facets=(*CMIP7_MATCHING_FACETS, "branded_variable", "frequency", "region"),
                    optional_matching_facets=tuple(),
                ),
                RequireTimerange(
                    group_by=CMIP7_MATCHING_FACETS,
                    start=PartialDateTime(year=1850, month=1),
                    end=PartialDateTime(year=2100, month=12),
                ),
                RequireFacets(
                    "experiment_id",
                    required_facets=(parent_experiment,),
                    group_by=CMIP7_MATCHING_FACETS,
                ),
                RequireFacets(
                    "variable_id",
                    required_facets=VARIABLES,
                    group_by=("experiment_id", "source_id", "variant_label", "grid_label"),
                ),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
            ),
        ),
    )


class ClimateAtGlobalWarmingLevels(ESMValToolDiagnostic):
    """
    Calculate climate variables at global warming levels.
    """

    name = "Climate variables at global warming levels"
    slug = "climate-at-global-warming-levels"
    base_recipe = "recipe_calculate_gwl_exceedance_stats.yml"

    variables = VARIABLES

    matching_facets = (
        "source_id",
        "member_id",
        "grid_label",
        "table_id",
        "variable_id",
    )

    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": variables,
                            "experiment_id": (
                                "ssp126",
                                "ssp245",
                                "ssp370",
                                "ssp585",
                            ),
                            "table_id": "Amon",
                        },
                    ),
                ),
                group_by=("experiment_id",),
                constraints=(
                    AddSupplementaryDataset(
                        supplementary_facets={"experiment_id": "historical"},
                        matching_facets=matching_facets,
                        optional_matching_facets=tuple(),
                    ),
                    RequireTimerange(
                        group_by=matching_facets,
                        start=PartialDateTime(year=1850, month=1),
                        end=PartialDateTime(year=2100, month=12),
                    ),
                    RequireFacets(
                        "experiment_id",
                        required_facets=("historical",),
                        group_by=matching_facets,
                    ),
                    RequireFacets(
                        "variable_id",
                        required_facets=variables,
                        group_by=("experiment_id", "source_id", "member_id", "grid_label", "table_id"),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                ),
            ),
        ),
        _cmip7_requirement(CONCENTRATION_DRIVEN_SCENARIOS, parent_experiment="historical"),
        _cmip7_requirement(EMISSIONS_DRIVEN_SCENARIOS, parent_experiment="esm-hist"),
    )
    facets = ("experiment_id", "global_warming_level", "metric")

    files = tuple(
        FileDefinition(
            file_pattern=f"plots/gwl_mean_plots_{var_name}/plot_gwl_stats/*.png",
            dimensions={
                "statistic": "mean",
                "variable_id": var_name,
            },
        )
        for var_name in variables
    ) + tuple(
        FileDefinition(
            file_pattern=f"work/gwl_mean_plots_{var_name}/plot_gwl_stats/*.nc",
            dimensions={
                "statistic": "mean",
                "variable_id": var_name,
            },
        )
        for var_name in variables
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
                            "experiment_id": ["ssp245", "historical"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", "pr", "tas"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("1850", "2100"),
                    ),
                ),
            ),
            # TODO: add a native CMIP7 scenario test case once a baseline is minted
        )
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pd.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Set up the datasets
        cmip_source = get_cmip_source_type(input_files)
        diagnostics = recipe["diagnostics"]
        for diagnostic in diagnostics.values():
            diagnostic.pop("additional_datasets")
        group_by: tuple[str, ...]
        if cmip_source == SourceDatasetType.CMIP7:
            group_by = CMIP7_MATCHING_FACETS
        else:
            group_by = (
                "source_id",
                "member_id",
                "grid_label",
                "table_id",
                "variable_id",
            )
        recipe_variables = dataframe_to_recipe(
            input_files[cmip_source],
            group_by=group_by,
        )
        datasets = recipe_variables["tas"]["additional_datasets"]
        datasets = [ds for ds in datasets if ds["exp"] not in PARENT_EXPERIMENTS]
        for dataset in datasets:
            dataset.pop("timerange")
            # The datasets are shared by tas and pr, so the branding is set per variable instead.
            dataset.pop("branding_suffix", None)
        recipe["datasets"] = datasets

        # Specify the timeranges
        diagnostics["calculate_gwl_exceedance_years"]["variables"]["tas_anomaly"] = {
            "short_name": "tas",
            "preprocessor": "calculate_anomalies",
            "timerange": "1850/2100",
        }

        diagnostics["gwl_mean_plots_tas"]["variables"]["tas"] = {
            "short_name": "tas",
            "preprocessor": "multi_model_gwl_stats",
            "timerange": "2000/2100",
        }

        diagnostics["gwl_mean_plots_pr"]["variables"]["pr"] = {
            "short_name": "pr",
            "preprocessor": "multi_model_gwl_stats",
            "timerange": "2000/2100",
        }

        if cmip_source == SourceDatasetType.CMIP7:
            branding = dict(branded.split("_", 1) for branded in CMIP7_BRANDED_VARIABLES)
            for diagnostic in diagnostics.values():
                for variable in diagnostic["variables"].values():
                    variable["branding_suffix"] = branding[variable["short_name"]]

    @staticmethod
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "global_warming_level",
                "metric",
            ],
            "global_warming_level": {},
            "metric": {"exceedance_year": {}},
        }

        df = pd.read_csv(
            result_dir
            / "work"
            / "calculate_gwl_exceedance_years"
            / "gwl_exceedance_calculation"
            / "GWL_exceedance_years.csv"
        )
        for row in df.itertuples(index=False):
            gwl = str(row.GWL)
            if gwl not in metric_args[MetricCV.DIMENSIONS.value]["global_warming_level"]:
                metric_args[MetricCV.DIMENSIONS.value]["global_warming_level"][gwl] = {}
            metric_args[MetricCV.RESULTS.value][gwl] = {
                "exceedance_year": int(str(row.Exceedance_Year)),
            }

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
