from pathlib import Path

import pandas
import pandas as pd

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class ENSOBasicClimatology(ESMValToolDiagnostic):
    """
    Calculate the ENSO CLIVAR metrics - background climatology.
    """

    name = "ENSO Basic Climatology"
    slug = "enso-basic-climatology"
    base_recipe = "ref/recipe_enso_basicclimatology.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": ("pr", "tauu"),
                        "experiment_id": "historical",
                        "table_id": "Amon",
                    },
                ),
                FacetFilter(
                    facets={
                        "variable_id": "tos",
                        "experiment_id": "historical",
                        "table_id": "Omon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                RequireFacets(
                    "variable_id",
                    (
                        "pr",
                        "tauu",
                        "tos",
                    ),
                ),
            ),
        ),
    )
    facets = ()

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="ENSO Basic Climatology from ACCESS-ESM1-5 historical",
                requests=(
                    CMIP6Request(
                        slug="atmos",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": ("pr", "tauu"),
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("1850-01", "2014-12"),
                    ),
                    CMIP6Request(
                        slug="ocean",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "tos",
                            "member_id": "r1i1p1f1",
                            "table_id": "Omon",
                        },
                        time_span=("1850-01", "2014-12"),
                    ),
                ),
            ),
        ),
    )

    series = (
        tuple(
            SeriesDefinition(
                file_pattern=f"diagnostic_metrics/plot_script/{source_id}_eq_{var_name}_bias.nc",
                dimensions=(
                    {
                        "statistic": (
                            f"zonal bias in the time-mean {var_name} structure across the equatorial Pacific"
                        ),
                    }
                    | ({} if source_id == "{source_id}" else {"reference_source_id": source_id})
                ),
                values_name="tos" if var_name == "sst" else var_name,
                index_name="lon",
                attributes=[],
            )
            for var_name in ("pr", "sst", "tauu")
            for source_id in ("{source_id}", "GPCP-V2.3", "TROPFLUX")
        )
        + tuple(
            SeriesDefinition(
                file_pattern=f"diagnostic_metrics/plot_script/{{source_id}}_eq_{var_name}_seacycle.nc",
                dimensions=(
                    {
                        "statistic": (
                            "zonal bias in the amplitude of the mean seasonal cycle of "
                            f"{var_name} in the equatorial Pacific"
                        ),
                    }
                    | ({} if source_id == "{source_id}" else {"reference_source_id": source_id})
                ),
                values_name="tos" if var_name == "sst" else var_name,
                index_name="lon",
                attributes=[],
            )
            for var_name in ("pr", "sst", "tauu")
            for source_id in ("{source_id}", "GPCP-V2.3", "TROPFLUX")
        )
        + tuple(
            SeriesDefinition(
                file_pattern="diagnostic_metrics/plot_script/{source_id}_pr_double.nc",
                dimensions=(
                    {
                        "statistic": (
                            "meridional bias in the time-mean pr structure across the eastern Pacific"
                        ),
                    }
                    | ({} if source_id == "{source_id}" else {"reference_source_id": source_id})
                ),
                values_name="pr",
                index_name="lat",
                attributes=[],
            )
            for source_id in ("{source_id}", "GPCP-V2.3")
        )
        + tuple(
            SeriesDefinition(
                file_pattern="diagnostic_metrics/plot_script/*_pr_double_seacycle.nc",
                dimensions=(
                    {
                        "statistic": (
                            "meridional bias in the amplitude of the mean seasonal "
                            "pr cycle in the eastern Pacific"
                        ),
                    }
                    | ({} if source_id == "{source_id}" else {"reference_source_id": source_id})
                ),
                values_name="pr",
                index_name="lat",
                attributes=[],
            )
            for source_id in ("{source_id}", "GPCP-V2.3")
        )
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        recipe.pop("datasets")
        for diagnostic in recipe["diagnostics"].values():
            for variable in diagnostic["variables"].values():
                variable["additional_datasets"].extend(
                    recipe_variables[variable["short_name"]]["additional_datasets"]
                )


class ENSOCharacteristics(ESMValToolDiagnostic):
    """
    Calculate the ENSO CLIVAR metrics - basic ENSO characteristics.
    """

    name = "ENSO Characteristics"
    slug = "enso-characteristics"
    base_recipe = "ref/recipe_enso_characteristics.yml"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": "tos",
                        "experiment_id": "historical",
                        "table_id": "Omon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP6),
                RequireFacets("variable_id", ("tos", "areacello")),
            ),
        ),
    )
    facets = ("grid_label", "member_id", "source_id", "region", "metric")
    # ENSO pattern and lifecycle are series, but the ESMValTool diagnostic
    # script does not save the values used in the figure.
    series = tuple()

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="ENSO Characteristics from ACCESS-ESM1-5 historical",
                requests=(
                    CMIP6Request(
                        slug="tos",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "tos",
                            "member_id": "r1i1p1f1",
                            "table_id": "Omon",
                        },
                        time_span=("1850-01", "2014-12"),
                    ),
                    CMIP6Request(
                        slug="areacello",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "areacello",
                            "table_id": "Ofx",
                        },
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
        recipe_variables = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])
        recipe["datasets"] = recipe_variables["tos"]["additional_datasets"]
        # TODO: update the observational data requirement once available on ESGF.
        # Observations - use only one per run
        recipe["datasets"].append(
            # {
            #     "dataset": "NOAA-ERSSTv5",
            #     "version": "v5",
            #     "project": "OBS6",
            #     "type": "reanaly",
            #     "tier": 2,
            # }
            {
                "dataset": "TROPFLUX",
                "version": "v1",
                "project": "OBS6",
                "type": "reanaly",
                "tier": 2,
            }
        )

    @staticmethod
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        metrics = pd.read_csv(
            result_dir / "work" / "diagnostic_metrics" / "plot_script" / "matrix.csv",
            names=["dataset", "metric_name", "metric_value"],
        )

        # Update the diagnostic bundle arguments with the computed diagnostics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "region",
                "metric",
            ],
            "region": {"global": {}},
            "metric": {metric: {} for metric in metrics.metric_name},
        }
        metric_args[MetricCV.RESULTS.value] = {
            "global": {
                metric_name: metric_value
                for metric_name, metric_value in zip(
                    metrics.metric_name,
                    metrics.metric_value,
                )
            },
        }

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
