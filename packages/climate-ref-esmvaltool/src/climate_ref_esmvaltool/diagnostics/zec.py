from pathlib import Path

import pandas
import xarray

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
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, fillvalues_to_nan
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class ZeroEmissionCommitment(ESMValToolDiagnostic):
    """
    Calculate the global mean Zero Emission Commitment (ZEC) temperature.
    """

    name = "Zero Emission Commitment"
    slug = "zero-emission-commitment"
    base_recipe = "recipe_zec.yml"

    experiments = (
        "1pctCO2",
        "esm-1pct-brch-1000PgC",
    )
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": ("tas",),
                        "experiment_id": experiments,
                        "table_id": "Amon",
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                RequireFacets("experiment_id", experiments),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )
    facets = ("grid_label", "member_id", "source_id", "region", "metric")
    series = (
        SeriesDefinition(
            file_pattern="work/zec/zec/zec.nc",
            sel={"dim0": 0},
            dimensions={
                "statistic": "zec",
            },
            values_name="zec",
            index_name="time",
            attributes=[],
        ),
    )

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="ZEC from ACCESS-ESM1-5 1pctCO2 and esm-1pct-brch-1000PgC",
                requests=(
                    CMIP6Request(
                        slug="1pctco2",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "1pctCO2",
                            "variable_id": "tas",
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("0101-01", "0300-12"),
                    ),
                    CMIP6Request(
                        slug="esm-1pct-brch-1000pgc",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "esm-1pct-brch-1000PgC",
                            "variable_id": "tas",
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("0158-01", "0300-12"),
                    ),
                    CMIP6Request(
                        slug="areacella",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "1pctCO2",
                            "variable_id": "areacella",
                            "table_id": "fx",
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
        # Prepare updated datasets section in recipe. It contains two
        # datasets, one for the "esm-1pct-brch-1000PgC" and one for the "piControl"
        # experiment.
        datasets = dataframe_to_recipe(input_files[SourceDatasetType.CMIP6])["tas"]["additional_datasets"]
        base_dataset = next(ds for ds in datasets if ds["exp"] == "1pctCO2")
        dataset = next(ds for ds in datasets if ds["exp"] == "esm-1pct-brch-1000PgC")
        start = dataset["timerange"].split("/")[0]
        base_start = f"{int(start[:4]) - 10:04d}{start[4:]}"
        base_end = f"{int(start[:4]) + 10:04d}{start[4:]}"
        base_dataset["timerange"] = f"{base_start}/{base_end}"
        variables = recipe["diagnostics"]["zec"]["variables"]
        variables["tas_base"] = {
            "short_name": "tas",
            "preprocessor": "anomaly_base",
            "additional_datasets": [base_dataset],
        }
        variables["tas"] = {
            "preprocessor": "spatial_mean",
            "additional_datasets": [dataset],
        }

    @classmethod
    def format_result(
        cls,
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        zec_ds = xarray.open_dataset(result_dir / "work" / "zec" / "zec" / "zec_50.nc")
        zec = float(fillvalues_to_nan(zec_ds["zec"].values)[0])

        # Update the diagnostic bundle arguments with the computed diagnostics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": ["region", "metric"],
            "region": {"global": {}},
            "metric": {"zec": {}},
        }
        metric_args[MetricCV.RESULTS.value] = {
            "global": {
                "zec": zec,
            },
        }

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
