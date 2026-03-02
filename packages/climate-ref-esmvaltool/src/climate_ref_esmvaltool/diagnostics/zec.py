from pathlib import Path

import pandas
import xarray

from climate_ref_core.constraints import (
    AddParentDataset,
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.metric_values.typing import FileDefinition, SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import (
    ESMValToolDiagnostic,
    fillvalues_to_nan,
    get_cmip_source_type,
)
from climate_ref_esmvaltool.recipe import get_child_and_parent_dataset
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
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": "tas",
                            "experiment_id": "esm-1pct-brch-1000PgC",
                            "table_id": "Amon",
                        },
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    AddParentDataset.from_defaults(SourceDatasetType.CMIP6),
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                ),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets={
                            "branded_variable": "tas_tavg-h2m-hxy-u",
                            "experiment_id": experiments,
                            "frequency": "mon",
                            "region": "glb",
                        },
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireFacets("experiment_id", experiments),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
    )
    facets = ("grid_label", "member_id", "variant_label", "source_id", "region", "metric")
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
    files = (
        FileDefinition(
            file_pattern="plots/zec/zec/*.png",
            dimensions={"statistic": "zec"},
        ),
        FileDefinition(
            file_pattern="work/zec/zec/zec_50.nc",
            dimensions={"metric": "zec"},
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
                            "experiment_id": ["1pctCO2", "esm-1pct-brch-1000PgC"],
                            "source_id": "ACCESS-ESM1-5",
                            "variable_id": ["areacella", "tas"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
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
                            "experiment_id": ["1pctCO2", "esm-1pct-brch-1000PgC"],
                            "source_id": "ACCESS-ESM1-5",
                            "variable_id": ["areacella", "tas"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                    ),
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
        # Prepare updated datasets section in recipe. It contains two
        # datasets, one for the "esm-1pct-brch-1000PgC" and one for the "1pctCO2"
        # experiment.
        cmip_source = get_cmip_source_type(input_files)
        df = input_files[cmip_source]
        child_dataset, parent_dataset = get_child_and_parent_dataset(
            df[df.variable_id == "tas"],
            parent_experiment="1pctCO2",
            child_duration_in_years=100,
            parent_offset_in_years=-10,
            parent_duration_in_years=20,
        )

        variables = recipe["diagnostics"]["zec"]["variables"]
        variables["tas_base"] = {
            "short_name": "tas",
            "preprocessor": "anomaly_base",
            "additional_datasets": [parent_dataset],
        }
        variables["tas"] = {
            "preprocessor": "spatial_mean",
            "additional_datasets": [child_dataset],
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
