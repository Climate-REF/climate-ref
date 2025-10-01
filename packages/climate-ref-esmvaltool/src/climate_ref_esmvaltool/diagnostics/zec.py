from pathlib import Path

import pandas
import xarray

from climate_ref_core.constraints import (
    AddParentDataset,
    AddSupplementaryDataset,
    RequireContiguousTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, fillvalues_to_nan
from climate_ref_esmvaltool.recipe import get_child_and_parent_dataset
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class ZeroEmissionCommitment(ESMValToolDiagnostic):
    """
    Calculate the global mean Zero Emission Commitment (ZEC) temperature.
    """

    name = "Zero Emission Commitment"
    slug = "zero-emission-commitment"
    base_recipe = "recipe_zec.yml"

    data_requirements = (
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
                AddParentDataset(),
                RequireContiguousTimerange(group_by=("instance_id",)),
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

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Prepare updated datasets section in recipe. It contains two
        # datasets, one for the "esm-1pct-brch-1000PgC" and one for the "1pctCO2"
        # experiment.
        df = input_files[SourceDatasetType.CMIP6]
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
