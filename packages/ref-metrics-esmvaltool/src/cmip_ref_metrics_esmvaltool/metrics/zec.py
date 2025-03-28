from pathlib import Path

import pandas
import xarray

from cmip_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
)
from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import DataRequirement
from cmip_ref_metrics_esmvaltool.metrics.base import ESMValToolMetric
from cmip_ref_metrics_esmvaltool.recipe import dataframe_to_recipe
from cmip_ref_metrics_esmvaltool.types import OutputBundle, Recipe


class ZeroEmissionCommitment(ESMValToolMetric):
    """
    Calculate the global mean Zero Emission Commitment (ZEC) temperature.
    """

    name = "Zero Emission Commitment"
    slug = "esmvaltool-zero-emission-commitment"
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
                    },
                ),
            ),
            group_by=("source_id", "member_id", "grid_label"),
            constraints=(
                RequireFacets("experiment_id", experiments),
                RequireContiguousTimerange(group_by=("instance_id",)),
                RequireOverlappingTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: pandas.DataFrame) -> None:
        """Update the recipe."""
        # Prepare updated datasets section in recipe. It contains two
        # datasets, one for the "esm-1pct-brch-1000PgC" and one for the "piControl"
        # experiment.
        datasets = dataframe_to_recipe(input_files)["tas"]["additional_datasets"]
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

    @staticmethod
    def format_result(result_dir: Path) -> OutputBundle:
        """Format the result."""
        zec_file = result_dir / "work/zec/zec/zec_50.nc"
        zec = xarray.open_dataset(zec_file)

        source_id = zec.dataset.values[0].decode("utf-8").strip()
        cmec_output = {
            "DIMENSIONS": {
                "model": {source_id: {}},
                "region": {"global": {}},
                "metric": {"zec": {}},
                "json_structure": [
                    "model",
                    "region",
                    "metric",
                ],
            },
            "RESULTS": {
                source_id: {"global": {"zec": float(zec.zec.values[0])}},
            },
        }

        return cmec_output
