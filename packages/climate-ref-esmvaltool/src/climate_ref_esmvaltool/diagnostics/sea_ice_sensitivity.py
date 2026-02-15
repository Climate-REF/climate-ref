from pathlib import Path

import pandas
import pandas as pd

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, get_cmip_source_type
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class SeaIceSensitivity(ESMValToolDiagnostic):
    """
    Calculate sea ice sensitivity.
    """

    name = "Sea ice sensitivity"
    slug = "sea-ice-sensitivity"
    base_recipe = "recipe_seaice_sensitivity.yml"

    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": "siconc",
                            "experiment_id": "historical",
                            "table_id": "SImon",
                        },
                    ),
                    FacetFilter(
                        facets={
                            "variable_id": "tas",
                            "experiment_id": "historical",
                            "table_id": "Amon",
                        },
                    ),
                ),
                group_by=("experiment_id",),  # this does nothing, but group_by cannot be empty
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1979, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    RequireFacets(
                        "variable_id",
                        required_facets=("siconc", "tas"),
                        group_by=("source_id", "member_id", "grid_label"),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                    AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP6),
                    RequireFacets(
                        "variable_id",
                        required_facets=("areacello",),
                        group_by=("source_id", "grid_label"),
                    ),
                ),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets={
                            "branded_variable_name": "siconc_tavg-u-hxy-u",
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                            "realm": "seaIce",
                        },
                    ),
                    FacetFilter(
                        facets={
                            "branded_variable_name": "tas_tavg-h2m-hxy-u",
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                            "realm": "atmos",
                        },
                    ),
                ),
                group_by=("experiment_id",),  # this does nothing, but group_by cannot be empty
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1979, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    RequireFacets(
                        "variable_id",
                        required_facets=("siconc", "tas"),
                        group_by=("source_id", "variant_label", "grid_label"),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                    AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP7),
                    RequireFacets(
                        "variable_id",
                        required_facets=("areacello",),
                        group_by=("source_id", "grid_label"),
                    ),
                ),
            ),
        ),
    )
    facets = ("experiment_id", "source_id", "region", "metric")

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
                            "variable_id": ["areacella", "areacello", "siconc", "tas"],
                            "frequency": ["fx", "mon"],
                        },
                        remove_ensembles=True,
                        time_span=("1979", "2014"),
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
                            "variable_id": ["areacella", "areacello", "siconc", "tas"],
                            "branded_variable_name": [
                                "areacella_ti-u-hxy-u",
                                "areacello_ti-u-hxy-u",
                                "siconc_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1979", "2014"),
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
        cmip_source = get_cmip_source_type(input_files)
        recipe_variables = dataframe_to_recipe(input_files[cmip_source])

        if cmip_source == SourceDatasetType.CMIP7:
            # CMIP7: use per-variable additional_datasets to preserve correct branding_suffix
            recipe["datasets"] = []
            for diagnostic in recipe["diagnostics"].values():
                for var_name, variable in diagnostic.get("variables", {}).items():
                    short_name = variable.get("short_name", var_name)
                    if short_name in recipe_variables:
                        datasets = recipe_variables[short_name]["additional_datasets"]
                        for ds in datasets:
                            ds.pop("mip", None)
                            ds["timerange"] = "1979/2014"
                        variable["additional_datasets"] = datasets
        else:
            datasets = recipe_variables["tas"]["additional_datasets"]
            for dataset in datasets:
                dataset.pop("mip")
                dataset["timerange"] = "1979/2014"
            recipe["datasets"] = datasets

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
                "source_id",
                "region",
                "metric",
            ],
            "source_id": {},
            "region": {},
            "metric": {},
        }
        for region in "antarctic", "arctic":
            df = pd.read_csv(
                result_dir / "work" / region / "sea_ice_sensitivity_script" / "plotted_values.csv"
            )
            df = df.rename(columns={"Unnamed: 0": "source_id"}).drop(columns=["label"])
            metric_args[MetricCV.DIMENSIONS.value]["region"][region] = {}
            for metric in df.columns[1:]:
                metric_args[MetricCV.DIMENSIONS.value]["metric"][metric] = {}
            for row in df.itertuples(index=False):
                source_id = row.source_id
                metric_args[MetricCV.DIMENSIONS.value]["source_id"][source_id] = {}
                for metric, value in zip(df.columns[1:], row[1:]):
                    if source_id not in metric_args[MetricCV.RESULTS.value]:
                        metric_args[MetricCV.RESULTS.value][source_id] = {}
                    if region not in metric_args[MetricCV.RESULTS.value][source_id]:
                        metric_args[MetricCV.RESULTS.value][source_id][region] = {}
                    metric_args[MetricCV.RESULTS.value][source_id][region][metric] = value

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
