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
from climate_ref_core.metric_values.typing import FileDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import (
    ESMValToolDiagnostic,
    fillvalues_to_nan,
    get_cmip_source_type,
)
from climate_ref_esmvaltool.recipe import dataframe_to_recipe, get_child_and_parent_dataset
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe


class TransientClimateResponseEmissions(ESMValToolDiagnostic):
    """
    Calculate the global mean Transient Climate Response to Cumulative CO2 Emissions.
    """

    name = "Transient Climate Response to Cumulative CO2 Emissions"
    slug = "transient-climate-response-emissions"
    base_recipe = "recipe_tcre.yml"

    variables = (
        "tas",
        "fco2antt",
    )

    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": "tas",
                            "experiment_id": "esm-1pctCO2",
                            "table_id": "Amon",
                        },
                    ),
                ),
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    AddParentDataset.from_defaults(SourceDatasetType.CMIP6),
                    AddSupplementaryDataset(
                        supplementary_facets={
                            "variable_id": "fco2antt",
                            "experiment_id": "esm-1pctCO2",
                        },
                        matching_facets=(
                            "source_id",
                            "member_id",
                            "table_id",
                            "grid_label",
                        ),
                        optional_matching_facets=("version",),
                    ),
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireFacets("variable_id", ("tas", "fco2antt")),
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
                            "branded_variable": (
                                "fco2antt_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                            ),
                            "experiment_id": "esm-1pctCO2",
                            "frequency": "mon",
                            "region": "glb",
                        },
                    ),
                    FacetFilter(
                        facets={
                            "branded_variable": "tas_tavg-h2m-hxy-u",
                            "experiment_id": "esm-piControl",
                            "frequency": "mon",
                            "region": "glb",
                        },
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    RequireFacets("experiment_id", ("esm-1pctCO2", "esm-piControl")),
                    RequireFacets("variable_id", variables),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
    )
    facets = ("grid_label", "member_id", "variant_label", "source_id", "region", "metric")
    # TODO: the ESMValTool diagnostic script does not save the data for the timeseries.
    series = tuple()
    files = (
        FileDefinition(
            file_pattern="plots/tcre/calculate_tcre/*.png",
            dimensions={"statistic": "tcre"},
        ),
        FileDefinition(
            file_pattern="work/tcre/calculate_tcre/tcre.nc",
            dimensions={"metric": "tcre"},
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
                            "experiment_id": ["esm-1pctCO2", "esm-piControl"],
                            "source_id": "MPI-ESM1-2-LR",
                            "variable_id": ["areacella", "fco2antt", "tas"],
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
                            "experiment_id": ["esm-1pctCO2", "esm-piControl"],
                            "source_id": "MPI-ESM1-2-LR",
                            "variable_id": ["areacella", "fco2antt", "tas"],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "fco2antt_tavg-u-hxy-u",
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
        # Prepare updated datasets section in recipe. It contains three
        # datasets, "tas" and "fco2antt" for the "esm-1pctCO2" and just "tas"
        # for the "esm-piControl" experiment.
        cmip_source = get_cmip_source_type(input_files)
        df = input_files[cmip_source]
        tas_esm_1pctCO2, tas_esm_piControl = get_child_and_parent_dataset(
            df[df.variable_id == "tas"],
            parent_experiment="esm-piControl",
            child_duration_in_years=65,
            parent_offset_in_years=0,
            parent_duration_in_years=65,
        )
        recipe_variables = dataframe_to_recipe(df[df.variable_id == "fco2antt"])

        fco2antt_esm_1pctCO2 = next(
            ds for ds in recipe_variables["fco2antt"]["additional_datasets"] if ds["exp"] == "esm-1pctCO2"
        )
        fco2antt_esm_1pctCO2["timerange"] = tas_esm_1pctCO2["timerange"]

        recipe["diagnostics"]["tcre"]["variables"] = {
            "tas_esm-1pctCO2": {
                "short_name": "tas",
                "preprocessor": "global_annual_mean_anomaly",
                "additional_datasets": [tas_esm_1pctCO2],
            },
            "tas_esm-piControl": {
                "short_name": "tas",
                "preprocessor": "global_annual_mean_anomaly",
                "additional_datasets": [tas_esm_piControl],
            },
            "fco2antt": {
                "preprocessor": "global_cumulative_sum",
                "additional_datasets": [fco2antt_esm_1pctCO2],
            },
        }
        recipe["diagnostics"].pop("barplot")

        # Update descriptions.
        dataset = tas_esm_1pctCO2["dataset"]
        ensemble = tas_esm_1pctCO2["ensemble"]
        settings = recipe["diagnostics"]["tcre"]["scripts"]["calculate_tcre"]
        settings["caption"] = (
            settings["caption"].replace("MPI-ESM1-2-LR", dataset).replace("r1i1p1f1", ensemble)
        )
        settings["pyplot_kwargs"]["title"] = (
            settings["pyplot_kwargs"]["title"].replace("MPI-ESM1-2-LR", dataset).replace("r1i1p1f1", ensemble)
        )

    @staticmethod
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        tcre_ds = xarray.open_dataset(result_dir / "work" / "tcre" / "calculate_tcre" / "tcre.nc")
        tcre = float(fillvalues_to_nan(tcre_ds["tcre"].values)[0])

        # Update the diagnostic bundle arguments with the computed diagnostics.
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": [
                "region",
                "metric",
            ],
            "region": {"global": {}},
            "metric": {"tcre": {}},
        }
        metric_args[MetricCV.RESULTS.value] = {
            "global": {
                "tcre": tcre,
            },
        }
        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
