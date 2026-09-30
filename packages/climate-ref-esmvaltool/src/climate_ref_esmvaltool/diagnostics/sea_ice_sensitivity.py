import copy
from pathlib import Path

import pandas
import pandas as pd

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireFacets,
    RequireTimerange,
    SelectFirstMember,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.metric_values.typing import FileDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput, OutputCV
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
    version = 2

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
                    # The diagnostic script expects a single member per model; the REF
                    # otherwise feeds every ingested member into this one execution.
                    SelectFirstMember(member_facet="member_id", group_by=("source_id",)),
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
                            "branded_variable": "siconc_tavg-u-hxy-u",
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                        },
                    ),
                    FacetFilter(
                        facets={
                            "branded_variable": "tas_tavg-h2m-hxy-u",
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
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
                    # The diagnostic script expects a single member per model; the REF
                    # otherwise feeds every ingested member into this one execution.
                    SelectFirstMember(member_facet="variant_label", group_by=("source_id",)),
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
    files = tuple(
        FileDefinition(
            file_pattern=f"plots/{region}/sea_ice_sensitivity_script/png/*.png",
            dimensions={"region": region},
        )
        for region in ("arctic", "antarctic")
    ) + tuple(
        FileDefinition(
            # ESMValTool v2.15 registers this file in its provenance without the .csv suffix.
            file_pattern=f"work/{region}/sea_ice_sensitivity_script/data_values*",
            dimensions={"region": region},
        )
        for region in ("arctic", "antarctic")
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
                            "branded_variable": [
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
        for variable in recipe_variables.values():
            for dataset in variable["additional_datasets"]:
                dataset.pop("mip", None)
                dataset["timerange"] = "1979/2014"

        # The REF supplies the models from the solve and has no ESMValTool observations available,
        # so drop the datasets the recipe carries and the anchors holding them.
        for key in (
            "datasets",
            "model_defaults",
            "model_datasets",
            "obs_defaults",
            "tasa_obs",
            "arctic_siconc_obs",
            "antarctic_siconc_obs",
        ):
            recipe.pop(key, None)

        for diagnostic in recipe["diagnostics"].values():
            variables = diagnostic["variables"]
            for name in list(variables):
                if name in recipe_variables:
                    variables[name]["additional_datasets"] = copy.deepcopy(
                        recipe_variables[name]["additional_datasets"]
                    )
                else:
                    del variables[name]

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
        dimensions = metric_args[MetricCV.DIMENSIONS.value]
        for region in "antarctic", "arctic":
            df = pd.read_csv(
                result_dir / "work" / region / "sea_ice_sensitivity_script" / "data_values.csv",
                header=[0, 1, 2],
                index_col=0,
            )
            is_type = df.columns.get_level_values("statistic") == "type"
            is_model = df.loc[:, is_type].iloc[:, 0] == "model"
            # The REF solve covers a single period, so keep the last one if the script adds more.
            period = str(df.columns.get_level_values("period")[-1])
            values = df.loc[is_model, period]

            dimensions["region"][region] = {}
            for regression, statistic in values.columns:
                dimensions["metric"][f"{regression}_{statistic}"] = {}
            for source_id, row in values.iterrows():
                dimensions["source_id"][source_id] = {}
                results = metric_args[MetricCV.RESULTS.value].setdefault(source_id, {}).setdefault(region, {})
                for (regression, statistic), value in row.items():
                    results[f"{regression}_{statistic}"] = float(value)

        # Restore the data_values suffix
        data = output_args[OutputCV.DATA.value]
        for key in [key for key in data if key.endswith("/data_values")]:
            entry = data.pop(key)
            entry[OutputCV.FILENAME.value] = f"{entry[OutputCV.FILENAME.value]}.csv"
            data[f"{key}.csv"] = entry

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
