import copy
from pathlib import Path

import numpy as np
import pandas
import xarray

from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    PartialDateTime,
    RequireTimerange,
)
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.esgf import CMIP6Request, CMIP7Request, Obs4MIPsRequest
from climate_ref_core.metric_values.typing import FileDefinition, SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric, MetricCV
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic, fillvalues_to_nan
from climate_ref_esmvaltool.recipe import dataframe_to_recipe
from climate_ref_esmvaltool.types import MetricBundleArgs, OutputBundleArgs, Recipe

REGIONS = (
    "Arabian-Peninsula",
    "Arabian-Sea",
    "Arctic-Ocean",
    "Bay-of-Bengal",
    "C.Australia",
    "C.North-America",
    "Caribbean",
    "Central-Africa",
    "E.Antarctica",
    "E.Asia",
    "E.Australia",
    "E.C.Asia",
    "E.Europe",
    "E.North-America",
    "E.Siberia",
    "E.Southern-Africa",
    "Equatorial.Atlantic-Ocean",
    "Equatorial.Indic-Ocean",
    "Equatorial.Pacific-Ocean",
    "Greenland/Iceland",
    "Madagascar",
    "Mediterranean",
    "N.Atlantic-Ocean",
    "N.Australia",
    "N.Central-America",
    "N.E.North-America",
    "N.E.South-America",
    "N.Eastern-Africa",
    "N.Europe",
    "N.Pacific-Ocean",
    "N.South-America",
    "N.W.North-America",
    "N.W.South-America",
    "New-Zealand",
    "Russian-Arctic",
    "Russian-Far-East",
    "S.Asia",
    "S.Atlantic-Ocean",
    "S.Australia",
    "S.Central-America",
    "S.E.Asia",
    "S.E.South-America",
    "S.Eastern-Africa",
    "S.Indic-Ocean",
    "S.Pacific-Ocean",
    "S.South-America",
    "S.W.South-America",
    "Sahara",
    "South-American-Monsoon",
    "Southern-Ocean",
    "Tibetan-Plateau",
    "W.Antarctica",
    "W.C.Asia",
    "W.North-America",
    "W.Siberia",
    "W.Southern-Africa",
    "West&Central-Europe",
    "Western-Africa",
)


class RegionalHistoricalAnnualCycle(ESMValToolDiagnostic):
    """
    Plot regional historical annual cycle of climate variables.
    """

    name = "Regional historical annual cycle of climate variables"
    slug = "regional-historical-annual-cycle"
    base_recipe = "ref/recipe_ref_annual_cycle_region.yml"

    variables = (
        "hus",
        "pr",
        "psl",
        "tas",
        "ua",
    )

    data_requirements = (
        (
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
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2009, 12),
                    ),
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
                                "hus_tavg-p19-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "psl_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "ua_tavg-p19-hxy-air",
                            ),
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "realm": "atmos",
                        },
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2009, 12),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": (
                                "psl",
                                "ua",
                            ),
                            "source_id": "ERA-5",
                            "frequency": "mon",
                        },
                    ),
                ),
                group_by=("source_id",),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2009, 12),
                    ),
                ),
                # TODO: Add obs4MIPs datasets once available and working:
                #
                # obs4MIPs dataset that cannot be ingested (https://github.com/Climate-REF/climate-ref/issues/260):
                # - GPCP-V2.3: pr
                #
                # Not yet available on obs4MIPs:
                # - ERA5: hus
                # - HadCRUT5_ground_5.0.1.0-analysis: tas
            ),
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
                            "experiment_id": ["historical"],
                            "frequency": ["fx", "mon"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", *variables],
                        },
                        remove_ensembles=True,
                        time_span=("1980", "2009"),
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
                            "experiment_id": ["historical"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", *variables],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "hus_tavg-p19-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "psl_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "ua_tavg-p19-hxy-air",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1980", "2009"),
                    ),
                ),
            ),
            TestCase(
                name="obs4mips",
                description="Test with obs4MIPs data.",
                requests=(
                    Obs4MIPsRequest(
                        slug="obs4mips",
                        facets={
                            "project": "obs4MIPs",
                            "source_id": "ERA-5",
                            "variable_id": [
                                "psl",
                                "ua",
                            ],
                        },
                        remove_ensembles=False,
                        time_span=("1980", "2009"),
                    ),
                ),
            ),
        ),
    )

    facets = ()
    files = tuple(
        FileDefinition(
            file_pattern=f"plots/anncyc-{region}/allplots/*_{var_name}_*.png",
            dimensions={
                "region": region,
                "variable_id": var_name,
                "statistic": "mean",
            },
        )
        for var_name in variables
        for region in REGIONS
    )
    series = tuple(
        SeriesDefinition(
            file_pattern=f"anncyc-{region}/allplots/*_{var_name}_*.nc",
            sel={"dim0": 0},  # Select the model and not the observation.
            dimensions=(
                {
                    "region": region,
                    "variable_id": var_name,
                    "statistic": "mean",
                }
            ),
            values_name=var_name,
            index_name="month_number",
            attributes=[],
        )
        for var_name in variables
        for region in REGIONS
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Extra datasets that are not published as obs4MIPs.
        extra_datasets: dict[str, list[dict[str, str | int]]] = {
            "tas": [
                {
                    "dataset": "HadCRUT5",
                    "project": "OBS",
                    "tier": 2,
                    "type": "ground",
                    "version": "5.0.1.0-analysis",
                },
            ],
            "pr": [
                {
                    "dataset": "GPCP-V2.3",
                    "project": "obs4MIPs",
                },
            ],
            "hus": [
                {
                    "dataset": "ERA5",
                    "project": "native6",
                    "tier": 3,
                    "type": "reanaly",
                    "version": "v1",
                },
            ],
        }

        # Remove the unused regions alias.
        recipe.pop("regions")
        # Update the dataset.
        recipe.pop("datasets")
        recipe_variables = dataframe_to_recipe(next(iter(input_files.values())))
        source_type = next(iter(input_files.keys()))
        project = str(source_type).split(".")[1]
        for diagnostic_name, diagnostic in dict(recipe["diagnostics"]).items():
            for variable_group, variable in dict(diagnostic["variables"]).items():
                short_name = variable_group.split("_")[0]
                datasets = []
                if short_name in recipe_variables:
                    dataset = copy.deepcopy(recipe_variables[short_name]["additional_datasets"][0])
                    dataset.pop("timerange", None)
                    datasets.append(dataset)
                if project == "obs4MIPs" and short_name in extra_datasets:
                    datasets.extend(copy.deepcopy(extra_datasets[short_name]))
                if datasets:
                    variable["additional_datasets"] = datasets
                else:
                    # If no data is available for a diagnostic, skip it.
                    recipe["diagnostics"].pop(diagnostic_name)
                    continue
            for script_settings in diagnostic["scripts"].values():
                if project == "obs4MIPs":
                    label = "{dataset}"
                else:
                    label = "{dataset}_{ensemble}_{grid}"
                script_settings["plot_filename"] = f"{{plot_type}}_{{real_name}}_{label}_{{shape_id}}"
                for plot_settings in script_settings["plots"].values():
                    plot_settings["plot_kwargs"]["default"]["label"] = label


class RegionalHistoricalTimeSeries(RegionalHistoricalAnnualCycle):
    """
    Plot regional historical mean and anomaly of climate variables.
    """

    name = "Regional historical mean and anomaly of climate variables"
    slug = "regional-historical-timeseries"
    base_recipe = "ref/recipe_ref_timeseries_region.yml"

    variables = (
        "hus",
        "pr",
        "psl",
        "tas",
        "ua",
    )

    data_requirements = (
        (
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
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2014, 12),
                    ),
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
                                "hus_tavg-p19-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "psl_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "ua_tavg-p19-hxy-air",
                            ),
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "region": "glb",
                            "realm": "atmos",
                        },
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": (
                                "psl",
                                "ua",
                            ),
                            "source_id": "ERA-5",
                            "frequency": "mon",
                        },
                    ),
                ),
                group_by=("source_id",),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2014, 12),
                    ),
                ),
                # TODO: Add obs4MIPs datasets once available and working:
                #
                # obs4MIPs dataset that cannot be ingested (https://github.com/Climate-REF/climate-ref/issues/260):
                # - GPCP-V2.3: pr
                #
                # Not yet available on obs4MIPs:
                # - ERA5: hus
                # - HadCRUT5_ground_5.0.1.0-analysis: tas
            ),
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
                            "experiment_id": ["historical"],
                            "frequency": ["fx", "mon"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", *variables],
                        },
                        remove_ensembles=True,
                        time_span=("1980", "2014"),
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
                            "experiment_id": ["historical"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", *variables],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "hus_tavg-p19-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "psl_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "ua_tavg-p19-hxy-air",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1980", "2014"),
                    ),
                ),
            ),
            TestCase(
                name="obs4mips",
                description="Test with obs4MIPs data.",
                requests=(
                    Obs4MIPsRequest(
                        slug="obs4mips",
                        facets={
                            "project": "obs4MIPs",
                            "source_id": "ERA-5",
                            "variable_id": [
                                "psl",
                                "ua",
                            ],
                        },
                        remove_ensembles=False,
                        time_span=("1980", "2014"),
                    ),
                ),
            ),
        ),
    )

    files = tuple(
        FileDefinition(
            file_pattern=f"plots/{diagnostic}-{region}/allplots/*_{var_name}_*.png",
            dimensions={
                "region": region,
                "variable_id": var_name,
                "statistic": ("mean" if diagnostic == "timeseries_abs" else "mean anomaly"),
            },
        )
        for var_name in variables
        for region in REGIONS
        for diagnostic in ["timeseries_abs", "timeseries"]
    )
    series = tuple(
        SeriesDefinition(
            file_pattern=f"{diagnostic}-{region}/allplots/*_{var_name}_*.nc",
            sel={"dim0": 0},
            dimensions=(
                {
                    "region": region,
                    "variable_id": var_name,
                    "statistic": ("mean" if diagnostic == "timeseries_abs" else "mean anomaly"),
                }
            ),
            values_name=var_name,
            index_name="time",
            attributes=[],
        )
        for var_name in variables
        for region in REGIONS
        for diagnostic in ["timeseries_abs", "timeseries"]
    )


class RegionalHistoricalTrend(ESMValToolDiagnostic):
    """
    Plot regional historical trend of climate variables.
    """

    name = "Regional historical trend of climate variables"
    slug = "regional-historical-trend"
    base_recipe = "ref/recipe_ref_trend_regions.yml"

    variables = (
        "hus",
        "pr",
        "psl",
        "tas",
        "ua",
    )

    data_requirements = (
        (
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
                group_by=("source_id", "member_id", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2009, 12),
                    ),
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
                                "hus_tavg-p19-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "psl_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "ua_tavg-p19-hxy-air",
                            ),
                            "experiment_id": "historical",
                            "frequency": "mon",
                            "realm": "atmos",
                        },
                    ),
                ),
                group_by=("source_id", "variant_label", "grid_label"),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2009, 12),
                    ),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                ),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": (
                                "psl",
                                "tas",
                                "ua",
                            ),
                            "source_id": "ERA-5",
                            "frequency": "mon",
                        },
                    ),
                ),
                group_by=("source_id",),
                constraints=(
                    RequireTimerange(
                        group_by=("instance_id",),
                        start=PartialDateTime(1980, 1),
                        end=PartialDateTime(2009, 12),
                    ),
                ),
                # TODO: Add obs4MIPs datasets once available and working:
                #
                # obs4MIPs dataset that cannot be ingested (https://github.com/Climate-REF/climate-ref/issues/260):
                # - GPCP-V2.3: pr
                #
                # Not yet available on obs4MIPs:
                # - ERA5: hus, pr
                # - HadCRUT5_ground_5.0.1.0-analysis: tas
            ),
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
                            "experiment_id": ["historical"],
                            "frequency": ["fx", "mon"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", *variables],
                        },
                        remove_ensembles=True,
                        time_span=("1980", "2009"),
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
                            "experiment_id": ["historical"],
                            "source_id": "CanESM5",
                            "variable_id": ["areacella", *variables],
                            "branded_variable": [
                                "areacella_ti-u-hxy-u",
                                "hus_tavg-p19-hxy-u",
                                "pr_tavg-u-hxy-u",
                                "psl_tavg-u-hxy-u",
                                "tas_tavg-h2m-hxy-u",
                                "ua_tavg-p19-hxy-air",
                            ],
                            "variant_label": "r1i1p1f1",
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                        time_span=("1980", "2009"),
                    ),
                ),
            ),
            TestCase(
                name="obs4mips",
                description="Test with obs4MIPs data.",
                requests=(
                    Obs4MIPsRequest(
                        slug="obs4mips",
                        facets={
                            "project": "obs4MIPs",
                            "source_id": "ERA-5",
                            "variable_id": [
                                "psl",
                                "tas",
                                "ua",
                            ],
                        },
                        remove_ensembles=False,
                        time_span=("1980", "2009"),
                    ),
                ),
            ),
        ),
    )

    facets = ("grid_label", "member_id", "variant_label", "source_id", "variable_id", "region", "metric")
    files = tuple(
        FileDefinition(
            file_pattern=f"plots/{var_name}_trends/plot/seaborn_barplot.png",
            dimensions={
                "variable_id": var_name,
                "statistic": "trend",
            },
        )
        for var_name in ("hus200", "pr", "psl", "tas", "ua200")
    ) + tuple(
        FileDefinition(
            file_pattern=f"work/{var_name}_trends/plot/seaborn_barplot.nc",
            dimensions={
                "variable_id": var_name,
                "statistic": "trend",
            },
        )
        for var_name in ("hus200", "pr", "psl", "tas", "ua200")
    )

    @staticmethod
    def update_recipe(
        recipe: Recipe,
        input_files: dict[SourceDatasetType, pandas.DataFrame],
    ) -> None:
        """Update the recipe."""
        # Extra datasets that are not published as CMIP6-style obs4MIPs.
        extra_datasets: dict[str, list[dict[str, str | int]]] = {
            "tas": [
                {
                    "dataset": "HadCRUT5",
                    "project": "OBS",
                    "tier": 2,
                    "type": "ground",
                    "version": "5.0.1.0-analysis",
                },
            ],
            "pr": [
                {
                    "dataset": "GPCP-V2.3",
                    "project": "obs4MIPs",
                },
                {
                    "dataset": "ERA5",
                    "project": "native6",
                    "type": "reanaly",
                    "tier": 3,
                    "version": "v1",
                },
            ],
            "hus": [
                {
                    "dataset": "ERA5",
                    "project": "native6",
                    "tier": 3,
                    "type": "reanaly",
                    "version": "v1",
                },
            ],
        }

        # Update the datasets.
        recipe.pop("datasets")
        recipe_variables = dataframe_to_recipe(next(iter(input_files.values())))
        source_type = next(iter(input_files.keys()))
        project = str(source_type).split(".")[1]
        for diagnostic_name, diagnostic in dict(recipe["diagnostics"]).items():
            for short_name, variable in dict(diagnostic["variables"]).items():
                datasets = []
                if short_name in recipe_variables:
                    dataset = recipe_variables[short_name]["additional_datasets"][0]
                    dataset.pop("timerange", None)
                    datasets.append(dataset)
                if project == "obs4MIPs" and short_name in extra_datasets:
                    datasets.extend(extra_datasets[short_name])
                if datasets:
                    variable["additional_datasets"] = datasets
                else:
                    # If no data is available for a diagnostic, skip it.
                    recipe["diagnostics"].pop(diagnostic_name)

    @staticmethod
    def format_result(
        result_dir: Path,
        execution_dataset: ExecutionDatasetCollection,
        metric_args: MetricBundleArgs,
        output_args: OutputBundleArgs,
    ) -> tuple[CMECMetric, CMECOutput]:
        """Format the result."""
        metric_args[MetricCV.DIMENSIONS.value] = {
            "json_structure": ["variable_id", "region", "metric"],
            "variable_id": {},
            "region": {},
            "metric": {"trend": {}},
        }
        for file in result_dir.glob("work/*_trends/plot/seaborn_barplot.nc"):
            ds = xarray.open_dataset(file)
            dataset_collection = next(iter(execution_dataset.values()))
            for source_id in dataset_collection.source_id.unique():
                select = source_id == np.array([s.strip() for s in ds.dataset.values.astype(str).tolist()])
                ds.isel(dim0=select)
                variable_id = next(iter(ds.data_vars.keys()))
                metric_args[MetricCV.DIMENSIONS.value]["variable_id"][variable_id] = {}
                metric_args[MetricCV.RESULTS.value][variable_id] = {}
                for region_value, trend_value in zip(
                    ds.shape_id.astype(str).values, fillvalues_to_nan(ds[variable_id].values)
                ):
                    region = region_value.strip()
                    trend = float(trend_value)
                    if region not in metric_args[MetricCV.DIMENSIONS.value]["region"]:
                        metric_args[MetricCV.DIMENSIONS.value]["region"][region] = {}
                    metric_args[MetricCV.RESULTS.value][variable_id][region] = {"trend": trend}

        return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
