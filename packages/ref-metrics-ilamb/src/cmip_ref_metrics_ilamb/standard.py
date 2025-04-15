from pathlib import Path
from typing import Any

import ilamb3  # type: ignore
import ilamb3.regions as ilr  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
from ilamb3 import run

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import (
    DataRequirement,
    Metric,
    MetricExecutionDefinition,
    MetricExecutionResult,
)
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput
from cmip_ref_metrics_ilamb.datasets import (
    ILAMB_DATA_VERSION,
    ILAMBRegistryFile,
    build_ilamb_data_registry,
    registry_to_collection,
)


def _build_cmec_bundle(name: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a CMEC boundle from information in the dataframe.

    TODO: Migrate to use pycmec when ready.
    TODO: Add plots and html output.
    """
    ilamb_regions = ilr.Regions()
    bundle = {
        "DIMENSIONS": {
            "json_structure": ["region", "model", "metric", "statistic"],
            "region": {
                r: {
                    "LongName": "None" if r == "None" else ilamb_regions.get_name(r),
                    "Description": "Reference data extents" if r == "None" else ilamb_regions.get_name(r),
                    "Generator": "N/A" if r == "None" else ilamb_regions.get_source(r),
                }
                for r in df["region"].unique()
            },
            "model": {m: {"Description": m, "Source": m} for m in df["source"].unique() if m != "Reference"},
            "metric": {
                name: {
                    "Name": name,
                    "Abstract": "benchmark score",
                    "URI": [
                        "https://www.osti.gov/biblio/1330803",
                        "https://doi.org/10.1029/2018MS001354",
                    ],
                    "Contact": "forrest AT climatemodeling.org",
                }
            },
            "statistic": {s: {} for s in df["name"].unique()},
        },
        "RESULTS": {
            r: {
                m: {
                    name: {
                        s: float(
                            df[(df["source"] == m) & (df["region"] == r) & (df["name"] == s)].iloc[0]["value"]
                        )
                        for s in df["name"].unique()
                    }
                }
                for m in df["source"].unique()
                if m != "Reference"
            }
            for r in df["region"].unique()
        },
    }
    return bundle


def _form_bundles(key: str, df: pd.DataFrame) -> tuple[CMECMetric, CMECOutput]:
    """
    Create the output bundles (really a lift to make Ruff happy with the size of run()).
    """
    metric_bundle = _build_cmec_bundle(key, df)
    output_bundle = CMECOutput.create_template()
    return CMECMetric.model_validate(metric_bundle), CMECOutput.model_validate(output_bundle)


def _set_ilamb3_options(registry_file: ILAMBRegistryFile) -> None:
    """
    Set options for ILAMB based on which registry file is being used.
    """
    ilamb3.conf.reset()
    reg = build_ilamb_data_registry(registry_file, ILAMB_DATA_VERSION)
    ilamb_regions = ilr.Regions()
    if registry_file == "ilamb.txt":
        ilamb_regions.add_netcdf(reg.fetch("regions/GlobalLand.nc"))
        ilamb_regions.add_netcdf(reg.fetch("regions/Koppen_coarse.nc"))
        ilamb3.conf.set(regions=["global", "tropical"])


def _measure_facets(registry_file: ILAMBRegistryFile) -> list[str]:
    """
    Set options for ILAMB based on which registry file is being used.
    """
    if registry_file == "ilamb.txt":
        return ["areacella", "sftlf"]
    return []


def _load_csv_and_merge(output_directory: Path) -> pd.DataFrame:
    """
    Load individual csv scalar data and merge into a dataframe.
    """
    df = pd.concat(
        [pd.read_csv(f, keep_default_na=False, na_values=["NaN"]) for f in output_directory.glob("*.csv")]
    ).drop_duplicates(subset=["source", "region", "analysis", "name"])
    df["name"] = df["name"] + " [" + df["units"] + "]"
    return df


class ILAMBStandard(Metric):
    """
    Apply the standard ILAMB analysis with respect to a given reference dataset.
    """

    def __init__(
        self,
        registry_file: ILAMBRegistryFile,
        metric_name: str,
        sources: dict[str, str],
        **ilamb_kwargs: Any,
    ):
        # Setup the metric
        if len(sources) != 1:
            raise ValueError("Only single source ILAMB metrics have been implemented.")
        self.variable_id = next(iter(sources.keys()))
        if "sources" not in ilamb_kwargs:  # pragma: no cover
            ilamb_kwargs["sources"] = sources
        if "relationships" not in ilamb_kwargs:
            ilamb_kwargs["relationships"] = {}
        self.ilamb_kwargs = ilamb_kwargs

        # REF stuff
        self.name = metric_name
        self.slug = self.name.lower().replace(" ", "-")
        self.data_requirements = (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": [
                                self.variable_id,
                                *ilamb_kwargs.get("relationships", {}).keys(),
                                *_measure_facets(registry_file),
                            ]
                        }
                    ),
                    FacetFilter(facets={"frequency": ["mon", "fx"]}),
                    FacetFilter(facets={"experiment_id": ("historical", "land-hist")}),
                ),
                group_by=("experiment_id",),
            ),
        )

        # Setup ILAMB data and options
        self.registry_file = registry_file
        self.ilamb_data = registry_to_collection(
            build_ilamb_data_registry(self.registry_file, ILAMB_DATA_VERSION)
        )

    def run(self, definition: MetricExecutionDefinition) -> MetricExecutionResult:
        """
        Run the ILAMB standard analysis.
        """
        plt.rcParams.update({"figure.max_open_warning": 0})
        _set_ilamb3_options(self.registry_file)
        ref_datasets = self.ilamb_data.datasets.set_index(self.ilamb_data.slug_column)
        run.run_simple(
            ref_datasets,
            self.slug,
            definition.metric_dataset[SourceDatasetType.CMIP6].datasets,
            definition.output_directory,
            **self.ilamb_kwargs,
        )
        df = _load_csv_and_merge(definition.output_directory)
        metric_bundle, output_bundle = _form_bundles(definition.dataset_key, df)
        return MetricExecutionResult.build_from_output_bundle(
            definition, cmec_output_bundle=output_bundle, cmec_metric_bundle=metric_bundle
        )
