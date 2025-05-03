import datetime
import json
from collections.abc import Iterable
from typing import Any

from loguru import logger

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import (
    CommandLineMetric,
    DataRequirement,
    MetricExecutionDefinition,
    MetricExecutionResult,
)
from cmip_ref_metrics_pmp.pmp_driver import build_glob_pattern, build_pmp_command, process_json_result


class ENSO(CommandLineMetric):
    """
    Calculate the ENSO performance metrics for a dataset
    """
    
    def __init__(self, metrics_collection: str) -> None:
        self.name = metrics_collection
        self.slug = metrics_collection.lower()
        self.metrics_collection = metrics_collection
        
        def _get_data_requirements(
            metrics_collection: str,
            extra_experiments: str | tuple[str, ...] | list[str] = (),
            remove_experiments: str | tuple[str, ...] | list[str] = (),
        ) -> tuple[DataRequirement, DataRequirement]:
            
            if metrics_collection == "ENSO_perf":
                model_variables = ("pr", "ts", "taux")
                obs_sources = ("GPCP", "ERA5")
            elif metrics_collection == "ENSO_tel":
                model_variables = ("pr", "ts")
                obs_sources = ("GPCP", "ERA5")
            elif metrics_collection == "ENSO_proc":
                model_variables = ("ts", "taux", "hfls", "hfss", "rlds", "rlus", "rsds", "rsus")
                obs_sources = ("GPCP", "ERA5", "TropFlux")
            else:
                raise ValueError(f"Unknown metrics collection: {metrics_collection}. Valid options are: ENSO_perf, ENSO_tel, ENSO_proc")
            
            obs_variables = model_variables
            
            filters = [
                FacetFilter(
                    facets={
                        "frequency": "mon",
                        "experiment_id": ("historical", *extra_experiments),
                        "variable_id": model_variables,
                    }
                )
            ]
            
            if remove_experiments:
                filters.append(FacetFilter(facets={"experiment_id": remove_experiments}, keep=False))
            
            return (
                DataRequirement(
                    source_type=SourceDatasetType.obs4MIPs,
                    filters=(
                        FacetFilter(facets={"source_id": obs_sources, "variable_id": obs_variables}),
                    ),
                    group_by=("source_id", "variable_id"),
                ),
                DataRequirement(
                    source_type=SourceDatasetType.CMIP6,
                    filters=tuple(filters),
                    group_by=("source_id", "experiment_id", "variant_label", "member_id"),
                ),
            )