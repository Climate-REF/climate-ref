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