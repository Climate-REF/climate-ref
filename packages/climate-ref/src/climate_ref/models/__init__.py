"""
Declaration of the models used by the REF.

These models are used to represent the data that is stored in the database.
"""

from climate_ref.models.base import Base, Table
from climate_ref.models.dataset import Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    ExecutionOutput,
)
from climate_ref.models.metric_value import (
    MetricValue,
    ScalarMetricValue,
    SeriesIndex,
    SeriesMetricValue,
)
from climate_ref.models.provider import Provider

__all__ = [
    "Base",
    "Dataset",
    "Diagnostic",
    "Execution",
    "ExecutionGroup",
    "ExecutionOutput",
    "MetricValue",
    "Provider",
    "ScalarMetricValue",
    "SeriesIndex",
    "SeriesMetricValue",
    "Table",
]
