"""
Metric Values

A metric is a single statistical evaluation contained within a diagnostic.
A diagnostic may consist of more than one metric.

Examples include bias, root mean squared error (RMSE), Earth Mover's Distance,
phase/timing of the seasonal cycle, amplitude of the seasonal cycle, spatial or temporal correlations,
interannual variability.
Not all metrics are useful for all variables or should be used with every observationally constrained dataset.
Each metric may be converted into a performance score.
"""

from .typing import MetricValueKind, ScalarMetricValue, SeriesMetricValue

__all__ = ["MetricValueKind", "ScalarMetricValue", "SeriesMetricValue"]
