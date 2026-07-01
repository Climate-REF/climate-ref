"""
Outlier detection for scalar metric values.

Outlier detection is *read-model* logic, not presentation policy: whether a value is flagged
determines which rows a view returns and what the collection-level counts mean. It therefore
lives in ``climate_ref`` (the source of truth) rather than in any single consumer.

The algorithm is source-id-aware IQR: within each ``group_by`` group, IQR bounds are computed on
the per-``source_id`` mean value (so each model is weighted equally regardless of ensemble size),
then applied to individual values. ``"Reference"`` values are never flagged; non-finite values
(NaN/inf) are always flagged.

This is a framework-agnostic port of the logic previously living in the ref-app backend; it takes
and returns plain data and imports no web framework.
"""

import math
import statistics
from collections.abc import Sequence

import attrs
import pandas as pd

from climate_ref.models.metric_value import ScalarMetricValue


@attrs.frozen(kw_only=True)
class OutlierPolicy:
    """
    Configuration for scalar outlier detection.

    The defaults reproduce the behaviour users currently see through the ref-app API
    (``factor=10.0``, source-id-aware IQR grouped by ``("statistic", "metric")``).
    """

    method: str = "iqr"
    """Detection method. ``"off"`` disables detection; ``"iqr"`` enables it."""

    factor: float = 10.0
    """Multiplier on the IQR to set the outlier bounds (``Q1 - factor*IQR``, ``Q3 + factor*IQR``)."""

    min_n: int = 4
    """Minimum number of source_ids (or values) in a group required to run detection."""

    group_by: tuple[str, ...] = ("statistic", "metric")
    """CV dimensions to group by before computing bounds. Missing dimensions are ignored."""

    @property
    def enabled(self) -> bool:
        """Whether detection should run."""
        return self.method == "iqr"


@attrs.frozen(kw_only=True)
class AnnotatedScalar:
    """A scalar ORM row paired with its outlier verdict."""

    value: ScalarMetricValue
    is_outlier: bool
    verification_status: str  # "verified" | "unverified"


def _flag_outliers_iqr(values: Sequence[float], factor: float, min_n: int) -> list[bool]:
    """Flag outliers with a plain IQR test (fallback when no source_id is available)."""
    n = len(values)
    if n < min_n:
        return [False] * n
    quantiles = statistics.quantiles(values, n=4, method="inclusive")
    q1, q3 = quantiles[0], quantiles[2]
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return [v < lower or v > upper for v in values]


def _iqr_bounds_by_source_id(df: pd.DataFrame, factor: float, min_n: int) -> tuple[float, float] | None:
    """IQR bounds computed on per-source_id means (equal model weighting)."""
    if "source_id" not in df.columns:
        return None
    non_reference = df[df["source_id"] != "Reference"]
    source_id_means = non_reference.groupby("source_id")["value"].mean()
    if len(source_id_means) < min_n:
        return None
    quantiles = statistics.quantiles(source_id_means.tolist(), n=4, method="inclusive")
    q1, q3 = quantiles[0], quantiles[2]
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def _is_finite(x: object) -> bool:
    return isinstance(x, int | float) and not math.isinf(x) and not math.isnan(x)


def detect_scalar_outliers(
    scalar_values: Sequence[ScalarMetricValue],
    policy: OutlierPolicy,
) -> tuple[list[AnnotatedScalar], int]:
    """
    Annotate scalar values with outlier verdicts.

    Parameters
    ----------
    scalar_values
        The full (unpaginated) set of scalar rows for the query scope. Detection must run over
        the whole set so IQR bounds are globally consistent.
    policy
        Detection configuration.

    Returns
    -------
    :
        A tuple of (annotated values in input order, total number of outliers).
    """
    if not scalar_values:
        return [], 0

    if not policy.enabled:
        return (
            [
                AnnotatedScalar(value=sv, is_outlier=False, verification_status="verified")
                for sv in scalar_values
            ],
            0,
        )

    df = pd.DataFrame(
        [{"scalar_value": sv, "value": sv.value, **sv.dimensions, "id": sv.id} for sv in scalar_values]
    )
    group_by = [g for g in policy.group_by if g in df.columns]

    verdict_by_id: dict[int, bool] = {}
    groups = df.groupby(list(group_by)) if group_by else [(None, df)]
    for _, group in groups:
        if "source_id" in group.columns and len(group) >= policy.min_n:
            bounds = _iqr_bounds_by_source_id(group, factor=policy.factor, min_n=policy.min_n)
            if bounds is not None:
                lower, upper = bounds
                source_flags = [
                    (row["value"] < lower or row["value"] > upper)
                    if row["source_id"] != "Reference"
                    else False
                    for _, row in group.iterrows()
                ]
            else:
                source_flags = [False] * len(group)
        elif len(group) >= policy.min_n:
            source_flags = _flag_outliers_iqr(
                group["value"].to_list(), factor=policy.factor, min_n=policy.min_n
            )
        else:
            source_flags = [False] * len(group)

        for (_, row), flagged in zip(group.iterrows(), source_flags):
            verdict_by_id[row["id"]] = bool(flagged) or not _is_finite(row["value"])

    annotated: list[AnnotatedScalar] = []
    total = 0
    for sv in scalar_values:
        is_outlier = verdict_by_id.get(sv.id, False)
        annotated.append(
            AnnotatedScalar(
                value=sv,
                is_outlier=is_outlier,
                verification_status="unverified" if is_outlier else "verified",
            )
        )
        total += int(is_outlier)
    return annotated, total
