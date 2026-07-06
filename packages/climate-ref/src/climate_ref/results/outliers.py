"""
Outlier detection for scalar metric values.

This is a port of the logic previously living in the ref-app backend,
and has been hoisted here to ensure all consumers use the same logic.

Outlier detection is read logic, not presentation policy.
All consumers should see the same set of outliers.
The outlier detection is performed at run-time (instead of a pre-computed flag)
as the outlier configuration may depend on the use-case.
We may also adopt other outlier detection algorithms in future.


The algorithm is source-id-aware Inter-Quartile Range (IQR).
Within each ``group_by`` group,
IQR bounds are computed on the per-``source_id`` mean value
(so each model is weighted equally regardless of ensemble size),
then applied to individual values.
The values that are outside ``factor`` * IQR are flagged as outliers.
``"Reference"`` values are never flagged.
Non-finite values (NaN/inf) are always flagged.

"""

import math
import statistics
from collections.abc import Sequence

import attrs
import pandas as pd

from climate_ref.models.metric_value import ScalarMetricValue

_SUPPORTED_METHODS = ("off", "iqr")


@attrs.frozen(kw_only=True)
class OutlierPolicy:
    """
    Configuration for scalar outlier detection.

    The defaults reproduce the behaviour users currently see through the ref-app API
    (``factor=10.0``, source-id-aware IQR grouped by ``("statistic", "metric")``).
    """

    method: str = attrs.field(default="iqr", validator=attrs.validators.in_(_SUPPORTED_METHODS))
    """Detection method. ``"off"`` disables detection; ``"iqr"`` enables it."""

    factor: float = 10.0
    """Multiplier on the IQR to set the outlier bounds (``Q1 - factor*IQR``, ``Q3 + factor*IQR``)."""

    min_n: int = 4
    """
    Minimum sample size required to run detection.

    On the source-id-aware path this counts distinct non-reference ``source_id`` with a finite mean.
    On the fallback path it counts finite values.
    Non-finite values (NaN/inf) are excluded from the count
    and are flagged unconditionally regardless of whether detection runs.
    """

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
    """The underlying scalar metric value row."""

    is_outlier: bool
    """Whether this value was flagged as an outlier."""

    verification_status: str
    """``"verified"`` or ``"unverified"``, mirroring ``is_outlier``."""


def _is_finite(x: object) -> bool:
    return isinstance(x, int | float) and not math.isinf(x) and not math.isnan(x)


def _flag_outliers_iqr(values: Sequence[float], factor: float, min_n: int) -> list[bool]:
    """
    Flag outliers with a plain IQR test (fallback when no source_id is available).

    Detection runs only over finite values: a single NaN/inf would otherwise poison the quantiles
    and disable the test for the whole group.
    Non-finite values are never flagged here (the caller flags them separately).
    When the finite spread is zero the bounds collapse, so nothing is flagged.
    Gated on the count of finite values against ``min_n``.
    """
    finite = [v for v in values if _is_finite(v)]
    if len(finite) < min_n:
        return [False] * len(values)
    q1, _, q3 = statistics.quantiles(finite, n=4, method="inclusive")
    iqr = q3 - q1
    if iqr == 0:
        return [False] * len(values)
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return [_is_finite(v) and (v < lower or v > upper) for v in values]


def _iqr_bounds_by_source_id(df: pd.DataFrame, factor: float, min_n: int) -> tuple[float, float] | None:
    """
    IQR bounds computed on per-source_id means (equal model weighting).

    A source whose mean is non-finite (e.g. all values NaN) is dropped before the ``min_n`` count
    and the quantile computation, so it cannot poison the bounds for the others. Returns ``None``
    when fewer than ``min_n`` sources have a finite mean or when the spread is zero.
    """
    if "source_id" not in df.columns:
        return None
    non_reference = df[df["source_id"] != "Reference"]
    source_id_means = non_reference.groupby("source_id")["value"].mean()
    finite_means = source_id_means[source_id_means.map(_is_finite)]
    if len(finite_means) < min_n:
        return None
    q1, _, q3 = statistics.quantiles(finite_means.tolist(), n=4, method="inclusive")
    iqr = q3 - q1
    if iqr == 0:
        return None
    return q1 - factor * iqr, q3 + factor * iqr


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
        values = group["value"]
        finite = values.map(_is_finite)
        if "source_id" in group.columns:
            # Distinct finite source_ids gate detection (see _iqr_bounds_by_source_id).
            bounds = _iqr_bounds_by_source_id(group, factor=policy.factor, min_n=policy.min_n)
            if bounds is not None:
                lower, upper = bounds
                out_of_bounds = (values < lower) | (values > upper)
                flags = finite & out_of_bounds & (group["source_id"] != "Reference")
            else:
                flags = pd.Series(False, index=group.index)
        else:
            # Finite value count gates detection (see _flag_outliers_iqr).
            flag_list = _flag_outliers_iqr(values.to_list(), factor=policy.factor, min_n=policy.min_n)
            flags = pd.Series(flag_list, index=group.index)

        verdicts = flags | ~finite  # non-finite values are always flagged
        for row_id, verdict in zip(group["id"], verdicts):
            verdict_by_id[row_id] = bool(verdict)

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
