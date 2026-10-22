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

import statistics

import attrs
import numpy as np
import pandas as pd

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

    min_n: int = attrs.field(default=4, validator=attrs.validators.ge(2))
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


def _iqr_bounds(samples: list[float], factor: float, min_n: int) -> tuple[float, float] | None:
    """
    ``Q1 - factor*IQR`` and ``Q3 + factor*IQR`` of finite ``samples``.

    Returns ``None`` when fewer than ``min_n`` samples are given or when the spread is zero,
    so collapsed bounds never flag anything.
    """
    if len(samples) < min_n:
        return None
    q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    iqr = q3 - q1
    if iqr == 0:
        return None
    return q1 - factor * iqr, q3 + factor * iqr


def _bounds_by_group(
    group_ids: np.ndarray, samples: np.ndarray, factor: float, min_n: int
) -> dict[int, tuple[float, float]]:
    """IQR bounds for each group id, from the finite samples carrying that id."""
    finite = np.isfinite(samples)
    group_ids, samples = group_ids[finite], samples[finite]
    if not len(samples):
        return {}
    order = np.argsort(group_ids, kind="stable")
    group_ids, samples = group_ids[order], samples[order]
    starts = np.flatnonzero(np.r_[True, group_ids[1:] != group_ids[:-1]])
    bounds = {}
    for gid, chunk in zip(group_ids[starts].tolist(), np.split(samples, starts[1:])):
        group_bounds = _iqr_bounds(chunk.tolist(), factor, min_n)
        if group_bounds is not None:
            bounds[gid] = group_bounds
    return bounds


def flag_outliers(frame: pd.DataFrame, policy: OutlierPolicy) -> np.ndarray:
    """
    Outlier verdict for each row of ``frame``, in row order.

    ``frame`` holds a ``value`` column and one column per CV dimension,
    where a dimension that is null in every row counts as absent.
    Rows with a null ``group_by`` key belong to no group and are never flagged.

    When any row has a ``source_id``, IQR bounds are computed on the per-``source_id`` means
    of each group (equal model weighting) and ``"Reference"`` values are never flagged.
    A source whose mean is non-finite is dropped before the ``min_n`` count,
    so it cannot poison the bounds for the others.
    Otherwise a plain IQR test runs over the finite values of each group.
    """
    if frame.empty:
        return np.zeros(0, dtype=bool)

    present = {c for c in frame.columns if frame[c].notna().any()}
    group_by = [g for g in policy.group_by if g in present]
    values = frame["value"].astype(float)
    finite = np.isfinite(values.to_numpy())

    if group_by:
        # ``ngroup`` labels rows with a null key as NaN.
        group_numbers = frame.groupby(group_by, sort=False).ngroup()
        in_group = group_numbers.notna().to_numpy()
        group_ids = group_numbers.fillna(-1).to_numpy(dtype=np.int64)
    else:
        in_group = np.ones(len(frame), dtype=bool)
        group_ids = np.zeros(len(frame), dtype=np.int64)

    if "source_id" in present:
        source_id = frame["source_id"]
        candidate = source_id != "Reference"
        candidate_rows = candidate.to_numpy() & in_group
        means = values[candidate_rows].groupby([group_ids[candidate_rows], source_id[candidate_rows]]).mean()
        mean_group_ids = means.index.get_level_values(0).to_numpy()
        bounds = _bounds_by_group(mean_group_ids, means.to_numpy(), policy.factor, policy.min_n)
        eligible = candidate.to_numpy()
    else:
        bounds = _bounds_by_group(
            group_ids[in_group], values.to_numpy()[in_group], policy.factor, policy.min_n
        )
        eligible = np.ones(len(frame), dtype=bool)

    n_groups = int(group_ids.max()) + 1
    lower_by_group = np.full(n_groups + 1, np.nan)
    upper_by_group = np.full(n_groups + 1, np.nan)
    for gid, (low, high) in bounds.items():
        lower_by_group[gid] = low
        upper_by_group[gid] = high
    # Rows outside any group carry id -1, which reads the trailing NaN slot.
    raw = values.to_numpy()
    with np.errstate(invalid="ignore"):
        out_of_bounds = (raw < lower_by_group[group_ids]) | (raw > upper_by_group[group_ids])

    # Non-finite values are always flagged.
    return in_group & ((finite & out_of_bounds & eligible) | ~finite)
