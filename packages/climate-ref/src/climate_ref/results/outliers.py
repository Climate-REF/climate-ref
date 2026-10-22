"""
Outlier detection for scalar metric values.

Hide values from a model that are far off the others, so one broken run does not dominate a comparison.

Configure detection with an [OutlierPolicy][climate_ref.results.outliers.OutlierPolicy]
and pass it as ``outliers`` to [scalar_values][climate_ref.results.values.ValuesReader.scalar_values].
Every reader then sees the same outliers.
Detection runs when you read, so you can change the settings without rerunning anything.

The method is the inter-quartile range (IQR) test.
The IQR is the spread of the middle half of the values, from the 25th to the 75th percentile (Q1 to Q3).
A value is an outlier when it lies more than ``factor`` times the IQR below Q1 or above Q3.

The test is model-aware:

* Values are tested separately for each metric and statistic (the ``group_by`` fields).
* The limits come from each model's mean over its ensemble members (``source_id``),
  so a model with many members does not outweigh one with a single member.
* The limits are then applied to every individual value.
* ``"Reference"`` (observation) values are never flagged.
* Values that are NaN or infinite are always flagged.

Examples
--------
Which values are far off the other models? Flag them with the default policy and read them back:

```python
from climate_ref.results import OutlierPolicy, Reader

with Database.from_config(config, read_only=True) as db:
    collection = Reader(db).values.scalar_values(outliers=OutlierPolicy())
    print(collection.had_outliers, collection.outlier_count)
    df = collection.to_pandas()
    flagged = df[df["is_outlier"]]
```
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

    Settings for hiding values that are far off the others.

    Pass the policy as ``outliers`` to
    [ValuesReader.scalar_values][climate_ref.results.values.ValuesReader.scalar_values].
    The defaults run the model-aware IQR test described in
    [climate_ref.results.outliers][], with ``factor=10.0``, separately for each statistic and metric.
    That is loose enough to catch only gross errors, such as a unit mistake or a broken run.
    Lower ``factor`` to flag more values.

    Raises
    ------
    ValueError
        If ``method`` is not ``"off"`` or ``"iqr"``, or ``min_n`` is below 2.

    Examples
    --------
    The defaults:

    >>> from climate_ref.results import OutlierPolicy
    >>> policy = OutlierPolicy()
    >>> policy.method, policy.factor, policy.min_n, policy.group_by
    ('iqr', 10.0, 4, ('statistic', 'metric'))
    >>> policy.enabled
    True

    A stricter policy that also tests each region separately:

    >>> strict = OutlierPolicy(factor=3.0, group_by=("statistic", "metric", "region"))
    >>> strict.factor
    3.0

    Detection can be switched off explicitly:

    >>> OutlierPolicy(method="off").enabled
    False

    Read PMP values with that policy, keeping the flagged values so they can be inspected:

    ```python
    with Database.from_config(config, read_only=True) as db:
        collection = Reader(db).values.scalar_values(
            MetricValueFilter(provider_slug="pmp"),
            outliers=strict,
            include_unverified=True,
        )
    ```
    With ``include_unverified=False`` (the default), flagged values are left out of the result.
    """

    method: str = attrs.field(default="iqr", validator=attrs.validators.in_(_SUPPORTED_METHODS))
    """Detection method. ``"off"`` disables detection and ``"iqr"`` enables it."""

    factor: float = 10.0
    """Multiplier on the IQR that sets the outlier bounds, ``Q1 - factor*IQR`` and ``Q3 + factor*IQR``."""

    min_n: int = attrs.field(default=4, validator=attrs.validators.ge(2))
    """
    Minimum sample size required to run detection.

    With a ``source_id`` column this counts the models (excluding the reference) that have a finite mean.
    Without one it counts the finite values.
    A group below the minimum is never flagged for being far off.
    Non-finite values (NaN/inf) are not counted and are always flagged.
    """

    group_by: tuple[str, ...] = ("statistic", "metric")
    """Dimensions to test separately, such as ``"metric"`` or ``"region"``. Missing dimensions are ignored."""

    @property
    def enabled(self) -> bool:
        """Whether detection runs, which is ``True`` for ``method="iqr"``."""
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

    This is the function behind ``scalar_values(outliers=...)``.
    Use it directly on a DataFrame you already hold.

    ``frame`` holds a ``value`` column and one column per dimension (``source_id``, ``metric`` and so on).
    A dimension that is null in every row counts as absent.
    Rows with a null ``group_by`` key belong to no group and are never flagged.

    When any row has a ``source_id``, the limits come from each model's mean over its members,
    so every model counts equally, and ``"Reference"`` values are never flagged.
    A model whose mean is NaN or infinite is left out of the limits, so it cannot distort them for the others.
    Without a ``source_id``, the limits come from the finite values of each group.

    Parameters
    ----------
    frame
        Values to test, with a ``value`` column and optional dimension columns such as ``source_id``.
    policy
        Detection settings. The result is all ``False`` for an empty frame.

    Returns
    -------
    :
        A boolean array, ``True`` where the row is an outlier.

    Examples
    --------
    Five models agree and a sixth is far away.
    The reference (observation) value is large too, but is never flagged:

    >>> import pandas as pd
    >>> from climate_ref.results.outliers import OutlierPolicy, flag_outliers
    >>> frame = pd.DataFrame(
    ...     {
    ...         "source_id": ["a", "b", "c", "d", "e", "f", "Reference"],
    ...         "value": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, 5000.0],
    ...     }
    ... )
    >>> flag_outliers(frame, OutlierPolicy()).tolist()
    [False, False, False, False, False, True, False]
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
