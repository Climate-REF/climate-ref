"""
Outlier detection for scalar metric values.

Hide values from a model that are far off the others, so one broken execution does not dominate a comparison.

Configure detection with an [OutlierPolicy][climate_ref.results.outliers.OutlierPolicy]
and pass it as ``outliers`` to [scalar_values][climate_ref.results.values.ValuesReader.scalar_values].
Every reader then sees the same outliers.
Detection runs when you read, so you can change the settings without rerunning anything.

The method is the inter-quartile range (IQR) test.
The IQR is the spread of the middle half of the values, from the 25th to the 75th percentile (Q1 to Q3).
A value is an outlier when it lies more than ``factor`` times the IQR below Q1 or above Q3.

The test is model-aware:

* Values are tested in cells.
  A cell is every value of one diagnostic that shares all labels except those naming the model:
  ``source_id``, ``member_id``, ``variant_label`` and ``grid_label``.
  So each region, season, experiment or reference dataset is tested on its own.
* The limits come from each model's mean over its finite values in the cell (one mean per ``source_id``),
  so a model with many members does not outweigh one with a single member.
* The limits are then applied to every individual value in the cell.
* Reference (observation) values are never flagged and never set the limits.
  A value is a reference when its ``kind`` is ``"reference"`` or its ``source_id`` is ``"Reference"``.
* Every other value that is missing, NaN or infinite is always flagged.

Detection runs in the database, so a read only loads the values it returns.

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
from typing import Any, cast

import attrs
import numpy as np
import pandas as pd
from sqlalchemy import ColumnElement, Select, and_, case, false, func, null, or_, select, true
from sqlalchemy.orm import aliased

from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import MetricValue, ScalarMetricValue
from climate_ref.query._converters import _as_str_tuple

_SUPPORTED_METHODS = ("off", "iqr")

_MODEL_DIMS = ("source_id", "member_id", "variant_label", "grid_label")
_NON_CELL_DIMS = ("kind", *_MODEL_DIMS)
"""Labels that name the model or ensemble member of a value rather than what it measures."""

_LEGACY_REFERENCE_SOURCE_ID = "Reference"
"""``source_id`` that marked reference values before ``kind`` was stored."""


def _check_group_by(instance: Any, attribute: Any, value: tuple[str, ...] | None) -> None:
    for dimension in value or ():
        if dimension in _NON_CELL_DIMS:
            raise ValueError(f"group_by cannot include {dimension!r}, which names a model or its role")


@attrs.frozen(kw_only=True)
class OutlierPolicy:
    """
    Configuration for scalar outlier detection.

    Settings for hiding values that are far off the others.

    Pass the policy as ``outliers`` to
    [ValuesReader.scalar_values][climate_ref.results.values.ValuesReader.scalar_values].
    The defaults run the model-aware IQR test described in
    [climate_ref.results.outliers][], with ``factor=10.0``, separately in each cell.
    That is loose enough to catch only gross errors, such as a unit mistake or a broken execution.
    Lower ``factor`` to flag more values.

    Raises
    ------
    ValueError
        If ``method`` is not ``"off"`` or ``"iqr"``, ``factor`` is negative, ``min_n`` is below 2,
        or ``group_by`` names ``kind`` or a model label such as ``source_id``.

    Examples
    --------
    The defaults:

    >>> from climate_ref.results import OutlierPolicy
    >>> policy = OutlierPolicy()
    >>> policy.method, policy.factor, policy.min_n, policy.group_by
    ('iqr', 10.0, 4, None)
    >>> policy.enabled
    True

    A stricter policy:

    >>> strict = OutlierPolicy(factor=3.0)
    >>> strict.factor
    3.0

    One that pools every region and season of a metric and statistic into one cell:

    >>> OutlierPolicy(group_by=["statistic", "metric"]).group_by
    ('statistic', 'metric')

    Detection can be switched off explicitly:

    >>> OutlierPolicy(method="off").enabled
    False

    Read PMP values with the stricter policy, keeping the flagged values so they can be inspected:

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

    factor: float = attrs.field(default=10.0, validator=attrs.validators.ge(0))
    """Multiplier on the IQR that sets the outlier bounds, ``Q1 - factor*IQR`` and ``Q3 + factor*IQR``."""

    min_n: int = attrs.field(default=4, validator=attrs.validators.ge(2))
    """
    Minimum number of models in a cell required to run detection.

    This counts the models (by ``source_id``, excluding references) that have a finite mean.
    A value without a ``source_id`` counts as a model of its own.
    A cell below the minimum is never flagged for being far off,
    but its non-finite values are still flagged.
    """

    group_by: tuple[str, ...] | None = attrs.field(
        default=None, converter=_as_str_tuple, validator=_check_group_by
    )
    """
    Labels that define a cell, such as ``("statistic", "metric")``, or ``None`` for every label.

    ``None`` (the default) tests each combination of labels apart from the model ones on its own.
    Naming labels pools the values of every other label into one cell, for example every region.
    Only do that when the pooled values share a scale, because the limits are applied across them all.
    Each diagnostic is always tested on its own.
    A label that is not a CV dimension raises ``ValueError`` when the values are read.
    """

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


def flag_outliers(frame: pd.DataFrame, policy: OutlierPolicy) -> np.ndarray:
    """
    Outlier verdict for each row of ``frame``, in row order.

    This applies the same test as ``scalar_values(outliers=...)`` to a DataFrame you already hold.

    ``frame`` holds a ``value`` column and one column per label (``source_id``, ``metric`` and so on).
    An optional ``kind`` column marks reference values with ``"reference"``.
    Every other column defines the cells, so leave out columns such as ``id``.
    Add a column such as ``diagnostic_id`` when the frame mixes diagnostics.
    With ``policy.group_by`` set, only those columns and ``diagnostic_id`` define the cells.
    A null label is a value of its own, so rows with no ``region`` form a cell apart from the others.

    Parameters
    ----------
    frame
        Values to test, with a ``value`` column and label columns such as ``source_id``.
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
    ...         "source_id": ["a", "b", "c", "d", "e", "f", "obs"],
    ...         "kind": ["model"] * 6 + ["reference"],
    ...         "value": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, 5000.0],
    ...     }
    ... )
    >>> flag_outliers(frame, OutlierPolicy()).tolist()
    [False, False, False, False, False, True, False]
    """
    if frame.empty:
        return np.zeros(0, dtype=bool)

    values = frame["value"].astype(float).to_numpy()
    finite = np.isfinite(values)
    reference = np.zeros(len(frame), dtype=bool)
    if "kind" in frame:
        reference |= (frame["kind"] == "reference").to_numpy()
    if "source_id" in frame:
        reference |= (frame["source_id"] == _LEGACY_REFERENCE_SOURCE_ID).to_numpy()

    if policy.group_by is None:
        cell_dims = [c for c in frame.columns if c != "value" and c not in _NON_CELL_DIMS]
    else:
        cell_dims = [c for c in frame.columns if c in policy.group_by or c == "diagnostic_id"]
    cells = (
        frame.groupby(cell_dims, dropna=False, sort=False).ngroup().to_numpy()
        if cell_dims
        else np.zeros(len(frame), dtype=np.int64)
    )

    # A value without a source_id is a model of its own.
    models = pd.factorize(frame["source_id"])[0] if "source_id" in frame else np.full(len(frame), -1)
    solo = models == -1
    models[solo] = models.max() + 1 + np.arange(solo.sum())
    samples = pd.Series(np.where(finite & ~reference, values, np.nan))
    means = samples.groupby([cells, models], sort=False).mean().dropna()

    lower_by_cell = np.full(cells.max() + 1, np.nan)
    upper_by_cell = np.full(cells.max() + 1, np.nan)
    for cell, cell_means in means.groupby(level=0):
        bounds = _iqr_bounds(cell_means.tolist(), policy.factor, policy.min_n)
        if bounds is not None:
            lower_by_cell[cast(int, cell)], upper_by_cell[cast(int, cell)] = bounds
    with np.errstate(invalid="ignore"):
        out_of_bounds = (values < lower_by_cell[cells]) | (values > upper_by_cell[cells])
    return ~reference & (~finite | out_of_bounds)


def _is_reference(columns: Any) -> ColumnElement[bool]:
    clauses = []
    if "kind" in columns:
        clauses.append(columns.kind == "reference")
    if "source_id" in columns:
        clauses.append(columns.source_id == _LEGACY_REFERENCE_SOURCE_ID)
    return or_(false(), *clauses)


def _is_finite(value: Any) -> ColumnElement[bool]:
    # Postgres sorts NaN above infinity, and SQLite stores NaN as null, so this is false for both.
    return and_(value > float("-inf"), value < float("inf"))


def select_outlier_flags(value_ids: Select[tuple[int]], policy: OutlierPolicy) -> Select[tuple[int, bool]]:
    """
    Build a ``Select`` of ``(id, is_outlier)`` for every scalar value that ``value_ids`` matches.

    The test is the one [flag_outliers][climate_ref.results.outliers.flag_outliers] applies,
    run in the database with the cells split by diagnostic.
    The quartiles interpolate linearly between the sorted model means,
    as ``statistics.quantiles(method="inclusive")`` and the pandas default do.

    Parameters
    ----------
    value_ids
        Ids of the scalar values to test, such as from
        [select_value_ids][climate_ref.query.values.select_value_ids].
    policy
        Detection settings.

    Raises
    ------
    ValueError
        If ``policy.group_by`` names a label that is not a CV dimension.
    """
    dims = ScalarMetricValue._cv_dimensions
    for g in policy.group_by or ():
        if g not in dims:
            raise ValueError(f"Unknown outlier group_by dimension {g!r}")
    if policy.group_by is None:
        cell_dims = [d for d in dims if d not in _NON_CELL_DIMS]
    else:
        cell_dims = list(policy.group_by)
    table = MetricValue.__table__.c
    execution = aliased(Execution)
    group = aliased(ExecutionGroup)
    diagnostic_id = (
        select(group.diagnostic_id)
        .join(execution, execution.execution_group_id == group.id)
        .where(execution.id == table.execution_id)
        .scalar_subquery()
    )
    labels = [d for d in dims if d in {"kind", "source_id", *cell_dims}]
    scoped = (
        value_ids.with_only_columns(
            table.id, table.value, *(table[d] for d in labels), diagnostic_id.label("diagnostic_id")
        )
        .order_by(None)
        .cte("outlier_scope")
    )
    c = scoped.c
    model = c.source_id if "source_id" in c else None
    cells = select(
        c.id,
        c.value,
        *(c[d] for d in labels),
        func.dense_rank().over(order_by=[c.diagnostic_id, *(c[d] for d in cell_dims)]).label("cell"),
        case((_is_reference(c), null()), (_is_finite(c.value), c.value)).label("sample"),
        # A value without a source_id is a model of its own.
        (case((model.is_(None), c.id)) if model is not None else c.id).label("solo"),
    ).cte("outlier_cells")

    model_keys = [cells.c.cell, cells.c.solo, *([cells.c.source_id] if model is not None else [])]
    means = (
        select(cells.c.cell, func.avg(cells.c.sample).label("mean"))
        .group_by(*model_keys)
        .having(func.avg(cells.c.sample).is_not(None))
        .subquery("outlier_means")
    )
    ranked = select(
        means.c.cell,
        means.c.mean,
        (func.row_number().over(partition_by=means.c.cell, order_by=means.c.mean) - 1).label("position"),
        func.count().over(partition_by=means.c.cell).label("n"),
    ).subquery("outlier_ranked")

    def quartile(i: int) -> Any:
        # statistics.quantiles(n=4, method="inclusive"): (x[j] * (4 - delta) + x[j + 1] * delta) / 4
        spread = i * (ranked.c.n - 1)
        j, delta = spread // 4, spread % 4
        weight = case((ranked.c.position == j, 4 - delta), (ranked.c.position == j + 1, delta), else_=0)
        return func.sum(ranked.c.mean * weight) / 4.0

    quartiles = (
        select(ranked.c.cell, quartile(1).label("q1"), quartile(3).label("q3"))
        .group_by(ranked.c.cell)
        .having(func.max(ranked.c.n) >= policy.min_n)
        .subquery("outlier_quartiles")
    )
    iqr = quartiles.c.q3 - quartiles.c.q1
    bounds = (
        select(
            quartiles.c.cell,
            (quartiles.c.q1 - policy.factor * iqr).label("lower"),
            (quartiles.c.q3 + policy.factor * iqr).label("upper"),
        )
        .where(iqr > 0)
        .subquery("outlier_bounds")
    )

    finite = _is_finite(cells.c.value)
    is_outlier = case(
        (_is_reference(cells.c), false()),
        (and_(finite, or_(cells.c.value < bounds.c.lower, cells.c.value > bounds.c.upper)), true()),
        (finite, false()),
        else_=true(),
    )
    return select(cells.c.id, is_outlier.label("is_outlier")).outerjoin(bounds, bounds.c.cell == cells.c.cell)
