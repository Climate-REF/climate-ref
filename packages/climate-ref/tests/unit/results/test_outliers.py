"""Outlier detection: the SQL path in ``scalar_values`` and ``flag_outliers`` against a plain reference."""

import math
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.metric_value import ScalarMetricValue
from climate_ref.results import MetricValueFilter, OutlierPolicy, Reader
from climate_ref.results.outliers import flag_outliers

_MODEL_DIMS = {"source_id", "member_id", "variant_label", "grid_label"}


def _add_rows(session, diagnostic_id, rows, key="outliers"):
    """Store ``rows`` (dicts of ``value`` plus dimensions) under a new group, returning their ids."""
    group = ExecutionGroup(key=f"{key}-{diagnostic_id}", diagnostic_id=diagnostic_id, selectors={})
    session.add(group)
    session.flush()
    execution = Execution(
        execution_group_id=group.id, output_fragment=key, dataset_hash=f"hash-{key}", successful=True
    )
    session.add(execution)
    session.flush()
    values = []
    for row in rows:
        dims = {k: v for k, v in row.items() if k != "value" and v is not None}
        values.append(
            ScalarMetricValue.build(
                execution_id=execution.id, value=row["value"], attributes=None, dimensions=dims
            )
        )
    session.add_all(values)
    session.flush()
    return [v.id for v in values]


def _read_flags(database, policy, **kwargs):
    collection = Reader(database).values.scalar_values(outliers=policy, include_unverified=True, **kwargs)
    return {v.id: v.is_outlier for v in collection}


@pytest.fixture
def diagnostic_ids(db_seeded):
    with db_seeded.session.begin():
        ids = [d.id for d in db_seeded.session.query(Diagnostic).order_by(Diagnostic.id).limit(2)]
    assert len(ids) == 2
    return ids


def _models(values, **dims):
    return [
        {"value": value, "source_id": f"M{i}", "statistic": "rmse", "metric": "tas", **dims}
        for i, value in enumerate(values)
    ]


class TestCells:
    def test_regions_on_different_scales_are_tested_apart(self, db_seeded, diagnostic_ids):
        rows = _models([1.0, 1.1, 0.9, 1.05, 0.95, 1.02], region="global")
        rows += _models([100.0, 110.0, 90.0, 105.0, 95.0, 102.0], region="tropics")
        with db_seeded.session.begin():
            _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        collection = Reader(db_seeded).values.scalar_values(outliers=OutlierPolicy())
        assert collection.outlier_count == 0
        assert collection.total_count == 12

    def test_an_outlier_within_one_region_is_still_found(self, db_seeded, diagnostic_ids):
        rows = _models([1.0, 1.1, 0.9, 1.05, 0.95, 1000.0], region="global")
        rows += _models([100.0, 110.0, 90.0, 105.0, 95.0, 102.0], region="tropics")
        with db_seeded.session.begin():
            ids = _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        assert _read_flags(db_seeded, OutlierPolicy()) == {i: i == ids[5] for i in ids}

    def test_group_by_pools_the_other_labels(self, db_seeded, diagnostic_ids):
        rows = _models([1.0, 1.1, 0.9, 1.05, 0.95, 1.02], region="global")
        rows += _models([100.0, 110.0, 90.0, 105.0, 95.0, 102.0], region="tropics")
        with db_seeded.session.begin():
            _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        pooled = OutlierPolicy(factor=0.1, group_by=("statistic", "metric"))
        assert sum(_read_flags(db_seeded, pooled).values()) == 12

    def test_members_of_a_model_count_once(self, db_seeded, diagnostic_ids):
        # Without the per-model mean, five members of M5 would set the quartiles and hide it.
        rows = _models([1.0, 1.1, 0.9, 1.05, 0.95])
        rows += [{"value": 50.0, "source_id": "M5", "member_id": f"r{i}", "metric": "tas"} for i in range(5)]
        for row in rows:
            row.pop("statistic", None)
        with db_seeded.session.begin():
            ids = _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        assert _read_flags(db_seeded, OutlierPolicy()) == {i: i in ids[5:] for i in ids}


class TestReferences:
    def test_kind_reference_is_never_flagged_or_counted(self, db_seeded, diagnostic_ids):
        # Three models are below min_n, so the reference would enable detection if it were counted.
        rows = _models([1.0, 2.0, 1000.0])
        rows.append({"value": 1.5, "source_id": "HadISST", "kind": "reference", "metric": "tas"})
        rows.append({"value": 1e9, "source_id": "ERA5", "kind": "reference", "metric": "tas"})
        for row in rows:
            row.pop("statistic", None)
        with db_seeded.session.begin():
            ids = _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        assert _read_flags(db_seeded, OutlierPolicy(factor=0.5)) == dict.fromkeys(ids, False)
        frame = pd.DataFrame(rows)
        assert not flag_outliers(frame, OutlierPolicy(factor=0.5)).any()

    @pytest.mark.parametrize(
        "reference", [{"kind": "reference", "source_id": "obs"}, {"source_id": "Reference"}]
    )
    def test_non_finite_reference_is_not_flagged(self, db_seeded, diagnostic_ids, reference):
        rows = _models([1.0, 1.1, 0.9, 1.05, 0.95])
        rows.append({"value": math.nan, "statistic": "rmse", "metric": "tas", **reference})
        with db_seeded.session.begin():
            ids = _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        assert _read_flags(db_seeded, OutlierPolicy()) == dict.fromkeys(ids, False)
        assert not flag_outliers(pd.DataFrame(rows), OutlierPolicy()).any()


class TestNonFinite:
    def test_non_finite_with_a_null_label_is_flagged(self, db_seeded, diagnostic_ids):
        rows = _models([1.0, 1.1, 0.9, 1.05, 0.95], region="global")
        rows.append({"value": math.nan, "source_id": "M9", "statistic": "rmse", "metric": "tas"})
        rows.append({"value": None, "source_id": "M8", "statistic": "rmse", "metric": "tas"})
        with db_seeded.session.begin():
            ids = _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        for policy in (OutlierPolicy(), OutlierPolicy(group_by=("region",))):
            assert _read_flags(db_seeded, policy) == {i: i in ids[5:] for i in ids}
            assert flag_outliers(pd.DataFrame(rows), policy).tolist() == [False] * 5 + [True] * 2

    def test_infinite_is_flagged_and_does_not_set_limits(self, db_seeded, diagnostic_ids):
        rows = _models([1.0, 1.1, 0.9, 1.05, 0.95, math.inf, -math.inf])
        with db_seeded.session.begin():
            ids = _add_rows(db_seeded.session, diagnostic_ids[0], rows)

        assert _read_flags(db_seeded, OutlierPolicy()) == {i: i in ids[5:] for i in ids}


class TestPolicyValidation:
    def test_negative_factor_raises(self):
        with pytest.raises(ValueError, match="factor"):
            OutlierPolicy(factor=-1.0)
        assert OutlierPolicy(factor=0).factor == 0

    def test_string_group_by_is_one_label(self):
        assert OutlierPolicy(group_by="metric").group_by == ("metric",)

    @pytest.mark.parametrize("label", ["source_id", "kind", "member_id"])
    def test_model_labels_in_group_by_raise(self, label):
        with pytest.raises(ValueError, match=label):
            OutlierPolicy(group_by=("metric", label))


def _reference_flags(rows, policy):
    """Plain loop over cells and models, as the module docstring describes the test."""

    def is_reference(row):
        return row.get("kind") == "reference" or row.get("source_id") == "Reference"

    def is_finite(row):
        return row["value"] is not None and math.isfinite(row["value"])

    cells = defaultdict(list)
    for row in rows:
        if policy.group_by is None:
            labels = sorted((k, v) for k, v in row.items() if k not in {"id", "value", "kind", *_MODEL_DIMS})
        else:
            labels = [("diagnostic_id", row["diagnostic_id"]), *((g, row.get(g)) for g in policy.group_by)]
        cells[tuple(labels)].append(row)

    flags = {}
    for members in cells.values():
        by_model = defaultdict(list)
        for row in members:
            if is_finite(row) and not is_reference(row):
                model = row.get("source_id") or ("row", row["id"])
                by_model[model].append(row["value"])
        means = [statistics.fmean(v) for v in by_model.values()]
        bounds = None
        if len(means) >= policy.min_n:
            q1, _, q3 = statistics.quantiles(means, n=4, method="inclusive")
            if q3 > q1:
                bounds = (q1 - policy.factor * (q3 - q1), q3 + policy.factor * (q3 - q1))
        for row in members:
            if is_reference(row):
                flags[row["id"]] = False
            elif not is_finite(row):
                flags[row["id"]] = True
            else:
                flags[row["id"]] = bounds is not None and not bounds[0] <= row["value"] <= bounds[1]
    return flags


def _random_rows(rng, diagnostic_id):
    rows = []
    for metric in ("rmse", "bias"):
        for region in ("global", "tropics", None):
            for experiment_id in ("historical", "ssp585"):
                scale = 10 ** rng.uniform(-2, 3)
                cell = {"metric": metric, "region": region, "experiment_id": experiment_id}
                for model in rng.choice(
                    [f"M{i}" for i in range(9)], rng.integers(2, 10), replace=False
                ).tolist():
                    model_scale = scale * rng.uniform(0.7, 1.3)
                    for member in range(rng.integers(1, 4)):
                        value = model_scale * rng.uniform(0.9, 1.1)
                        roll = rng.random()
                        if roll < 0.05:
                            value *= 1000
                        elif roll < 0.08:
                            value = math.nan
                        elif roll < 0.09:
                            value = math.inf
                        rows.append({"value": value, "source_id": model, "member_id": f"r{member}", **cell})
                if rng.random() < 0.5:
                    rows.append({"value": rng.random() * 1e6, "source_id": "Reference", **cell})
                if rng.random() < 0.5:
                    obs = float(rng.choice([1.0, math.nan])) * scale
                    rows.append({"value": obs, "source_id": "obs", "kind": "reference", **cell})
                if rng.random() < 0.3:
                    rows.append({"value": scale * rng.uniform(0.5, 1.5), **cell})
    for row in rows:
        row["diagnostic_id"] = diagnostic_id
    return rows


@pytest.mark.parametrize(
    "policy",
    [
        OutlierPolicy(),
        OutlierPolicy(factor=1.5),
        OutlierPolicy(factor=0.0, min_n=3),
        OutlierPolicy(factor=1.0, group_by=("metric",)),
    ],
    ids=["default", "factor-1.5", "factor-0", "pooled"],
)
def test_sql_and_pandas_match_the_reference(db_seeded, diagnostic_ids, policy):
    rng = np.random.default_rng(20261023)
    rows = []
    with db_seeded.session.begin():
        for diagnostic_id in diagnostic_ids:
            diagnostic_rows = _random_rows(rng, diagnostic_id)
            stored = [{k: v for k, v in row.items() if k != "diagnostic_id"} for row in diagnostic_rows]
            for row, value_id in zip(diagnostic_rows, _add_rows(db_seeded.session, diagnostic_id, stored)):
                row["id"] = value_id
            rows += diagnostic_rows

    expected = _reference_flags(rows, policy)
    assert 0 < sum(expected.values()) < len(rows)
    assert _read_flags(db_seeded, policy) == expected

    frame = pd.DataFrame(rows).drop(columns="id")
    assert dict(zip((r["id"] for r in rows), flag_outliers(frame, policy).tolist())) == expected

    values = Reader(db_seeded).values
    kept = sorted(i for i, flagged in expected.items() if not flagged)
    first = values.scalar_values(outliers=policy, limit=50)
    second = values.scalar_values(outliers=policy, offset=50, limit=50, with_count=False, with_facets=False)
    assert first.total_count == len(kept)
    assert first.outlier_count == second.outlier_count == sum(expected.values())
    assert second.total_count is None
    assert [v.id for v in first] + [v.id for v in second] == kept[:100]
    assert all(v.is_outlier is False and v.verification_status == "verified" for v in first)


def test_empty_read_counts_nothing(db_seeded):
    collection = Reader(db_seeded).values.scalar_values(
        MetricValueFilter(execution_group_ids=[]), outliers=OutlierPolicy()
    )
    assert (collection.total_count, collection.outlier_count, collection.had_outliers) == (0, 0, False)
