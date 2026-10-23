"""Unit tests for scoping results by the datasets an execution ran against."""

import pytest

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.dataset import CMIP6Dataset, CMIP7Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.metric_value import MetricValueType, ScalarMetricValue
from climate_ref.results import (
    DiagnosticFilter,
    ExecutionGroupFilter,
    InputDatasetFilter,
    MetricValueFilter,
    Reader,
)
from climate_ref_core.source_types import SourceDatasetType

CMIP6 = SourceDatasetType.CMIP6
CMIP7 = SourceDatasetType.CMIP7
BOTH_ERAS = (CMIP6, CMIP7)


def _cmip6(source_id: str) -> CMIP6Dataset:
    slug = f"CMIP6.CMIP.TEST.{source_id}.historical.r1i1p1f1.Amon.tas.gn.v1"
    return CMIP6Dataset(
        slug=slug,
        activity_id="CMIP",
        experiment_id="historical",
        institution_id="TEST",
        source_id=source_id,
        member_id="r1i1p1f1",
        table_id="Amon",
        variable_id="tas",
        grid_label="gn",
        version="v1",
        instance_id=slug,
        variant_label="r1i1p1f1",
    )


def _cmip7(source_id: str) -> CMIP7Dataset:
    slug = f"CMIP7.CMIP.TEST.{source_id}.historical.r1i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.gn.v1"
    return CMIP7Dataset(
        slug=slug,
        activity_id="CMIP",
        institution_id="TEST",
        source_id=source_id,
        experiment_id="historical",
        variant_label="r1i1p1f1",
        variable_id="tas",
        grid_label="gn",
        frequency="mon",
        region="glb",
        branding_suffix="tavg-h2m-hxy-u",
        version="v1",
        mip_era="CMIP7",
        instance_id=slug,
    )


@pytest.fixture
def db_with_inputs(db_seeded):
    """
    Groups across two diagnostics with known input datasets.

    * ``cmip6-a``: one execution against CMIP6 ``FACET-A``, with a scalar value.
    * ``cmip7-a``: one execution against CMIP7 ``FACET-A``, with a scalar value.
    * ``cmip6-b-old``: an older execution against CMIP6 ``FACET-B`` and a newer one with no inputs.
    * ``other-b``: a second diagnostic, one successful execution against CMIP6 ``FACET-B``.
    """
    session = db_seeded.session
    ids = {}
    with session.begin():
        diag_1, diag_2 = session.query(Diagnostic).order_by(Diagnostic.id).limit(2).all()
        cmip6_a, cmip6_b, cmip7_a = _cmip6("FACET-A"), _cmip6("FACET-B"), _cmip7("FACET-A")
        session.add_all([cmip6_a, cmip6_b, cmip7_a])

        def add_group(key, diagnostic, executions, selectors):
            group = ExecutionGroup(key=key, diagnostic_id=diagnostic.id, selectors=selectors)
            session.add(group)
            session.flush()
            for n, datasets in enumerate(executions):
                execution = Execution(
                    execution_group_id=group.id,
                    output_fragment=f"{key}-{n}",
                    dataset_hash=f"{key}-{n}",
                    successful=True,
                )
                execution.datasets = datasets
                session.add(execution)
                session.flush()
                session.add(
                    ScalarMetricValue.build(
                        execution_id=execution.id,
                        value=float(n),
                        attributes=None,
                        dimensions={"source_id": key},
                    )
                )
            ids[key] = group.id

        add_group("cmip6-a", diag_1, [[cmip6_a]], {"cmip6": []})
        add_group("cmip7-a", diag_1, [[cmip7_a]], {"cmip7": []})
        add_group("cmip6-b-old", diag_1, [[cmip6_b], []], {"cmip6": []})
        add_group("other-b", diag_2, [[cmip6_b]], {"cmip6": []})
        ids["diag_1"] = (diag_1.provider.slug, diag_1.slug)
        ids["diag_2"] = (diag_2.provider.slug, diag_2.slug)

    db_seeded.ids = ids
    return db_seeded


@pytest.fixture
def db_with_obs_only(db_with_inputs):
    """``db_with_inputs`` plus ``obs-only``, a group on ``diag_2`` whose execution has no input datasets."""
    session = db_with_inputs.session
    with session.begin():
        diag_2 = session.query(Diagnostic).order_by(Diagnostic.id).offset(1).first()
        group = ExecutionGroup(key="obs-only", diagnostic_id=diag_2.id, selectors={"obs4mips": []})
        session.add(group)
        session.flush()
        execution = Execution(
            execution_group_id=group.id, output_fragment="obs-only", dataset_hash="obs-only", successful=True
        )
        session.add(execution)
        session.flush()
        session.add(
            ScalarMetricValue.build(
                execution_id=execution.id, value=0.0, attributes=None, dimensions={"source_id": "obs-only"}
            )
        )
        db_with_inputs.ids["obs-only"] = group.id

        mixed = ExecutionGroup(key="mixed", diagnostic_id=diag_2.id, selectors={"cmip6": [], "cmip7": []})
        session.add(mixed)
        session.flush()
        execution = Execution(
            execution_group_id=mixed.id, output_fragment="mixed", dataset_hash="mixed", successful=True
        )
        execution.datasets = [session.query(CMIP6Dataset).first(), session.query(CMIP7Dataset).first()]
        session.add(execution)
        session.flush()
        session.add(
            ScalarMetricValue.build(
                execution_id=execution.id, value=0.0, attributes=None, dimensions={"source_id": "mixed"}
            )
        )
        db_with_inputs.ids["mixed"] = mixed.id
    return db_with_inputs


def _group_ids(db, input_datasets, **kwargs):
    filters = ExecutionGroupFilter(input_datasets=input_datasets, **kwargs)
    return {g.id for g in Reader(db).executions.groups(filters)}


def _value_sources(db, input_datasets=None, **kwargs):
    values = Reader(db).values.scalar_values(MetricValueFilter(input_datasets=input_datasets, **kwargs))
    return {v.dimensions["source_id"] for v in values}


class TestInputDatasetFilter:
    def test_unknown_facet_raises(self):
        with pytest.raises(ValueError, match="Unknown facet 'not_a_column'"):
            InputDatasetFilter(facets={"not_a_column": "x"})

    def test_empty_filter_raises(self):
        with pytest.raises(ValueError, match="`facets` or `dataset_ids`"):
            InputDatasetFilter()

    def test_facets_no_single_type_carries_raise(self):
        with pytest.raises(ValueError, match="No dataset type has every facet"):
            InputDatasetFilter(facets={"table_id": "Amon", "region": "glb"})

    def test_dataset_ids_alone_is_enough(self):
        assert InputDatasetFilter(dataset_ids=[1, 2]).dataset_ids == (1, 2)

    def test_bare_string_value_is_one_value(self):
        assert InputDatasetFilter(facets={"source_id": "FACET-A"}).facets == {"source_id": ("FACET-A",)}


class TestExecutionGroups:
    def test_match_across_eras(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-A"})
        assert _group_ids(db_with_inputs, f) == {ids["cmip6-a"], ids["cmip7-a"]}

    def test_source_type_restricts_era(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-A"})
        assert _group_ids(db_with_inputs, f, source_types=[CMIP7]) == {ids["cmip7-a"]}

    def test_no_match(self, db_with_inputs):
        f = InputDatasetFilter(facets={"source_id": "NO-SUCH-MODEL"})
        assert _group_ids(db_with_inputs, f) == set()

    def test_multi_value_is_or(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": ["FACET-A", "FACET-B"]})
        assert _group_ids(db_with_inputs, f, source_types=[CMIP6]) == {ids["cmip6-a"], ids["other-b"]}

    def test_only_the_latest_execution_counts(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-B"})
        groups = Reader(db_with_inputs).executions.groups(ExecutionGroupFilter(input_datasets=f))
        assert {g.id for g in groups} == {ids["other-b"]}
        assert groups.total_count == 1

    def test_latest_follows_latest_successful(self, db_with_inputs):
        ids = db_with_inputs.ids
        session = db_with_inputs.session
        with session.begin():
            newest = (
                session.query(Execution)
                .filter_by(execution_group_id=ids["cmip6-b-old"])
                .order_by(Execution.id.desc())
                .first()
            )
            newest.successful = False
        f = InputDatasetFilter(facets={"source_id": "FACET-B"})
        assert ids["cmip6-b-old"] not in _group_ids(db_with_inputs, f)
        assert ids["cmip6-b-old"] in _group_ids(db_with_inputs, f, latest_successful=True)

    def test_facet_missing_from_one_era_excludes_it(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"mip_era": "CMIP7", "source_id": "FACET-A"})
        assert _group_ids(db_with_inputs, f) == {ids["cmip7-a"]}

    def test_source_types_without_the_facet_raise(self, db_with_inputs):
        f = InputDatasetFilter(facets={"mip_era": "CMIP7"})
        with pytest.raises(ValueError, match="No dataset type in cmip6"):
            _group_ids(db_with_inputs, f, source_types=[CMIP6])

    def test_combines_with_diagnostic_filter(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-B"})
        diagnostic = ids["diag_2"][1]
        assert _group_ids(db_with_inputs, f, diagnostic_contains=[diagnostic]) == {ids["other-b"]}


class TestDatasetIds:
    @staticmethod
    def _dataset_id(db, entity, source_id):
        return db.session.query(entity.id).filter_by(source_id=source_id).scalar()

    def test_matches_executions_that_used_the_dataset(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(dataset_ids=[self._dataset_id(db_with_inputs, CMIP6Dataset, "FACET-B")])
        assert _group_ids(db_with_inputs, f) == {ids["other-b"]}

    def test_any_dataset_type(self, db_with_inputs):
        ids = db_with_inputs.ids
        dataset_ids = [
            self._dataset_id(db_with_inputs, CMIP6Dataset, "FACET-A"),
            self._dataset_id(db_with_inputs, CMIP7Dataset, "FACET-A"),
        ]
        f = InputDatasetFilter(dataset_ids=dataset_ids)
        assert _group_ids(db_with_inputs, f) == {ids["cmip6-a"], ids["cmip7-a"]}

    def test_facets_and_ids_apply_to_the_same_dataset(self, db_with_inputs):
        cmip7_a = self._dataset_id(db_with_inputs, CMIP7Dataset, "FACET-A")
        f = InputDatasetFilter(facets={"source_id": "FACET-A"}, dataset_ids=[cmip7_a])
        assert _group_ids(db_with_inputs, f) == {db_with_inputs.ids["cmip7-a"]}

    def test_mixed_group_does_not_match_through_another_type(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        cmip7 = InputDatasetFilter(dataset_ids=[self._dataset_id(db_with_obs_only, CMIP7Dataset, "FACET-A")])
        assert ids["mixed"] in _group_ids(db_with_obs_only, cmip7, source_types=[CMIP7])
        assert _group_ids(db_with_obs_only, cmip7, source_types=[CMIP6]) == set()

    def test_empty_matches_nothing(self, db_with_inputs):
        f = InputDatasetFilter(dataset_ids=[])
        assert _group_ids(db_with_inputs, f) == set()
        assert _value_sources(db_with_inputs, f) == set()

    def test_empty_matches_nothing_with_no_model_input(self, db_with_obs_only):
        f = InputDatasetFilter(dataset_ids=[])
        assert _group_ids(db_with_obs_only, f, source_types=BOTH_ERAS, or_no_model_input=True) == set()

    def test_values(self, db_with_inputs):
        f = InputDatasetFilter(dataset_ids=[self._dataset_id(db_with_inputs, CMIP6Dataset, "FACET-A")])
        assert _value_sources(db_with_inputs, f) == {"cmip6-a"}


class TestSourceTypes:
    def test_groups_by_source_type(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        assert _group_ids(db_with_obs_only, None, source_types=[CMIP6]) == {
            ids["cmip6-a"],
            ids["cmip6-b-old"],
            ids["other-b"],
            ids["mixed"],
        }
        assert _group_ids(db_with_obs_only, None, source_types=[CMIP7]) == {ids["cmip7-a"], ids["mixed"]}

    def test_a_single_type_needs_no_list(self, db_with_obs_only):
        assert _group_ids(db_with_obs_only, None, source_types=CMIP7) == _group_ids(
            db_with_obs_only, None, source_types=[CMIP7]
        )

    def test_mixed_group_matches_both(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        for source_type in BOTH_ERAS:
            assert ids["mixed"] in _group_ids(db_with_obs_only, None, source_types=[source_type])

    def test_obs_only_group_is_excluded_by_default(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        assert ids["obs-only"] not in _group_ids(db_with_obs_only, None, source_types=BOTH_ERAS)

    def test_or_no_model_input_keeps_obs_only_groups(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        assert _group_ids(db_with_obs_only, None, source_types=[CMIP7], or_no_model_input=True) == {
            ids["cmip7-a"],
            ids["mixed"],
            ids["obs-only"],
        }

    def test_empty_matches_nothing_unless_or_no_model_input(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        assert _group_ids(db_with_obs_only, None, source_types=[]) == set()
        assert _group_ids(db_with_obs_only, None, source_types=[], or_no_model_input=True) == {
            ids["obs-only"]
        }

    def test_combines_with_input_datasets(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-A"})
        assert _group_ids(db_with_obs_only, f, source_types=[CMIP7], or_no_model_input=True) == {
            ids["cmip7-a"],
            ids["mixed"],
        }

    def test_never_executed_groups_are_kept_by_source_type(self, db_with_obs_only):
        session = db_with_obs_only.session
        with session.begin():
            diag_2 = session.query(Diagnostic).order_by(Diagnostic.id).offset(1).first()
            session.add(ExecutionGroup(key="never-run", diagnostic_id=diag_2.id, selectors={"cmip7": []}))
        reader = Reader(db_with_obs_only)
        groups = reader.executions.groups(ExecutionGroupFilter(source_types=[CMIP7]))
        assert {g.key for g in groups} == {"cmip7-a", "mixed", "never-run"}
        assert groups.total_count == len(groups.items)
        assert {g.key for g in reader.executions.groups(ExecutionGroupFilter(source_types=[CMIP6]))} == {
            "cmip6-a",
            "cmip6-b-old",
            "other-b",
            "mixed",
        }
        _, diag_slug = db_with_obs_only.ids["diag_2"]
        views = reader.diagnostics.list(DiagnosticFilter(source_types=[CMIP7]))
        assert {v.diagnostic_slug: v.groups.not_started for v in views}[diag_slug] == 1

    def test_values(self, db_with_obs_only):
        assert _value_sources(db_with_obs_only, source_types=[CMIP7]) == {"cmip7-a", "mixed"}
        assert _value_sources(db_with_obs_only, source_types=[CMIP7], or_no_model_input=True) == {
            "cmip7-a",
            "mixed",
            "obs-only",
        }
        assert _value_sources(db_with_obs_only, source_types=[]) == set()
        assert _value_sources(db_with_obs_only, source_types=[], or_no_model_input=True) == {"obs-only"}

    def test_latest_only_values(self, db_with_obs_only):
        values = _value_sources(db_with_obs_only, source_types=[CMIP6], latest_only=True)
        assert "cmip6-a" in values
        assert "obs-only" not in values

    @staticmethod
    def _counts(reader, filters):
        # Diagnostics with no matching group are listed with zeros.
        counts = {(d.provider_slug, d.diagnostic_slug): d.groups.total for d in reader.list(filters)}
        return {key: n for key, n in counts.items() if n}

    def test_diagnostics(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        reader = Reader(db_with_obs_only).diagnostics
        assert self._counts(reader, DiagnosticFilter(source_types=[CMIP7])) == {
            ids["diag_1"]: 1,
            ids["diag_2"]: 1,
        }
        obs = DiagnosticFilter(source_types=[], or_no_model_input=True)
        assert self._counts(reader, obs) == {ids["diag_2"]: 1}

    def test_with_values(self, db_with_obs_only):
        ids = db_with_obs_only.ids
        reader = Reader(db_with_obs_only).diagnostics
        scalar = MetricValueType.SCALAR
        assert reader.with_values(scalar, DiagnosticFilter(source_types=[CMIP7])) == {
            ids["diag_1"],
            ids["diag_2"],
        }
        assert reader.with_values(scalar, DiagnosticFilter(source_types=[], or_no_model_input=True)) == {
            ids["diag_2"]
        }
        assert reader.with_values(scalar, DiagnosticFilter(source_types=[])) == set()


class TestMetricValues:
    def test_scoped_to_the_producing_execution(self, db_with_inputs):
        f = InputDatasetFilter(facets={"source_id": "FACET-B"})
        # The newer cmip6-b-old execution had no inputs, so only its first value matches.
        values = Reader(db_with_inputs).values.scalar_values(MetricValueFilter(input_datasets=f))
        assert sorted((v.dimensions["source_id"], v.value) for v in values) == [
            ("cmip6-b-old", 0.0),
            ("other-b", 0.0),
        ]

    def test_match_and_no_match(self, db_with_inputs):
        both = InputDatasetFilter(facets={"source_id": "FACET-A"})
        none = InputDatasetFilter(facets={"source_id": "NO-SUCH-MODEL"})
        assert _value_sources(db_with_inputs, both) == {"cmip6-a", "cmip7-a"}
        assert _value_sources(db_with_inputs, none) == set()


class TestDiagnostics:
    def test_list_scopes_counts_and_drops_unmatched(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-A"})
        items = Reader(db_with_inputs).diagnostics.list(DiagnosticFilter(input_datasets=f)).items
        assert [(d.provider_slug, d.diagnostic_slug) for d in items] == [ids["diag_1"]]
        (view,) = items
        assert (view.groups.total, view.execution_count, view.groups.successful) == (
            2,
            2,
            2,
        )

    def test_combines_with_diagnostic_contains(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-B"})
        filters = DiagnosticFilter(input_datasets=f, diagnostic_contains=[ids["diag_2"][1]])
        items = Reader(db_with_inputs).diagnostics.list(filters).items
        assert [(d.provider_slug, d.diagnostic_slug, d.groups.total) for d in items] == [(*ids["diag_2"], 1)]

    def test_stats_scoped(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(facets={"source_id": "FACET-B"})
        items = Reader(db_with_inputs).diagnostics.list(DiagnosticFilter(input_datasets=f)).items
        assert {(d.provider_slug, d.diagnostic_slug): d.groups.total for d in items} == {ids["diag_2"]: 1}

    def test_with_values_scoped(self, db_with_inputs):
        ids = db_with_inputs.ids
        reader = Reader(db_with_inputs).diagnostics
        cmip7 = InputDatasetFilter(facets={"source_id": "FACET-A"})
        missing = InputDatasetFilter(facets={"source_id": "NO-SUCH-MODEL"})
        assert reader.with_values(MetricValueType.SCALAR, DiagnosticFilter(input_datasets=cmip7)) == {
            ids["diag_1"]
        }
        assert reader.with_values(MetricValueType.SCALAR, DiagnosticFilter(input_datasets=missing)) == set()
