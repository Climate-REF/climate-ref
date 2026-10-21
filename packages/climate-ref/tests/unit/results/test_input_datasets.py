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

        def add_group(key, diagnostic, executions):
            group = ExecutionGroup(key=key, diagnostic_id=diagnostic.id, selectors={})
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

        add_group("cmip6-a", diag_1, [[cmip6_a]])
        add_group("cmip7-a", diag_1, [[cmip7_a]])
        add_group("cmip6-b-old", diag_1, [[cmip6_b], []])
        add_group("other-b", diag_2, [[cmip6_b]])
        ids["diag_1"] = (diag_1.provider.slug, diag_1.slug)
        ids["diag_2"] = (diag_2.provider.slug, diag_2.slug)

    db_seeded.ids = ids
    return db_seeded


def _group_ids(db, input_datasets, **kwargs):
    filters = ExecutionGroupFilter(input_datasets=input_datasets, **kwargs)
    return {g.id for g in Reader(db).executions.groups(filters)}


def _value_sources(db, input_datasets):
    values = Reader(db).values.scalar_values(MetricValueFilter(input_datasets=input_datasets))
    return {v.dimensions["source_id"] for v in values}


class TestInputDatasetFilter:
    def test_unknown_facet_raises(self):
        with pytest.raises(ValueError, match="Unknown facet 'not_a_column'"):
            InputDatasetFilter(source_types=BOTH_ERAS, facets={"not_a_column": "x"})

    def test_empty_source_types_raises(self):
        with pytest.raises(ValueError, match="source_types"):
            InputDatasetFilter(source_types=())

    def test_bare_string_value_is_one_value(self):
        f = InputDatasetFilter(source_types=[CMIP6], facets={"source_id": "FACET-A"})
        assert f.source_types == (CMIP6,)
        assert f.facets == {"source_id": ("FACET-A",)}


class TestExecutionGroups:
    def test_match_across_eras(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=BOTH_ERAS, facets={"source_id": "FACET-A"})
        assert _group_ids(db_with_inputs, f) == {ids["cmip6-a"], ids["cmip7-a"]}

    def test_source_type_restricts_era(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=[CMIP7], facets={"source_id": "FACET-A"})
        assert _group_ids(db_with_inputs, f) == {ids["cmip7-a"]}

    def test_no_match(self, db_with_inputs):
        f = InputDatasetFilter(source_types=BOTH_ERAS, facets={"source_id": "NO-SUCH-MODEL"})
        assert _group_ids(db_with_inputs, f) == set()

    def test_multi_value_is_or(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=[CMIP6], facets={"source_id": ["FACET-A", "FACET-B"]})
        assert _group_ids(db_with_inputs, f) == {ids["cmip6-a"], ids["cmip6-b-old"], ids["other-b"]}

    def test_any_execution_matches_not_only_latest(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=[CMIP6], facets={"source_id": "FACET-B"})
        assert ids["cmip6-b-old"] in _group_ids(db_with_inputs, f)

    def test_facet_missing_from_one_era_excludes_it(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=BOTH_ERAS, facets={"mip_era": "CMIP7", "source_id": "FACET-A"})
        assert _group_ids(db_with_inputs, f) == {ids["cmip7-a"]}

    def test_facets_no_single_era_carries_match_nothing(self, db_with_inputs):
        f = InputDatasetFilter(source_types=BOTH_ERAS, facets={"table_id": "Amon", "region": "glb"})
        assert _group_ids(db_with_inputs, f) == set()
        assert _value_sources(db_with_inputs, f) == set()

    def test_combines_with_diagnostic_filter(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=[CMIP6], facets={"source_id": "FACET-B"})
        diagnostic = ids["diag_2"][1]
        assert _group_ids(db_with_inputs, f, diagnostic_contains=[diagnostic]) == {ids["other-b"]}


class TestMetricValues:
    def test_scoped_to_the_producing_execution(self, db_with_inputs):
        f = InputDatasetFilter(source_types=[CMIP6], facets={"source_id": "FACET-B"})
        # The newer cmip6-b-old execution had no inputs, so only its first value matches.
        values = Reader(db_with_inputs).values.scalar_values(MetricValueFilter(input_datasets=f))
        assert sorted((v.dimensions["source_id"], v.value) for v in values) == [
            ("cmip6-b-old", 0.0),
            ("other-b", 0.0),
        ]

    def test_match_and_no_match(self, db_with_inputs):
        both = InputDatasetFilter(source_types=BOTH_ERAS, facets={"source_id": "FACET-A"})
        none = InputDatasetFilter(source_types=BOTH_ERAS, facets={"source_id": "NO-SUCH-MODEL"})
        assert _value_sources(db_with_inputs, both) == {"cmip6-a", "cmip7-a"}
        assert _value_sources(db_with_inputs, none) == set()


class TestDiagnostics:
    def test_list_scopes_counts_and_drops_unmatched(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=BOTH_ERAS, facets={"source_id": "FACET-A"})
        items = Reader(db_with_inputs).diagnostics.list(DiagnosticFilter(input_datasets=f)).items
        assert [(d.provider_slug, d.slug) for d in items] == [ids["diag_1"]]
        (view,) = items
        assert (view.execution_group_count, view.total, view.successful) == (2, 2, 2)

    def test_combines_with_diagnostic_contains(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=[CMIP6], facets={"source_id": "FACET-B"})
        filters = DiagnosticFilter(input_datasets=f, diagnostic_contains=[ids["diag_2"][1]])
        items = Reader(db_with_inputs).diagnostics.list(filters).items
        assert [(d.provider_slug, d.slug, d.execution_group_count) for d in items] == [(*ids["diag_2"], 1)]

    def test_stats_scoped(self, db_with_inputs):
        ids = db_with_inputs.ids
        f = InputDatasetFilter(source_types=[CMIP6], facets={"source_id": "FACET-B"})
        stats = Reader(db_with_inputs).diagnostics.stats(input_datasets=f)
        assert {(s.provider, s.diagnostic): s.total for s in stats} == {ids["diag_1"]: 1, ids["diag_2"]: 1}

    def test_with_values_scoped(self, db_with_inputs):
        ids = db_with_inputs.ids
        reader = Reader(db_with_inputs).diagnostics
        cmip7 = InputDatasetFilter(source_types=[CMIP7], facets={"source_id": "FACET-A"})
        missing = InputDatasetFilter(source_types=[CMIP7], facets={"source_id": "NO-SUCH-MODEL"})
        assert reader.with_values(MetricValueType.SCALAR, DiagnosticFilter(input_datasets=cmip7)) == {
            ids["diag_1"]
        }
        assert reader.with_values(MetricValueType.SCALAR, DiagnosticFilter(input_datasets=missing)) == set()
