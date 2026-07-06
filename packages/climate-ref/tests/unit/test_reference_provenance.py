"""Tests for reference-dataset provenance linkage."""

import pytest

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models.dataset import ESMValToolReferenceDataset
from climate_ref.models.execution import Execution
from climate_ref.reference_provenance import link_reference_datasets, resolve_reference_dataset_ids
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.diagnostics import ReferenceDatasetSelector


@pytest.fixture
def db() -> Database:
    config = Config.default()
    database = Database("sqlite:///:memory:")
    database.migrate(config)
    yield database
    database.close()


def _add_reference_dataset(db, *, source_id, variable_id, table_id, version="v1", project="OBS") -> int:
    slug = f"esmvaltool-reference.{project}.{source_id}.{table_id}.{variable_id}.{version}"
    dataset = ESMValToolReferenceDataset(
        slug=slug,
        project=project,
        source_id=source_id,
        variable_id=variable_id,
        table_id=table_id,
        version=version,
        instance_id=slug,
        finalised=True,
    )
    db.session.add(dataset)
    db.session.flush()
    return dataset.id


class _StubDiagnostic:
    """Minimal stand-in exposing reference selectors."""

    def __init__(self, selectors):
        self._selectors = selectors

    def reference_dataset_selectors(self):
        return self._selectors


_OSI = ReferenceDatasetSelector(
    source_type=SourceDatasetType.ESMValToolReference,
    facets={"project": "OBS", "source_id": "OSI-450-nh", "table_id": "OImon"},
)


def test_resolve_matches_only_selected_facets(db):
    with db.session.begin():
        osi_id = _add_reference_dataset(db, source_id="OSI-450-nh", variable_id="sic", table_id="OImon")
        _add_reference_dataset(db, source_id="CERES-EBAF", variable_id="rlut", table_id="Amon")

    assert resolve_reference_dataset_ids(db, [_OSI]) == [osi_id]


def test_resolve_matches_all_variables_of_a_dataset(db):
    # A selector without a variable facet links every variable of the reference dataset.
    with db.session.begin():
        ids = {
            _add_reference_dataset(
                db, source_id="ERA5", project="native6", variable_id="hus", table_id="mon"
            ),
            _add_reference_dataset(db, source_id="ERA5", project="native6", variable_id="pr", table_id="mon"),
        }
    selector = ReferenceDatasetSelector(
        source_type=SourceDatasetType.ESMValToolReference,
        facets={"project": "native6", "source_id": "ERA5"},
    )
    assert set(resolve_reference_dataset_ids(db, [selector])) == ids


def test_resolve_warns_and_returns_empty_when_nothing_ingested(db, caplog):
    with caplog.at_level("WARNING"):
        result = resolve_reference_dataset_ids(db, [_OSI])
    assert result == []
    assert "No ingested esmvaltool-reference datasets match reference selector" in caplog.text


def test_link_records_reference_datasets_idempotently(db):
    with db.session.begin():
        osi_id = _add_reference_dataset(db, source_id="OSI-450-nh", variable_id="sic", table_id="OImon")
        execution = Execution(execution_group_id=1, output_fragment="frag", dataset_hash="hash")
        db.session.add(execution)
        db.session.flush()
        execution_id = execution.id

    diagnostic = _StubDiagnostic([_OSI])

    with db.session.begin():
        execution = db.session.get(Execution, execution_id)
        assert link_reference_datasets(db, execution, diagnostic) == 1
        assert {d.id for d in execution.datasets} == {osi_id}

    # Second call links nothing new.
    with db.session.begin():
        execution = db.session.get(Execution, execution_id)
        assert link_reference_datasets(db, execution, diagnostic) == 0
        assert {d.id for d in execution.datasets} == {osi_id}


def test_link_noop_for_diagnostic_without_reference_datasets(db):
    with db.session.begin():
        execution = Execution(execution_group_id=1, output_fragment="frag", dataset_hash="hash")
        db.session.add(execution)
        db.session.flush()
        execution_id = execution.id

    with db.session.begin():
        execution = db.session.get(Execution, execution_id)
        assert link_reference_datasets(db, execution, _StubDiagnostic([])) == 0
