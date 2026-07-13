"""Tests for ``ReferenceDatasetMixin`` and the reference dataset subtypes declared on top of it.

These are declaration-level tests that read the mapper and metadata. Persistence and the schema
itself are covered by ``test_migrations_reference_subtypes.py``.

The column-set assertions pin the mixin extraction as behaviour preserving: ``obs4mips_dataset``
and ``pmp_climatology_dataset`` must keep exactly the columns they had before the refactor, so
the extraction itself needs no migration.
"""

import pytest
from sqlalchemy.orm import configure_mappers

from climate_ref.models.base import Base
from climate_ref.models.dataset import (
    Dataset,
    ESMValToolReferenceDataset,
    Obs4MIPsDataset,
    Obs4REFDataset,
    PMPClimatologyDataset,
    ReferenceDatasetMixin,
)
from climate_ref_core.source_types import SourceDatasetType

# The exact column set obs4MIPs and PMP climatology carried before the mixin was extracted,
# so a column added to the mixin cannot silently widen these two existing tables without a migration.
_EXPECTED_REFERENCE_COLUMNS = {
    "id",
    "activity_id",
    "frequency",
    "grid",
    "grid_label",
    "institution_id",
    "long_name",
    "nominal_resolution",
    "realm",
    "product",
    "source_id",
    "source_type",
    "units",
    "variable_id",
    "variant_label",
    "version",
    "vertical_levels",
    "source_version_number",
    "instance_id",
}

_MIXIN_SUBTYPES = [Obs4MIPsDataset, PMPClimatologyDataset, Obs4REFDataset]


def _column_names(model: type[Dataset]) -> set[str]:
    return {column.name for column in model.__table__.columns}


class TestReferenceDatasetMixin:
    @pytest.mark.parametrize("model", _MIXIN_SUBTYPES)
    def test_subtype_has_the_expected_columns(self, model: type[Dataset]) -> None:
        """Each mixin-backed subtype carries exactly the shared reference column block."""
        assert _column_names(model) == _EXPECTED_REFERENCE_COLUMNS

    def test_existing_tables_are_unchanged_by_the_refactor(self) -> None:
        """obs4MIPs and PMP climatology keep the columns they had before the mixin existed.

        This is what makes the extraction migration-free. If it ever fails, the refactor has
        changed a live table and needs a migration.
        """
        assert _column_names(Obs4MIPsDataset) == _EXPECTED_REFERENCE_COLUMNS
        assert _column_names(PMPClimatologyDataset) == _EXPECTED_REFERENCE_COLUMNS

    def test_each_subtype_gets_independent_column_objects(self) -> None:
        """The mixin must not share one ``Column`` object across the subtype tables."""
        for column_name in ("source_id", "variable_id", "instance_id"):
            columns = [model.__table__.c[column_name] for model in _MIXIN_SUBTYPES]
            assert len({id(column) for column in columns}) == len(_MIXIN_SUBTYPES)
            assert {column.table.name for column in columns} == {
                "obs4mips_dataset",
                "pmp_climatology_dataset",
                "obs4ref_dataset",
            }

    def test_esmvaltool_reference_does_not_use_the_mixin(self) -> None:
        """ESMValTool reference data is not obs4MIPs compliant, so it has its own smaller set."""
        assert not issubclass(ESMValToolReferenceDataset, ReferenceDatasetMixin)
        assert _column_names(ESMValToolReferenceDataset) == {
            "id",
            "project",
            "source_id",
            "variable_id",
            "frequency",
            "version",
            "data_type",
            "tier",
            "long_name",
            "units",
            "instance_id",
        }

    def test_esmvaltool_reference_stores_frequency_not_a_mip_table(self) -> None:
        """``native6`` carries no MIP table, so frequency is the only shared temporal axis."""
        columns = _column_names(ESMValToolReferenceDataset)
        assert "frequency" in columns
        assert "table_id" not in columns

    def test_esmvaltool_frequency_is_non_nullable(self) -> None:
        """``frequency`` is part of the dataset identity, so it can never be NULL.

        A nullable identity column would let monthly and daily data for the same variable share
        an ``instance_id``, and would break the ``RANK`` partition of the latest-version window.
        """
        assert not ESMValToolReferenceDataset.__mapper__.columns["frequency"].nullable
        # Descriptive columns, by contrast, are allowed to be absent.
        assert ESMValToolReferenceDataset.__mapper__.columns["data_type"].nullable
        assert ESMValToolReferenceDataset.__mapper__.columns["tier"].nullable


class TestNewDatasetSubtypes:
    def test_mappers_configure(self) -> None:
        """Adding the subtypes must not break mapper configuration for the whole hierarchy."""
        configure_mappers()

    @pytest.mark.parametrize(
        "source_type, model, tablename",
        [
            (SourceDatasetType.obs4REF, Obs4REFDataset, "obs4ref_dataset"),
            (
                SourceDatasetType.ESMValToolReference,
                ESMValToolReferenceDataset,
                "esmvaltool_reference_dataset",
            ),
        ],
    )
    def test_polymorphic_identity_is_registered(
        self, source_type: SourceDatasetType, model: type[Dataset], tablename: str
    ) -> None:
        assert Dataset.__mapper__.polymorphic_map[source_type].class_ is model
        assert model.__tablename__ == tablename
        assert tablename in Base.metadata.tables

    def test_ordered_source_types_keep_their_relative_order(self) -> None:
        """``SourceDatasetType.ordered()`` sorts by value, and callers skip absent types.

        The new members slot into the alphabetical ordering without reordering the existing
        ones, so execution group keys built from ``ordered()`` stay stable.
        """
        ordered = [source_type.value for source_type in SourceDatasetType.ordered()]
        existing = ["cmip6", "cmip7", "obs4mips", "pmp-climatology"]
        assert [value for value in ordered if value in existing] == existing
