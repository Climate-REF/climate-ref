import datetime
from typing import Any, ClassVar

from sqlalchemy import BigInteger, ColumnElement, ForeignKey, event, func
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from climate_ref.models.base import Base
from climate_ref_core.datasets import version_sort_key
from climate_ref_core.source_types import SourceDatasetType


class Dataset(Base):
    """
    Represents a dataset

    A dataset is a collection of data files, that is used as an input to the benchmarking process.
    Adding/removing or updating a dataset will trigger a new diagnostic calculation.

    A polymorphic association is used to capture the different types of datasets as each
    dataset type may have different metadata fields.
    This enables the use of a single table to store all datasets,
    but still allows for querying specific metadata fields for each dataset type.
    """

    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(unique=True)
    """
    Globally unique identifier for the dataset.

    In the case of CMIP6 datasets, this is the instance_id.
    """
    dataset_type: Mapped[SourceDatasetType] = mapped_column(nullable=False, index=True)
    """
    Type of dataset
    """
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    """
    When the dataset was added to the database
    """
    updated_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    """
    When the dataset was updated.

    Updating a dataset will trigger a new diagnostic calculation.
    """

    # Universal finalisation flag for all dataset types
    # Only CMIP6 currently uses unfinalised datasets in practice; other types should be finalised on creation.
    finalised: Mapped[bool] = mapped_column(default=True, nullable=False)
    """
    Whether the complete set of metadata for the dataset has been finalised.

    For CMIP6, ingestion may initially create unfinalised datasets (False) until all metadata is extracted.
    For other dataset types (e.g., obs4MIPs, PMP climatology), this should be True upon creation.
    """

    version_key: Mapped[int] = mapped_column(BigInteger, default=-1, server_default="-1", nullable=False)
    """
    Numeric ordering key for the subclass's ``version`` column,
    computed by :func:`climate_ref_core.datasets.version_sort_key`.

    Kept in sync by the ``_sync_version_key`` mapper event.
    Rows with no ``version`` attribute (base-table-only inserts) keep the ``-1`` backstop.

    Lives on the base table (not a subclass) so the SQL latest-version window function
    (``select_datasets(..., latest_group_by=...)``) can read it off any polymorphic row while
    partitioning on the subclass's ``dataset_id_metadata`` columns.

    Core writes to ``version``
    (``session.execute(update(...))``, ``connection.execute(...)``, ``bulk_update_mappings``)
    bypass the event and leave ``version_key`` stale, silently corrupting latest-version dedup.
    ALWAYS mutate ``version`` through an ORM instance.
    """

    retracted_at: Mapped[datetime.datetime | None] = mapped_column(nullable=True, default=None)
    """
    When the dataset was retracted, or ``None`` while it is active.

    Retraction exists because a dataset can never be hard-deleted once it has run
    (``execution_datasets.dataset_id`` has no ``ondelete="CASCADE"``,
    so removing the row would either violate the foreign key or destroy execution provenance).
    A retracted dataset's row and files are left intact, but excluded from future solves.

    ``select_datasets`` (``climate_ref.models.dataset_query``) excludes retracted rows by default
    and only includes them when a caller passes ``DatasetFilter(include_retracted=True)``,
    so provenance/history reads
    (e.g. reconstructing what an existing execution used, or ``ref datasets list``) can still see them.
    """

    def __repr__(self) -> str:
        return f"<Dataset slug={self.slug} dataset_type={self.dataset_type} >"

    __mapper_args__: ClassVar[Any] = {"polymorphic_on": dataset_type}  # type: ignore


@event.listens_for(Dataset, "before_insert", propagate=True)
@event.listens_for(Dataset, "before_update", propagate=True)
def _sync_version_key(mapper: Any, connection: Any, target: Dataset) -> None:
    """Keep ``version_key`` numerically in sync with the subclass's ``version`` column.

    ``propagate=True`` fires this for every ``Dataset`` subclass,
    reading the final attribute value on the instance so it is write path independent.
    """
    target.version_key = version_sort_key(getattr(target, "version", None))


class DatasetFile(Base):
    """
    Capture the metadata for a file in a dataset

    A dataset may have multiple files, but is represented as a single dataset in the database.
    A lot of the metadata will be duplicated for each file in the dataset,
    but this will be more efficient for querying, filtering and building a data catalog.
    """

    __tablename__ = "dataset_file"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True
    )
    """
    Foreign key to the dataset table
    """

    start_time: Mapped[str] = mapped_column(nullable=True)
    """
    Start time of a given file (ISO string, supports cftime calendars)
    """

    end_time: Mapped[str] = mapped_column(nullable=True)
    """
    End time of a given file (ISO string, supports cftime calendars)
    """

    path: Mapped[str] = mapped_column()
    """
    Prefix that describes where the dataset is stored relative to the data directory
    """

    @validates("start_time", "end_time")
    def _coerce_time_to_str(self, _key: str, value: object) -> str | None:
        """Cast cftime/datetime objects to ISO strings for DB storage."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    tracking_id: Mapped[str] = mapped_column(nullable=True)
    """
    Unique file identifier.

    For CMIP7, this is the handle identifier (e.g., "hdl:21.14107/uuid").
    """

    dataset = relationship("Dataset", backref="files")


class CMIP6Dataset(Dataset):
    """
    Represents a CMIP6 dataset

    Fields that are not in the DRS are marked optional.
    """

    __tablename__ = "cmip6_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    activity_id: Mapped[str] = mapped_column()
    branch_method: Mapped[str] = mapped_column(nullable=True)
    branch_time_in_child: Mapped[float] = mapped_column(nullable=True)
    branch_time_in_parent: Mapped[float] = mapped_column(nullable=True)
    experiment: Mapped[str] = mapped_column(nullable=True)
    experiment_id: Mapped[str] = mapped_column(index=True)
    frequency: Mapped[str] = mapped_column(nullable=True)
    grid: Mapped[str] = mapped_column(nullable=True)
    grid_label: Mapped[str] = mapped_column()
    institution_id: Mapped[str] = mapped_column()
    long_name: Mapped[str] = mapped_column(nullable=True)
    member_id: Mapped[str] = mapped_column(index=True)
    nominal_resolution: Mapped[str] = mapped_column(nullable=True)
    parent_activity_id: Mapped[str] = mapped_column(nullable=True)
    parent_experiment_id: Mapped[str] = mapped_column(nullable=True)
    parent_source_id: Mapped[str] = mapped_column(nullable=True)
    parent_time_units: Mapped[str] = mapped_column(nullable=True)
    parent_variant_label: Mapped[str] = mapped_column(nullable=True)
    realm: Mapped[str] = mapped_column(nullable=True)
    product: Mapped[str] = mapped_column(nullable=True)
    source_id: Mapped[str] = mapped_column(index=True)
    standard_name: Mapped[str] = mapped_column(nullable=True)
    source_type: Mapped[str] = mapped_column(nullable=True)
    sub_experiment: Mapped[str] = mapped_column(nullable=True)
    sub_experiment_id: Mapped[str] = mapped_column(nullable=True)
    table_id: Mapped[str] = mapped_column()
    units: Mapped[str] = mapped_column(nullable=True)
    variable_id: Mapped[str] = mapped_column()
    variant_label: Mapped[str] = mapped_column()
    vertical_levels: Mapped[int] = mapped_column(nullable=True)
    version: Mapped[str] = mapped_column()
    """
    Dataset version string (e.g. ``"v2"``).

    Only write this through an ORM instance.
    The base-table ``version_key`` ordering key is synced by the ``_sync_version_key`` mapper event.
    """

    instance_id: Mapped[str] = mapped_column(index=True)
    """
    Unique identifier for the dataset (including the version).
    """

    time_units: Mapped[str] = mapped_column(nullable=True)
    """Time encoding units (e.g. 'days since 1850-01-01')"""

    calendar: Mapped[str] = mapped_column(nullable=True)
    """CF calendar type (e.g. 'standard', '360_day', 'noleap')"""

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.CMIP6}  # type: ignore


class ReferenceDatasetMixin:
    """
    Shared column block for reference/observational dataset types.

    These datasets look like obs4MIPs datasets (they share its metadata conventions),
    but are stored in separate polymorphic subtype tables.

    """

    activity_id: Mapped[str] = mapped_column()
    frequency: Mapped[str] = mapped_column()
    grid: Mapped[str] = mapped_column()
    grid_label: Mapped[str] = mapped_column()
    institution_id: Mapped[str] = mapped_column()
    long_name: Mapped[str | None] = mapped_column(nullable=True)
    nominal_resolution: Mapped[str] = mapped_column()
    realm: Mapped[str] = mapped_column()
    product: Mapped[str] = mapped_column()
    source_id: Mapped[str] = mapped_column()
    source_type: Mapped[str] = mapped_column()
    units: Mapped[str] = mapped_column()
    variable_id: Mapped[str] = mapped_column()
    variant_label: Mapped[str] = mapped_column()
    version: Mapped[str] = mapped_column()
    """
    Dataset version string.

    Only write this through an ORM instance: the base-table ``version_key`` ordering key is
    synced by the ``_sync_version_key`` mapper event, which Core-level updates bypass.
    """
    vertical_levels: Mapped[int] = mapped_column()
    source_version_number: Mapped[str] = mapped_column()

    instance_id: Mapped[str] = mapped_column()
    """
    Unique identifier for the dataset.
    """


class Obs4MIPsDataset(ReferenceDatasetMixin, Dataset):
    """
    Represents a obs4mips dataset
    """

    __tablename__ = "obs4mips_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.obs4MIPs}  # type: ignore


class PMPClimatologyDataset(ReferenceDatasetMixin, Dataset):
    """
    Represents a climatology dataset from PMP

    These data are similar to obs4MIPs datasets, but are post-processed
    """

    __tablename__ = "pmp_climatology_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.PMPClimatology}  # type: ignore


class Obs4REFDataset(ReferenceDatasetMixin, Dataset):
    """
    Represents an obs4REF dataset

    obs4REF is REF-curated observational data that shares the obs4MIPs metadata
    conventions but is not published to the obs4MIPs ESGF archive.

    The format of this table may change as there isn't an official standard for obs4REF datasets yet.
    The current columns are based on the obs4MIPs metadata conventions.
    """

    __tablename__ = "obs4ref_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.obs4REF}  # type: ignore


class ESMValToolReferenceDataset(Dataset):
    """
    Represents a reference (observational/reanalysis) dataset used by ESMValTool.

    Unlike obs4MIPs and PMP climatology datasets,
    ESMValTool reference data is not strictly CMOR/obs4MIPs compliant.
    The data are stored in ESMValTool's own layout
    (``OBS``/``OBS6``, ``native6`` and non-compliant ``obs4MIPs``),
    so the available metadata is limited to what the path and filename encode.
    This model therefore has a smaller column set than :class:`ReferenceDatasetMixin`.

    It is a deliberately separate dataset type:
    the data describes different sources
    and may carry different versions to the obs4MIPs data of the same name,
    but will have a different ``instance_id``.
    """

    __tablename__ = "esmvaltool_reference_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    project: Mapped[str] = mapped_column(index=True)
    """ESMValCore project the data is loaded as: ``OBS``, ``OBS6``, ``native6`` or ``obs4MIPs``."""

    source_id: Mapped[str] = mapped_column(index=True)
    """Reference dataset name, e.g. ``CERES-EBAF``, ``ERA5``, ``OSI-450-nh``."""

    variable_id: Mapped[str] = mapped_column(index=True)
    """ESMValTool short name of the variable, e.g. ``tas``, ``sic``, ``rlut``."""

    frequency: Mapped[str] = mapped_column(index=True)
    """
    Temporal resolution, as a CMIP6 ``frequency`` CV value, e.g. ``mon``, ``day``, ``fx``.
    """

    version: Mapped[str] = mapped_column()
    """Dataset version as encoded in the ESMValTool layout, e.g. ``v3``, ``Ed4.2``."""

    data_type: Mapped[str] = mapped_column(nullable=True)
    """ESMValTool observation type where available: ``reanaly``, ``sat``, ``ground``."""

    tier: Mapped[int] = mapped_column(nullable=True)
    """ESMValTool data tier (accessibility), where the layout encodes it."""

    long_name: Mapped[str] = mapped_column(nullable=True)
    units: Mapped[str] = mapped_column(nullable=True)

    instance_id: Mapped[str] = mapped_column(index=True)
    """
    Unique identifier for the dataset.
    """
    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.ESMValToolReference}  # type: ignore


class CMIP7Dataset(Dataset):
    """
    Represents a CMIP7 dataset

    Based on CMIP7 Global Attributes v1.0 (DOI: 10.5281/zenodo.17250297).
    Includes core DRS attributes, additional mandatory attributes, and parent info.
    """

    __tablename__ = "cmip7_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    # Core DRS Attributes (required for directory/filename/instance_id)
    activity_id: Mapped[str] = mapped_column()
    """CV - e.g., "CMIP", "ScenarioMIP" """

    institution_id: Mapped[str] = mapped_column()
    """CV - registered by modeling group"""

    source_id: Mapped[str] = mapped_column(index=True)
    """CV - model identifier"""

    experiment_id: Mapped[str] = mapped_column(index=True)
    """CV - experiment name"""

    variant_label: Mapped[str] = mapped_column()
    """Template - e.g., "r1i1p1f1" (CMIP7 uses prefixed strings)"""

    variable_id: Mapped[str] = mapped_column()
    """CV - variable root name"""

    grid_label: Mapped[str] = mapped_column()
    """CV - e.g., "gn", "gr" """

    frequency: Mapped[str] = mapped_column()
    """CV - e.g., "mon", "day" """

    region: Mapped[str] = mapped_column()
    """CV - e.g., "glb" (global)"""

    branding_suffix: Mapped[str] = mapped_column()
    """Template - e.g., "tavg-h2m-hxy-u" """

    version: Mapped[str] = mapped_column()
    """Template - e.g., "v20250622".

    Only write this through an ORM instance.
    The base-table ``version_key`` ordering key is synced by the ``_sync_version_key`` mapper event.
    """

    # Additional Mandatory Attributes
    mip_era: Mapped[str] = mapped_column()
    """Always "CMIP7" """

    realm: Mapped[str] = mapped_column(nullable=True)
    """CV - e.g., "atmos", "ocean" (replaces table_id for filtering)"""

    nominal_resolution: Mapped[str] = mapped_column(nullable=True)
    """CV - e.g., "100 km" """

    # Conditionally Required - Parent Info (nullable)
    branch_time_in_child: Mapped[float] = mapped_column(nullable=True)
    """Float - when parent exists"""

    branch_time_in_parent: Mapped[float] = mapped_column(nullable=True)
    """Float - when parent exists"""

    parent_activity_id: Mapped[str] = mapped_column(nullable=True)
    """String - parent activity identifier"""

    parent_experiment_id: Mapped[str] = mapped_column(nullable=True)
    """String - parent experiment identifier"""

    parent_mip_era: Mapped[str] = mapped_column(nullable=True)
    """String - "CMIP6" or "CMIP7" """

    parent_source_id: Mapped[str] = mapped_column(nullable=True)
    """String - parent model identifier"""

    parent_time_units: Mapped[str] = mapped_column(nullable=True)
    """String - time units used in parent"""

    parent_variant_label: Mapped[str] = mapped_column(nullable=True)
    """String - parent variant label"""

    # Additional Mandatory Attributes
    license_id: Mapped[str] = mapped_column(nullable=True)
    """CV - e.g., "CC-BY-4.0", "CC0-1.0" """

    # Conditionally Required Attributes
    external_variables: Mapped[str] = mapped_column(nullable=True)
    """Space-separated list of cell measure variable names (when cell_measures are specified)"""

    # Variable Metadata (optional, useful for display)
    standard_name: Mapped[str] = mapped_column(nullable=True)
    """CF standard name"""

    long_name: Mapped[str] = mapped_column(nullable=True)
    """Human-readable description"""

    units: Mapped[str] = mapped_column(nullable=True)
    """Variable units"""

    # Unique Identifier
    instance_id: Mapped[str] = mapped_column(index=True)
    """CMIP7 DRS format unique identifier"""

    time_units: Mapped[str] = mapped_column(nullable=True)
    """Time encoding units (e.g. 'days since 1850-01-01')"""

    calendar: Mapped[str] = mapped_column(nullable=True)
    """CF calendar type (e.g. 'standard', '360_day', 'noleap')"""

    @hybrid_property
    def branded_variable(self) -> str:
        """Return branded variable: ``{variable_id}_{branding_suffix}``."""
        return f"{self.variable_id}_{self.branding_suffix}"

    @branded_variable.inplace.expression
    @classmethod
    def _branded_variable_expression(cls) -> ColumnElement[str]:
        return cls.variable_id + "_" + cls.branding_suffix

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.CMIP7}  # type: ignore
