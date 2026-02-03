import datetime
from typing import Any, ClassVar

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from climate_ref.models.base import Base
from climate_ref_core.datasets import SourceDatasetType


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

    def __repr__(self) -> str:
        return f"<Dataset slug={self.slug} dataset_type={self.dataset_type} >"

    __mapper_args__: ClassVar[Any] = {"polymorphic_on": dataset_type}  # type: ignore


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

    start_time: Mapped[datetime.datetime] = mapped_column(nullable=True)
    """
    Start time of a given file
    """

    end_time: Mapped[datetime.datetime] = mapped_column(nullable=True)
    """
    Start time of a given file
    """

    path: Mapped[str] = mapped_column()
    """
    Prefix that describes where the dataset is stored relative to the data directory
    """

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

    instance_id: Mapped[str] = mapped_column(index=True)
    """
    Unique identifier for the dataset (including the version).
    """

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.CMIP6}  # type: ignore


class Obs4MIPsDataset(Dataset):
    """
    Represents a obs4mips dataset

    TODO: Should the metadata fields be part of the file or dataset?
    """

    __tablename__ = "obs4mips_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    activity_id: Mapped[str] = mapped_column()
    frequency: Mapped[str] = mapped_column()
    grid: Mapped[str] = mapped_column()
    grid_label: Mapped[str] = mapped_column()
    institution_id: Mapped[str] = mapped_column()
    long_name: Mapped[str] = mapped_column()
    nominal_resolution: Mapped[str] = mapped_column()
    realm: Mapped[str] = mapped_column()
    product: Mapped[str] = mapped_column()
    source_id: Mapped[str] = mapped_column()
    source_type: Mapped[str] = mapped_column()
    units: Mapped[str] = mapped_column()
    variable_id: Mapped[str] = mapped_column()
    variant_label: Mapped[str] = mapped_column()
    version: Mapped[str] = mapped_column()
    vertical_levels: Mapped[int] = mapped_column()
    source_version_number: Mapped[str] = mapped_column()

    instance_id: Mapped[str] = mapped_column()
    """
    Unique identifier for the dataset.
    """
    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.obs4MIPs}  # type: ignore


class PMPClimatologyDataset(Dataset):
    """
    Represents a climatology dataset from PMP

    These data are similar to obs4MIPs datasets, but are post-processed
    """

    __tablename__ = "pmp_climatology_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    activity_id: Mapped[str] = mapped_column()
    frequency: Mapped[str] = mapped_column()
    grid: Mapped[str] = mapped_column()
    grid_label: Mapped[str] = mapped_column()
    institution_id: Mapped[str] = mapped_column()
    long_name: Mapped[str] = mapped_column()
    nominal_resolution: Mapped[str] = mapped_column()
    realm: Mapped[str] = mapped_column()
    product: Mapped[str] = mapped_column()
    source_id: Mapped[str] = mapped_column()
    source_type: Mapped[str] = mapped_column()
    units: Mapped[str] = mapped_column()
    variable_id: Mapped[str] = mapped_column()
    variant_label: Mapped[str] = mapped_column()
    version: Mapped[str] = mapped_column()
    vertical_levels: Mapped[int] = mapped_column()
    source_version_number: Mapped[str] = mapped_column()

    instance_id: Mapped[str] = mapped_column()
    """
    Unique identifier for the dataset.
    """
    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.PMPClimatology}  # type: ignore


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
    """Template - e.g., "v20250622" """

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

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.CMIP7}  # type: ignore
