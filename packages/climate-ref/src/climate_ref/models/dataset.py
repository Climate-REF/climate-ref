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


class CMIP7Dataset(Dataset):
    """
    Represents a CMIP7 dataset.

    Based on CMIP7 Global Attributes V1.0 (DOI: 10.5281/zenodo.17250297).
    CMIP7 datasets have branding suffix components, variant indices as strings,
    and additional DRS version information.
    """

    __tablename__ = "cmip7_dataset"
    id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), primary_key=True)

    # Core identification (mandatory per CMIP7 spec)
    activity_id: Mapped[str] = mapped_column()
    institution_id: Mapped[str] = mapped_column()
    source_id: Mapped[str] = mapped_column(index=True)
    experiment_id: Mapped[str] = mapped_column(index=True)
    variant_label: Mapped[str] = mapped_column()
    variable_id: Mapped[str] = mapped_column()
    grid_label: Mapped[str] = mapped_column()
    version: Mapped[str] = mapped_column()

    # CMIP7-specific mandatory fields
    mip_era: Mapped[str] = mapped_column(default="CMIP7")
    region: Mapped[str] = mapped_column(default="glb")  # lowercase per spec
    frequency: Mapped[str] = mapped_column()
    branding_suffix: Mapped[str] = mapped_column()
    branded_variable: Mapped[str] = mapped_column()

    # Branding suffix components (mandatory per spec)
    temporal_label: Mapped[str] = mapped_column(nullable=True)
    vertical_label: Mapped[str] = mapped_column(nullable=True)
    horizontal_label: Mapped[str] = mapped_column(nullable=True)
    area_label: Mapped[str] = mapped_column(nullable=True)

    # DRS and spec info (mandatory per spec)
    drs_specs: Mapped[str] = mapped_column(default="MIP-DRS7")
    data_specs_version: Mapped[str] = mapped_column(default="MIP-DS7.1.0.0")
    product: Mapped[str] = mapped_column(default="model-output")
    license_id: Mapped[str] = mapped_column(default="CC-BY-4.0")

    # Variant indices (CMIP7 uses prefixed strings, e.g., "r1", "i1", "p1", "f1")
    realization_index: Mapped[str] = mapped_column(nullable=True)
    initialization_index: Mapped[str] = mapped_column(nullable=True)
    physics_index: Mapped[str] = mapped_column(nullable=True)
    forcing_index: Mapped[str] = mapped_column(nullable=True)

    # Optional metadata
    realm: Mapped[str] = mapped_column(nullable=True)
    nominal_resolution: Mapped[str] = mapped_column(nullable=True)
    tracking_id: Mapped[str] = mapped_column(nullable=True)
    standard_name: Mapped[str] = mapped_column(nullable=True)
    long_name: Mapped[str] = mapped_column(nullable=True)
    units: Mapped[str] = mapped_column(nullable=True)

    # Conditionally required parent fields (when parent exists)
    branch_time_in_child: Mapped[float] = mapped_column(nullable=True)
    branch_time_in_parent: Mapped[float] = mapped_column(nullable=True)
    parent_activity_id: Mapped[str] = mapped_column(nullable=True)
    parent_experiment_id: Mapped[str] = mapped_column(nullable=True)
    parent_mip_era: Mapped[str] = mapped_column(nullable=True)
    parent_source_id: Mapped[str] = mapped_column(nullable=True)
    parent_time_units: Mapped[str] = mapped_column(nullable=True)
    parent_variant_label: Mapped[str] = mapped_column(nullable=True)
    external_variables: Mapped[str] = mapped_column(nullable=True)

    instance_id: Mapped[str] = mapped_column(index=True)
    """
    Unique identifier for the dataset (CMIP7 DRS format).
    """

    __mapper_args__: ClassVar[Any] = {"polymorphic_identity": SourceDatasetType.CMIP7}  # type: ignore


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
