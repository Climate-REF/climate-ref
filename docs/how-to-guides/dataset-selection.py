# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Diagnostic Dataset Selection
# A diagnostic defines the requirements for the data it needs to run.
# The requirements are defined in the `data_requirements` attribute of the diagnostic class.
#
# This notebook provides some examples querying and filtering datasets, which is useful
# to understand how to configure dataset selection when integrating a diagnostic into the
# REF.

# %% tags=["remove_input"]
import pandas as pd
from IPython.display import display
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement

logger.remove()

# %%
config = Config.default()
db = Database.from_config(config)

# %% [markdown]
#
# Each source dataset type has a corresponding adapter that can be used to load the data catalog.
#
# The adapter provides a consistent interface for ingesting
# and querying datasets across different dataset types.
# It contains information such as the columns that are expected.
# %%
from climate_ref.datasets import get_dataset_adapter

adapter = get_dataset_adapter("cmip6")
adapter

# %% [markdown]
# ## Data Catalog
# Below is an example of a data catalog of the CMIP6 datasets that have already been ingested.
#
# This data catalog contains information about the datasets that are available for use in the diagnostics.
# The data catalog is a pandas DataFrame that contains information about the datasets,
# such as the `variable`, `source_id`, and other metadata.
#
# Each row represents an individual NetCDF file,
# with the rows containing the metadata associated with that file.
# There are ~36 different **facets** of metadata for a CMIP6 data file.
# Each of these facets can be used to refine the datasets that are needed for a given diagnostic execution.

# %%
data_catalog = adapter.load_catalog(db)
data_catalog


# %% [markdown]
# A dataset may consist of more than one file. In the case of CMIP6 datasets,
# the modelling centers who produce the data may chunk a dataset along the time axis.
# The size of these chunks is at the discretion of the modelling center.
#
# Datasets share a common set of metadata (see `adapter.dataset_specific_metadata`)
# which do not vary for a given dataset,
# while some facets vary within the dataset (`adapter.file_specific_metadata`).
#
# Each data catalog will have a facet that can be used to split the catalog into unique datasets
# (see `adapter.slug_column`).

# %%
adapter.slug_column

# %%
for unique_id, dataset_files in data_catalog.groupby(adapter.slug_column):
    print(unique_id)
    display(dataset_files)
    print()

# %% [markdown]
# ## Data Requirements
# Each diagnostic may be run multiple times with different groups of datasets.
#
# Determining which diagnostic executions should be performed is a three-step process:
# 1. Filter the data catalog based on the diagnostic's requirements
# 2. Group the filtered data catalog using unique metadata fields
# 3. Apply constraints to the groups to ensure the correct data is available
#
# Each group that passes the constraints is a valid group for the diagnostic to be executed.
#
# [extract_covered_datasets](api/climate_ref/solver/#climate_ref.solver.extract_covered_datasets)
# extracts the different groups
# of datasets within the data catalog that match the requirements.
# Below are some examples showing different data requests
# and the corresponding groups of datasets that would be executed.
#
# Multiple sets of these data requirements are also supported if a list of lists of data requirements
# are specified.

# %%
from climate_ref.solver import extract_covered_datasets


# %% tags=["remove_input"]
def display_groups(frames):
    for selector, frame in frames.items():
        display(f"selector: {selector}")
        display(frame[["instance_id", "source_id", "variable_id"]].drop_duplicates())


# %% [markdown]
#
# ### Facet filters
# The simplest data request is a `FacetFilter`.
# This filters the data catalog to include only the data required for a given diagnostic run.

# %%
data_requirement = DataRequirement(
    source_type=SourceDatasetType.CMIP6,
    filters=(
        # Only include "tas" and "rsut"
        FacetFilter(facets={"variable_id": ("tas", "rsut")}),
    ),
    group_by=None,
)

groups = extract_covered_datasets(data_catalog, data_requirement)

display_groups(groups)

# %% [markdown]
#
# ### Group by
# The `group_by` field can be used to split the filtered data into multiple groups,
# each of which has a unique set of values in the specified facets.
# This executions in multiple groups of datasets, each of which would correspond to a diagnostic execution.

# %%
data_requirement = DataRequirement(
    source_type=SourceDatasetType.CMIP6,
    filters=(
        # Only include "tas" and "rsut"
        FacetFilter(facets={"variable_id": ("tas", "rsut")}),
    ),
    group_by=(
        "variable_id",
        "source_id",
    ),
)

groups = extract_covered_datasets(data_catalog, data_requirement)

display_groups(groups)


# %% [markdown]
#
# ### Constraints
# A data requirement can optionally specify `Constraint`s.
# These constraints are applied to each group independently to modify a group or ignore it.
# A group must not be empty after modification for it to be executed.
#
# Constraints can be used to remove datasets or include additional datasets from the catalog,
# which is useful to select common datasets for all groups (e.g. cell areas).
#
# Below, an `IncludeTas` GroupOperation is included which adds the corresponding `tas` dataset to each group.


# %%
class IncludeTas:
    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        # we will probably need to include some helpers
        tas = data_catalog[
            (data_catalog["variable_id"] == "tas")
            & data_catalog["source_id"].isin(group["source_id"].unique())
            & data_catalog["experiment_id"].isin(group["experiment_id"].unique())
            & data_catalog["member_id"].isin(group["member_id"].unique())
        ]

        return pd.concat([group, tas])


data_requirement = DataRequirement(
    source_type=SourceDatasetType.CMIP6,
    filters=(FacetFilter(facets={"frequency": "mon"}),),
    group_by=("variable_id", "source_id", "member_id", "experiment_id"),
    constraints=(IncludeTas(),),
)

groups = extract_covered_datasets(data_catalog, data_requirement)

display_groups(groups)


# %% [markdown]
# In addition to operations adding datasets, it is also possible to remove datasets.


# %%
class AtLeast2:
    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        if len(group["variable_id"].drop_duplicates()) >= 2:
            return group
        return group.loc[[]]


# %% [markdown]
# Here we add a simple validator which ensures that at least 2 unique datasets are present.
# This removes the groups from above where tas was not available.

# %%
data_requirement = DataRequirement(
    source_type=SourceDatasetType.CMIP6,
    filters=(FacetFilter(facets={"frequency": "mon"}),),
    group_by=("variable_id", "source_id", "member_id", "experiment_id"),
    constraints=(IncludeTas(), AtLeast2()),
)

groups = extract_covered_datasets(data_catalog, data_requirement)

display_groups(groups)
