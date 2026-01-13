# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using pre-computed results
#
# Results computed by the CMIP7 Assessment Fast Track Rapid Evaluation Framework are available from
# the website: https://dashboard.climate-ref.org and the associated API: https://api.climate-ref.org.
# This API provides an [OpenAPI](https://www.openapis.org) schema that documents what queries are available.
# The API documentation can be viewed at: https://api.climate-ref.org/docs.
#
# This Jupyter notebook shows how to use this API to download pre-computed results and use those to do
# your own analyses.

# %% [markdown]
# ## Generate and install
#
# We start by generating and installing a Python package for interacting with the API
# from the OpenAPI-compatible [schema](https://api.climate-ref.org/api/v1/openapi.json).

# %%
# !uvx --quiet --from openapi-python-client openapi-python-client generate --url https://api.climate-ref.org/api/v1/openapi.json --meta setup --output-path climate_ref_client --overwrite

# %%
# !pip install --quiet ./climate_ref_client

# %% [markdown]
# ## Set up the notebook
#
# Import some libraries and load the [rich](https://rich.readthedocs.io/en/latest/introduction.html)
# Jupyter notebook extension for conveniently viewing the large data structures produced by the client
# package.

# %%
from pathlib import Path

import cartopy.crs
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import xarray as xr
from climate_rapid_evaluation_framework_client import Client
from climate_rapid_evaluation_framework_client.api.diagnostics import (
    diagnostics_list,
    diagnostics_list_metric_values,
)
from climate_rapid_evaluation_framework_client.api.executions import executions_get
from climate_rapid_evaluation_framework_client.models.metric_value_type import (
    MetricValueType,
)
from IPython.display import Markdown
from pandas_indexing import formatlevel

# %%
# %load_ext rich

# %% [markdown]
# ## View the available diagnostics
#
# We start by setting up a client for interacting with the server:

# %%
client = Client("https://api.climate-ref.org")

# %% [markdown]
# Retrieve the available diagnostics from the server, and inspect the first one:

# %%
diagnostics = diagnostics_list.sync(client=client).data
diagnostics[0]

# %% [markdown]
# To get an idea of what is available, we create a list of all diagnostics
# with short descriptions (a full overview is available in Appendix C of
# [Hoffman et al., 2025](https://doi.org/10.5194/egusphere-2025-2685)):

# %%
txt = ""
for diagnostic in sorted(diagnostics, key=lambda diagnostic: diagnostic.name):
    title = f"### {diagnostic.name}"
    description = diagnostic.description.strip()
    if not description.endswith("."):
        description += "."
    if diagnostic.aft_link:
        description += f" {diagnostic.aft_link.short_description.strip()}"
        if not description.endswith("."):
            description += "."
        if (aft_description := diagnostic.aft_link.description.strip()) != "nan":
            description += f" {aft_description}"
        if not description.endswith("."):
            description += "."
    txt += f"{title}\n{description}\n\n"
Markdown(txt)

# %% [markdown]
# ## Metrics
#
# Many of the diagnostics provide "metric" values, single values that describe some property
# of a model. Here we show how to access these values and create a plot.

# %%
# Select the "Atlantic Meridional Overturning Circulation (RAPID)"
# diagnostic as an example
diagnostic_name = "Atlantic Meridional Overturning Circulation (RAPID)"
diagnostic = next(d for d in diagnostics if d.name == diagnostic_name)
# Inspect an example value.
diagnostics_list_metric_values.sync(
    diagnostic.provider.slug,
    diagnostic.slug,
    value_type=MetricValueType.SCALAR,
    client=client,
).data[0]

# %% [markdown]
# Read the metric values into a Pandas DataFrame:

# %%
df = (
    pd.DataFrame(
        metric.dimensions.additional_properties | {"value": metric.value}
        for metric in diagnostics_list_metric_values.sync(
            diagnostic.provider.slug,
            diagnostic.slug,
            value_type=MetricValueType.SCALAR,
            client=client,
        ).data
    )
    .replace("None", pd.NA)
    .drop_duplicates()
)
# Drop a few columns that appear to be the same for all entries of
# particular diagnostic.
df.drop(columns=["experiment_id", "metric", "region"], inplace=True)
# Use the columns that do not contain the metric value for indexing
df.set_index([c for c in df.columns if c != "value"], inplace=True)
df

# %% [markdown]
# and create a portrait diagram:

# %%
# Use the median metric value for models with multiple ensemble
# members to keep the figure readable.
df = df.groupby(level=["source_id", "grid_label", "statistic"]).median()
# Convert df to a "2D" dataframe for use with the seaborn heatmap plot
df_2D = (
    formatlevel(df, model="{source_id}.{grid_label}", drop=True)
    .reset_index()
    .pivot(columns="statistic", index="model", values="value")
)
figure, ax = plt.subplots(figsize=(5, 8))
_ = sns.heatmap(
    df_2D / df_2D.median(),
    annot=df_2D,
    cmap="viridis",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Color indicates value relative to the median"},
)
# %% [markdown]
# ## Series
#
# Many of the diagnostics provide "series" values, a range of values along with an index
# that describe some property of a model. Here we show how to access these values and create a plot.

# %%
# Select the "Sea Ice Area Basic Metrics" diagnostic as an example
diagnostic_name = "Sea Ice Area Basic Metrics"
diagnostic = next(d for d in diagnostics if d.name == diagnostic_name)
# Inspect an example series value:
diagnostics_list_metric_values.sync(
    diagnostic.provider.slug,
    diagnostic.slug,
    value_type=MetricValueType.SERIES,
    client=client,
).data[0]

# %% [markdown]
# Read the metric values into a Pandas DataFrame:

# %%
statistic_name = "20-year average seasonal cycle"
value_name = "sea ice area (1e6 km2)"
df = pd.DataFrame(
    metric.dimensions.additional_properties | {value_name: value, "month": int(month)}
    for metric in diagnostics_list_metric_values.sync(
        diagnostic.provider.slug,
        diagnostic.slug,
        value_type=MetricValueType.SERIES,
        client=client,
    ).data
    if metric.dimensions.additional_properties["statistic"].startswith(statistic_name)
    for value, month in zip(metric.values, metric.index)
    if value < 1e10  # Ignore some invalid values.
)
df

# %% [markdown]
# and create a plot:

# %%
_ = sns.relplot(
    data=df.sort_values("source_id"),
    x="month",
    y=value_name,
    col="region",
    hue="source_id",
    kind="line",
)
# %% [markdown]
# ## Files
#
# Many of the diagnostics produce NetCDF files that can be used for further analysis or custom plotting.
# We will look at the global warming levels diagnostic and create our own figure using the available data.
#
# Each diagnostic can be run (executed) multiple times with different input data. The global warmings
# levels diagnostic has been executed several times, leading to multiple "execution groups":

# %%
diagnostic_name = "Climate at Global Warming Levels"
diagnostic = next(d for d in diagnostics if d.name == diagnostic_name)
[executions_get.sync(g, client=client).key for g in diagnostic.execution_groups]

# %% [markdown]
# Let's select the "ssp585" scenario and look at the output files that were produced:

# %%
for group in diagnostic.execution_groups:
    execution = executions_get.sync(group, client=client)
    if execution.key.endswith("ssp585"):
        ssp585_outputs = execution.latest_execution.outputs
        break
else:
    msg = "Failed to find the ssp585 execution group"
    raise ValueError(msg)
[o.filename for o in ssp585_outputs]

# %% [markdown]
# Select one of the output files and inspect it:

# %%
filename = "tas/plot_gwl_stats/CMIP6_mm_mean_2.0.nc"
file = next(f for f in ssp585_outputs if f.filename.endswith(filename))
file

# %% [markdown]
# Download the file and open it with `xarray`:

# %%
local_file = Path("/tmp") / Path(file.filename).name  # noqa: S108
local_file.write_bytes(requests.get(file.url, timeout=120).content)
ds = xr.open_dataset(local_file).drop_vars("cube_label")
ds

# %% [markdown]
# Create our own plot:

# %%
plot = ds.tas.plot.contourf(
    cmap="viridis",
    vmin=-30,
    vmax=30,
    levels=11,
    figsize=(12, 5),
    transform=cartopy.crs.PlateCarree(),
    subplot_kws={
        "projection": cartopy.crs.Orthographic(
            central_longitude=-100,
            central_latitude=40,
        ),
    },
)
_ = plot.axes.coastlines()
