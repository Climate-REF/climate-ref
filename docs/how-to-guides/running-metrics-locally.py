# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # Testing metric providers locally
# Metric providers can be run locally without requiring the rest of the REF infrastructure.
# This is useful for testing and debugging metrics.
#
# Running a metric locally requires that the target REF metrics package, e.g. `cmip_ref_metrics_example`,
# and its dependencies are installed in the current Python environment.
#
# This guide will walk you through how to run a metric provider locally.


# %% tags=["remove_input"]
import json
from pathlib import Path

import cmip_ref_metrics_example
import pandas as pd
import prettyprinter

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.datasets import get_dataset_adapter
from cmip_ref.solver import solve_metric_executions
from cmip_ref_core.datasets import SourceDatasetType

prettyprinter.install_extras(["attrs"])

# %%
provider = cmip_ref_metrics_example.provider
provider

# %% [markdown]
# We select a metric which simply calculates the annual mean, global mean timeseries of a dataset.
#
# The data requirements of this metric filter out all variables except `tas` and `rsut`.
# The `group_by` specification ensures that each execution has a unique combination of
# `source_id`, `variant_id`, `variable_id` and `experiment_id` values.

# %%
metric = provider.get("global-mean-timeseries")

# %%
prettyprinter.pprint(metric.data_requirements[0])

# %% tags=["hide_code"]
config = Config.default()
provider.configure(config)
db = Database.from_config(config)

# %% [markdown]
# Load the data catalog containing the CMIP6 datasets.
# This contains the datasets that have been ingested into the REF database.
# You could also use the `find_local_datasets` function to find local datasets on disk,
# thereby bypassing the need for a database.

# %%
# Load the data catalog containing the
data_catalog = get_dataset_adapter("cmip6").load_catalog(db)
data_catalog.head()

# %% [markdown]
# Below, the unique combinations of the metadata values that apply to the groupby are shown:

# %%
data_catalog[["source_id", "variant_label", "variable_id", "experiment_id"]].drop_duplicates()

# %% [markdown]
#
# ## Metric Executions
#
# A metric execution is a combination of a metric, a provider, and the data needed to run the metric.
#
# The `MetricSolver` is used to determine which metric executions are required given a set of requirements
# and the currently available dataset.
# This doesn't require the use of the REF database.

# %%
metric_executions = solve_metric_executions(
    data_catalog={
        SourceDatasetType.CMIP6: data_catalog,
    },
    metric=provider.get("global-mean-timeseries"),
    provider=provider,
)

# Convert from a generator to a list to inspect the complete set of results
metric_executions = list(metric_executions)
prettyprinter.pprint(metric_executions)

# %% [markdown]
# We get multiple proposed executions.

# %%
pd.concat(execution.metric_dataset["cmip6"] for execution in metric_executions)[
    ["experiment_id", "variable_id"]
].drop_duplicates()

# %% [markdown]
# Each execution contains a single unique dataset because of the groupby definition.
# The data catalog for the metric execution may contain more than one row
# as a dataset may contain multiple files.

# %%
metric_executions[0].metric_dataset["cmip6"].instance_id.unique().tolist()

# %%
metric_executions[0].metric_dataset["cmip6"]

# %% [markdown]
#
# ## Metric Definitions
#
# Each metric execution requires a `MetricExecutionDefinition` object.
# This object contains the information about where data should be stored
# and which datasets should be used for the metric calculation.

# %%
output_directory = Path("./out")
definition = metric_executions[0].build_metric_execution_info(output_directory)
prettyprinter.pprint(definition)


# %% [markdown]
# ### Running directly locally
#
# A metric can be run directly if you want to run a calculation synchronously
# without any additional infrastructure.
#
# This will not perform and validation/verification of the output results.

# %%
direct_result = metric.run(definition=definition)
assert direct_result.successful

prettyprinter.pprint(direct_result)

# %% [markdown]
# ## Metric calculations
#
# Metric calculations are typically run using an
# [Executor](../../api/cmip_ref_core/executor/#cmip_ref_core.executor.Executor)
# which provides an abstraction to enable metrics to be run in multiple different ways.
# These executors can run metrics locally, on a cluster, or on a remote service
#
# The simplest executor is the
# [LocalExecutor](../../api/cmip_ref/executor/local/#cmip_ref.executor.local.LocalExecutor).
# This executor runs a given metric synchronously in the current process.
#
# The executor can be specified in the configuration, or
# using the `REF_EXECUTOR` environment variable which takes precedence.
# The [LocalExecutor](../../api/cmip_ref/executor/local/#cmip_ref.executor.local.LocalExecutor)
# is the default executor,
# if no other configuration is provided.

# %%
executor = config.executor.build(config=config, database=db)
metric = provider.get("global-mean-timeseries")

executor.run_metric(provider, metric, definition=definition)

# %%
output_file = definition.to_output_path("output.json")
with open(output_file) as fh:
    # Load the output bundle and pretty print
    loaded_result = json.loads(fh.read())
    print(json.dumps(loaded_result, indent=2))


# %%
