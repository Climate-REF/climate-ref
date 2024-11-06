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

# Running a metric locally requires that the target REF metrics package, e.g. `ref-metrics-example`,
# and its dependencies are installed in the current Python environment.
#
# This guide will walk you through how to run a metric provider locally.
# %%
import json
import pathlib

import ref_metrics_example
from ref_core.executor import run_metric
from ref_core.metrics import Configuration, TriggerInfo

# %%
provider = ref_metrics_example.provider
provider

# %%
# Relative path to some CMIP6 data
example_dataset = (
    pathlib.Path("CMIP6")
    / "ScenarioMIP"
    / "CSIRO"
    / "ACCESS-ESM1-5"
    / "ssp126"
    / "r1i1p1f1"
    / "Amon"
    / "tas"
    / "gn"
    / "v20210318"
)

# %%
configuration = Configuration(output_fragment=pathlib.Path("out"))
trigger = TriggerInfo(dataset=example_dataset)

# %% [markdown]
# ## Metric calculations
#
# Metric calculations are typically run using an [Executor](ref_core.executor.Executor)
# which provides an abstraction to enable metrics to be run in multiple different ways.
#
# The simplest executor is the `LocalExecutor`.
# This executor runs a given metric synchronously in the current process.
#
# The LocalExectuor is the default executor when using the  `ref_core.executor.run_metric` function.
# This can be overridden by specifying the `REF_EXECUTOR` environment variable.

# %%
result = run_metric("annual-global-mean-timeseries", provider, configuration=configuration, trigger=trigger)
result

# %%
with open(configuration.as_output_path(result.output_bundle)) as fh:
    # Load the output bundle and pretty print
    loaded_result = json.loads(fh.read())
    print(json.dumps(loaded_result, indent=2))

# %% [markdown]
# ### Running directly
#
# The local executor can be bypassed if you need access to running the metric calculation directly.
# This will not perform and validation/verification of the output results.

# %%
metric = provider.get("annual-global-mean-timeseries")

direct_result = metric.run(configuration=configuration, trigger=trigger)
assert direct_result.successful

direct_result

# %%
