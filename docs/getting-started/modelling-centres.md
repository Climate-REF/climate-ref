# Using the REF in a Modelling Centre

This page is the entry point for modelling centres preparing local model output for evaluation with Climate-REF.
It summarises what currently works well for centres, what deployment choices are available,
and what information should be captured before scaling up a REF run.

Modelling centres often need to evaluate unpublished or pre-publication data,
run on shared HPC systems, and work within local data-management conventions.
The REF can support this workflow as long as the model output is CMOR-compliant.
Publication to ESGF is not required.

## What works today

For most modelling centres, the recommended starting point is a local REF deployment using the
[HPCExecutor](../how-to-guides/hpc_executor.md).
This allows users to run commands on the login node,
while submitting diagnostic work via the batch scheduler.

The [HPCExecutor](../how-to-guides/hpc_executor.md) currently supports Slurm and PBS.
This makes it a good fit for many shared HPC systems where compute-heavy work cannot run on login nodes.
The REF master process runs lightly on the login node, while diagnostic execution is submitted to compute nodes.

If the available hardware is a workstation or single shared server,
the default [LocalExecutor](../how-to-guides/executors.md#localexecutor-default) may be enough for small runs.
For larger or repeated centre runs, moving to the HPC executor is usually the better path.

## Choosing a deployment approach

There are several ways to deploy the REF.
The right choice depends mostly on available hardware, scheduler access, and site policy.

| Deployment approach | When it fits | Notes |
| --- | --- | --- |
| Local CLI with `LocalExecutor` | Small tests, single-host runs, early troubleshooting | Easiest way to validate configuration and data ingestion. |
| Local CLI with `HPCExecutor` | Modelling centres running on Slurm or PBS HPC systems | Recommended deployment path. Keeps data local and uses the batch scheduler for diagnostic work. |
| Docker or service deployment | Production service deployments with container support | Often difficult on HPC systems where Docker is unavailable or restricted. See the [Docker deployment guide](../how-to-guides/docker_deployment.md). |
| API and dashboard deployment | Shared production access to results | Can be hard to deploy at HPC centres because it requires network access, port forwarding, or port sharing. |

Docker-based deployments are generally more appropriate for production services than for first runs at modelling centres.
Many HPC sites do not provide Docker directly,
and container or network policies can make a full API/dashboard deployment harder than a local CLI deployment.

## Accessing results

The current local workflow is CLI and filesystem based.
After `ref solve` completes, use:

```bash
ref executions list-groups
ref executions inspect <group_id>
```

The result files are stored under the configured results directory,
usually `$REF_CONFIGURATION/results`.
See the [visualise results tutorial](05-visualise.md) for the current on-disk result layout.

The REF dashboard and API provide a richer way to explore results,
but deploying them at HPC centres may require network arrangements that are not always available.
The existing [pre-computed results notebook](../how-to-guides/using-pre-computed-results.py)
uses the hosted REF API rather than local result files.

A Python API for querying local REF results would be a useful extension because it would avoid the need to deploy the API/dashboard just to analyse local results.
That local-results API, and a notebook based on it, are not currently funded.

## Suggested first run

Start with a single model, experiment, ensemble member, and monthly output where possible.
This keeps ingestion and solve outputs easy to inspect before scaling to a full archive.

1. Configure REF paths and providers using the [configuration tutorial](01-configure.md).
2. Fetch the required reference datasets using the [download datasets tutorial](02-download-datasets.md).
3. Ingest the modelling centre's CMIP data using the [ingestion tutorial](03-ingest.md).
4. Run `ref datasets list` and confirm that `institution_id`, `source_id`, `experiment_id`, member or variant, `variable_id`, `grid_label`, and `version` match expectations.
5. Configure the [HPCExecutor](../how-to-guides/hpc_executor.md) if running on Slurm or PBS.
6. Run `ref solve` for the selected data and review missing dataset messages before expanding the run.

For large archives, prefer the DRS parser so ingestion can collect path-derived metadata without opening every netCDF file which is painful for some HPC file systems.
The REF will finalise matched datasets at solve time by reading only the files needed for diagnostics.

```toml
cmip6_parser = "drs"
cmip7_parser = "drs"
```

The conda environments for the diagnostic providers consist of lots of small files.

## Unpublished data

The REF can evaluate data that has not been published to ESGF.
The important requirement is that local files have been CMOR-ised ready for publication to ESGF, and (for CMIP7) run through the QA/QC tools.
The REF does not validate every controlled-vocabulary value,
which allows centres to evaluate pre-publication data or custom `source_id` values.

## Scaling up

After the first model run is understood, scale in small steps:

- Add variables required by more diagnostics.
- Add the remaining ensemble members or variants.
- Add additional experiments.
- Add additional models or model configurations.
- Move from local execution to the HPC executor if diagnostic throughput becomes the limiting factor.

Keep the centre information up to date as model output changes.
Re-ingest after new versions become available so the REF can determine which execution groups need to be rerun.
