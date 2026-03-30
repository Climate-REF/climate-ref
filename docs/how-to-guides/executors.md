# Executors

Executors determine how the REF schedules and runs diagnostic computations.


You can configure which executor to use in your `ref.toml` under the `[executor]` section:

```toml
[executor]
executor = "climate_ref.executor.LocalExecutor"
```

Additionally, you can configure executor-specific options in the `[executor.config]` section of your `ref.toml`.
For example, for the LocalExecutor, you can set the number of parallel jobs:

```toml
[executor.config]
n = 2
```

The REF supports four built-in executors:

## [LocalExecutor][climate_ref.executor.local.LocalExecutor] (default)

- Runs diagnostics in parallel on your local machine using a process pool.
- Good for typical desktop or laptop usage.
- Use when you want maximum CPU utilization on a single host.

## [SynchronousExecutor][climate_ref.executor.synchronous.SynchronousExecutor]

- Runs each diagnostic serially in the main Python process.
- Useful for debugging or profiling individual diagnostics.
- To enable:

```toml
[executor]
executor = "climate_ref.executor.SynchronousExecutor"
```

## [HPCExecutor][climate_ref.executor.hpc.HPCExecutor]

- Submits diagnostics as batch jobs on HPC clusters using Slurm + Parsl.
- Coordinates a master process on the login node and worker jobs on compute nodes.
- See the [HPCExecutor guide](hpc_executor.md) for setup and configuration options.

## [CeleryExecutor][climate_ref_celery.executor.CeleryExecutor]

- Distributes tasks via Celery and a message broker (e.g., Redis).
- Ideal for running REF on multi-node clusters or cloud environments.
- See the [Docker deployment guide](docker_deployment.md) for a Celery + Redis example.




## Choosing an executor

- **LocalExecutor** is recommended for most local workflows.
- **SynchronousExecutor** helps isolate issues in individual diagnostics.
- **HPCExecutor** is ideal for large-scale runs on HPC systems.
- **CeleryExecutor** suits distributed deployments in containerized or cloud setups.

Once configured, run `ref solve` as usual and the REF will use your chosen executor to schedule and execute diagnostics.

/// note | Configuring Dask
It is recommended to configure Dask for the system running your executors, to
avoid excessive memory use or context switching. Go to
[How to control memory use and parallism guide](control-memory-use.md) for
detailed instructions.
///
