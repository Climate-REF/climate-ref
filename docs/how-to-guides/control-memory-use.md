# How to control memory use and parallism

The diagnostics packages used by the REF all use [Dask](https://dask.org/) to
process data that is larger than memory in parallel. By default, Dask
uses its threaded [scheduler](https://docs.dask.org/en/stable/scheduling.html),
which may not be optimal for more complicated computations, and it configures
this threaded scheduler to use as many worker threads as there are CPU cores on
the machine. Because the REF typically runs multiple executors in parallel
(see [Executors](executors.md)), and if unconfigured, each executor uses as many
threads as there are CPU cores, this can lead to excessive memory use and too
much parallism, which can cause the system to run out of memory or become slow
because of excessive context switching. Inefficient scheduling by the threaded
scheduler can also lead to excessive memory use and/or slow computations.
Therefore, it is highly recommended that you take a moment to configure Dask
for your system. For an in-depth introduction to these topics, see the Dask documentation on
[configuration](https://docs.dask.org/en/latest/configuration.html) and
[scheduling](https://docs.dask.org/en/latest/scheduling.html).

## Configuring ESMValTool

[ESMValCore](https://docs.esmvaltool.org/projects/ESMValCore/en/latest/), the
framework powering [ESMValTool](https://docs.esmvaltool.org), works best with
the Dask Distributed scheduler. It is recommended to set `max_parallel_tasks`
(an ESMValCore setting), to a low number, e.g. 1, 2 or 3, because only one
[ESMValCore preprocessing task](https://docs.esmvaltool.org/projects/ESMValCore/en/latest/recipe/overview.html#the-diagnostics-section-defines-tasks) will submit jobs to the Distributed
scheduler at a time to avoid overloading the workers. Therefore, the following
settings are recommended for ESMValCore:

```yaml
max_parallel_tasks: 2
dask:
  use: local_distributed
  profiles:
    local_distributed:
      cluster:
        type: distributed.LocalCluster
        n_workers: 2
        threads_per_worker: 2
        memory_limit: 4GiB
```

These settings should be put in a file with the extension `.yaml` in the
directory `~/.config/esmvaltool`, for example: `~/.config/esmvaltool/dask.yaml`.

With the settings above, the total memory per REF diagnostic execution will be
`n_workers * memory_limit` = 8GB.
It is recommended to use at least 4GB of RAM per
[Dask Distributed worker](https://distributed.dask.org/en/latest/worker.html).
Some diagnostics may be able to run with 2GB per worker, but probably not all of them.
You can tune the total memory / CPU use by specifying the number of workers.
The number of CPU cores used will be `n_workers * threads_per_worker`.

Note that the REF may run multiple executors in parallel, and each executor
running an ESMValTool diagnostic will use the resources specified above.

More information on how to configure ESMValCore is available in its
[documentation](https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html).

/// tip | ESMValTool users
If you are using ESMValTool outside of the REF on the same computer, it is highly
recommended that you create a separate ESMValTool configuration directory for
the version of ESMValTool used by the REF to avoid conflicts, e.g. accidentally
using input data that is not managed by the REF. You can do this by setting the
`ESMVALTOOL_CONFIG_DIR` environment variable to a different directory,
e.g. `~/.config/esmvaltool-ref`, and then creating the Dask configuration file
in that directory, e.g. `~/.config/esmvaltool-ref/dask.yaml` with the settings
described above.
///

# Configuring PMP and ILAMB/IOMB

Both [ILAMB/IOMB](https://ilamb3.readthedocs.io) and
[PMP](https://pcmdi.github.io/pcmdi_metrics/) use Dask through Xarray, but they
do not expose their own Dask configuration options. Therefore, you need to
configure Dask globally for these packages by creating a Dask configuration
file, e.g. at `~/.config/dask/config.yaml`.

While testing the REF, we have seen occasional crashes when running ILAMB/IOMB
and PMP diagnostics with the threaded scheduler, so we recommend using the
synchronous scheduler for these diagnostics by adding the following content
to the global Dask configuration file:

```yaml
scheduler: synchronous
```

For faster processing, you can can use the threaded scheduler with a limited
number of worker threads (adjust to your system's resources):

```yaml
scheduler: threads
num_workers: 4
```

Note that the REF may run multiple executors in parallel, and each executor
running a PMP diagnostic will use the resources specified above.

/// note | ILAMB/IOMB diagnostics
The ILAMB/IOMB diagnostics are currently [restricted to the synchronous scheduler](https://github.com/Climate-REF/climate-ref/blob/24ada711091f6821793c5b3034e615c50dad68df/packages/climate-ref-ilamb/src/climate_ref_ilamb/standard.py#L651), so they will not respect the global Dask settings.
///
