# HPCExecutor -  Run REF on HPCs

High Performance Computing Centers (HPCs) generally do not let users run computationally intensive programs and apps on their login nodes. Users must submit a batch or interactive job into a queue that provisions resources to run user programs. Therefore, the workflow for running REF on HPCs differs from that on users' local computers or workstations.

Here, we introduce the [HPCExecutor][climate_ref.executor.hpc.HPCExecutor] for running REF on HPC resources.

You could use HPCExecutor if:

  - The login nodes allow users to run a program for a long time like several hours with little computational resources (less than 25% of one CPU core and negligible memory).
  - You want to run REF under the HPC workflow i.e., submitting batch jobs.
  - The scheduler on your HPC is __slurm__ or __pbs__. We may include other schedules in the future if needed. Please make an [issue](https://github.com/Climate-REF/climate-ref/issues) describing your requirements.

## Pre-requirements

Since compute nodes on HPCs may not have internet access, you may want to prepare everything on the login node before submitting jobs:

```bash
# Set up all provider environments and fetch required data
ref providers setup

# Ingest your datasets
ref datasets ingest --source-type cmip6 /path/to/cmip6/data
ref datasets ingest --source-type obs4mips /path/to/obs4mips/data

# Clean mamba cache to avoid stale lock files
rm -rf ${HOME}/.cache/mamba/proc/
```

The `ref providers setup` command creates conda environments and downloads any reference data that diagnostics need (climatologies, auxiliary files, etc.). This ensures diagnostics can run offline on compute nodes without making network calls.

## HPCExecutor

The HPCExecutor will use the slurm provider and srun launch from parsl to submit jobs and run REF on compute nodes. Blocks are the basic computational units of parsl. It could include a single node or several compute nodes. Within a node, parsl creates several workers that run a python app across multiple cores. Each worker will compute one or several diagnostics. If the number of diagnostics is larger than the worker number, some workers will run more than one diagnostics serially. However, the more workers, the more overhead required by parsl. The steps to use HPCExecutor are as follows:

1. On the login nodes, if possible, open a `screen` or `tmux` session to keep the master process of HPCExecutor alive. Only the master process will be run on the login nodes with less than 10% of one CPU core and negligible memory, and real computations will be on compute nodes through the job submissions. If the HPC's queue time is short and the connection to the login nodes is stable, users can use the login nodes directly without using the sessions of `screen` and `tmux`.
2. Edit the `ref.toml` under the config directory if not create one.
3. Change the ref executor to HPCExecutor as follows:
```toml
[executor]
executor = "climate_ref.executor.HPCExecutor"
```
4. Add the configuration or options for the HPCExecutor based on your system and your account. The example of configuration used in the DOE Perlmutter HPC is as follows:
```toml
[executor.config]
scheduler = "slurm"
account = "m2467"
req_nodes = 1
walltime = "00:30:00"
username = "minxu"
qos = "debug"
scheduler_options = "#SBATCH -C cpu"
cores_per_worker = 1
max_workers_per_node = 64
```
If you are using NCI Gadi, you can use the following configuration:
```toml
[executor.config]
scheduler = "pbs"
account = "m1234"
queue = "normal"
walltime = "00:30:00"
username = "minxu"
ncpus = 1
mem = "4GB"
jobfs = "10GB"
storage = "/scratch/XX" # Make sure you have access to the storage
scheduler_options = ""
cores_per_worker = 1
max_workers_per_node = 64
```
5. Run `ref solve` in the session of `screen` or `tmux` or directly on the login nodes. It will have a master process running on the login nodes to monitor and submit jobs to run the real diagnostics on compute nodes.
6. Check the logs under the `runinfo` directory and `${REF_CONFIGURATION}/log` to see if there are any errors. Once the master process is done, please run `ref executions list-groups` to check the results or `ref executions inspect id` to see the error message from the providers.


/// admonition | Note

If more sbatch derivatives are needed, they could be added in the scheduler_options with `\n` as a separator.
There are other configurations available, for example:

- `log_dir` to save the parsl log and stderr and stdout from the job;
- `cores_per_worker` its default is 1, but if the providers use dask or OpenMP, its value can be bigger;
- `mem_per_worker` generally do not need to set;
- `max_workers_per_node` to set the maximum worker in a node;
- `overriders`to add additional options for `srun`

The configurations of the __HPCExecutor__ for the slurm scheduler are validated now. They include:

- `scheduler: str`, the HPC job scheduler name, either `slurm` or `pbs`
- `account: str`, the account name to be charged
- `username: str`, the name of the user to run the REF
- `partition: str`, the slurm partition name
- `qos: str`, the slurm quality of service
- `req_nodes: int`, requested node for the REF run
- `validation: boolean`, true to validate the above options with pyslurm

The following __HPCExecutor__ options are directly from parsl. Please refer to [link](https://parsl.readthedocs.io/en/stable/stubs/parsl.providers.SlurmProvider.html)

- `log_dir: str`, default="run_info"
- `cores_per_worker: int`, default=1
- `mem_per_worker: float`
- `max_workers_per_node: int`, default=16
- `walltime: str`, default="00:30:00"
- `scheduler_options: str`
- `retries: int`, default=2
- `max_blocks: int`, default=1
- `worker_init: str`
- `overrides: str`
- `cmd_timeout: int`, default=120
- `cpu_affinity: str`, default="none"

///

## Performance benchmarking

Due to the HPCExecutor parallelism distributing the computation across workers, performance is generally determined by the slowest diagnostic on a single core and the overhead introduced by parsl. However, the HPCExecutor should have good scalability with an increase in the number of diagnostics and fine grain parallelism (use of dask, OpenMP, and MPI) implemented by diagnostic providers in the future. From the following table, the overhead of parsl is almost negligible as the number of workers increases.

### Single node performance (58 diagnostics)

Determining the optimal number of workers and cores per worker is crucial for performance.
The following table shows the performance of the HPCExecutor on the NERSC Perlmutter system with 58 diagnostics,
using different numbers of workers and cores per worker.

Please run your own benchmarks to find the optimal configuration for your system and diagnostics.

| HPC     | Executor        | CPUs              | No. of Workers | No. of Cores/Worker | Time (minutes)|
| :-------|---------------- | :-----------------|:-------------: | :-----------------: | ------------: |
| NERSC   | LocalExecutor   | AMD EPYC 7763     | 58             | 1                   | 18.2          |
| NERSC   | HPCExecutor     | AMD EPYC 7763     | 64             | 1                   | 16.3          |
| NERSC   | HPCExecutor     | AMD EPYC 7763     | 32             | 1                   | 18.1          |
| NERSC   | HPCExecutor     | AMD EPYC 7763     | 16             | 1                   | 28.6          |
