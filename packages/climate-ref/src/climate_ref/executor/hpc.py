"""
HPC-based Executor to use job schedulers.

If you want to
- run REF under the HPC workflows
- run REF in multiple nodes

"""

try:
    import parsl
except ImportError:  # pragma: no cover
    raise ImportError("The HPCExecutor requires the `parsl` package")

import os
import time
from typing import Any

import parsl
from loguru import logger
from parsl import python_app
from parsl.config import Config as ParslConfig
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher, SrunLauncher
from parsl.providers import SlurmProvider
from tqdm import tqdm

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Execution
from climate_ref.slurm import HAS_REAL_SLURM, SlurmChecker
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import DiagnosticError, ExecutionError
from climate_ref_core.executor import execute_locally

from .local import ExecutionFuture, process_result
from .pbs_scheduler import SmartPBSProvider


@python_app
def _process_run(definition: ExecutionDefinition, log_level: str) -> ExecutionResult:
    """Run the function on computer nodes"""
    # This is a catch-all for any exceptions that occur in the process and need to raise for
    # parsl retries to work
    try:
        return execute_locally(definition=definition, log_level=log_level, raise_error=True)
    except DiagnosticError as e:  # pragma: no cover
        # any diagnostic error will be caught here
        logger.exception("Error running diagnostic")
        raise e


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, int | float):
        return float(x)
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    try:
        return int(float(x))  # Handles both "123" and "123.0"
    except (ValueError, TypeError):
        return None


class HPCExecutor:
    """
    Run diagnostics by submitting a job script

    """

    name = "hpc"

    def __init__(
        self,
        *,
        database: Database | None = None,
        config: Config | None = None,
        **executor_config: str | float | int,
    ) -> None:
        config = config or Config.default()
        database = database or Database.from_config(config, run_migrations=False)

        self.config = config
        self.database = database

        self.scheduler = executor_config.get("scheduler", "slurm")
        self.account = str(executor_config.get("account", os.environ.get("USER")))
        self.username = executor_config.get("username", os.environ.get("USER"))
        self.partition = str(executor_config.get("partition")) if executor_config.get("partition") else None
        self.queue = str(executor_config.get("queue")) if executor_config.get("queue") else None
        self.qos = str(executor_config.get("qos")) if executor_config.get("qos") else None
        self.req_nodes = int(executor_config.get("req_nodes", 1)) if self.scheduler == "slurm" else 1
        self.walltime = str(executor_config.get("walltime", "00:10:00"))
        self.log_dir = str(executor_config.get("log_dir", "runinfo"))

        self.cores_per_worker = _to_int(executor_config.get("cores_per_worker"))
        self.mem_per_worker = _to_float(executor_config.get("mem_per_worker"))

        hours, minutes, seconds = map(int, self.walltime.split(":"))
        total_minutes = hours * 60 + minutes + seconds / 60
        self.total_minutes = total_minutes

        if executor_config.get("validation") and HAS_REAL_SLURM:
            self._validate_slurm_params()

        self._initialize_parsl()

        self.parsl_results: list[ExecutionFuture] = []

    def _validate_slurm_params(self) -> None:
        """Validate the Slurm configuration using SlurmChecker.

        Raises
        ------
            ValueError: If account, partition or QOS are invalid or inaccessible.
        """
        slurm_checker = SlurmChecker()
        if self.account and not slurm_checker.get_account_info(self.account):
            raise ValueError(f"Account: {self.account} not valid")

        partition_limits = None
        node_info = None

        if self.partition:
            if not slurm_checker.get_partition_info(self.partition):
                raise ValueError(f"Partition: {self.partition} not valid")

            if not slurm_checker.can_account_use_partition(self.account, self.partition):
                raise ValueError(f"Account: {self.account} cannot access partiton: {self.partition}")

            partition_limits = slurm_checker.get_partition_limits(self.partition)
            node_info = slurm_checker.get_node_from_partition(self.partition)

        qos_limits = None
        if self.qos:
            if not slurm_checker.get_qos_info(self.qos):
                raise ValueError(f"QOS: {self.qos} not valid")

            if not slurm_checker.can_account_use_qos(self.account, self.qos):
                raise ValueError(f"Account: {self.account} cannot access qos: {self.qos}")

            qos_limits = slurm_checker.get_qos_limits(self.qos)

        max_cores_per_node = int(node_info["cpus"]) if node_info else None
        if max_cores_per_node and self.cores_per_worker:
            if self.cores_per_worker > max_cores_per_node:
                raise ValueError(
                    f"cores_per_work:{self.cores_per_worker}"
                    f"larger than the maximum in a node {max_cores_per_node}"
                )

        max_mem_per_node = float(node_info["real_memory"]) if node_info else None
        if max_mem_per_node and self.mem_per_worker:
            if self.mem_per_worker > max_mem_per_node:
                raise ValueError(
                    f"mem_per_work:{self.mem_per_worker}"
                    f"larger than the maximum mem in a node {max_mem_per_node}"
                )

        max_walltime_partition = (
            partition_limits["max_time_minutes"] if partition_limits else self.total_minutes
        )
        max_walltime_qos = qos_limits["max_time_minutes"] if qos_limits else self.total_minutes

        max_walltime_minutes = min(float(max_walltime_partition), float(max_walltime_qos))

        if self.total_minutes > float(max_walltime_minutes):
            raise ValueError(
                f"Walltime: {self.walltime} exceed the maximum time "
                f"{max_walltime_minutes} allowed by {self.partition} and {self.qos}"
            )

    def _initialize_parsl(self) -> None:
        executor_config = self.config.executor.config

        provider: SlurmProvider | SmartPBSProvider
        if self.scheduler == "slurm":
            provider = SlurmProvider(
                account=self.account,
                partition=self.partition,
                qos=self.qos,
                nodes_per_block=self.req_nodes,
                max_blocks=int(executor_config.get("max_blocks", 1)),
                scheduler_options=executor_config.get("scheduler_options", "#SBATCH -C cpu"),
                worker_init=executor_config.get("worker_init", "source .venv/bin/activate"),
                launcher=SrunLauncher(
                    debug=True,
                    overrides=executor_config.get("overrides", ""),
                ),
                walltime=self.walltime,
                cmd_timeout=int(executor_config.get("cmd_timeout", 120)),
            )

        elif self.scheduler == "pbs":
            provider = SmartPBSProvider(
                account=self.account,
                queue=self.queue,
                worker_init=executor_config.get("worker_init", "source .venv/bin/activate"),
                nodes_per_block=_to_int(executor_config.get("nodes_per_block", 1)),
                cpus_per_node=_to_int(executor_config.get("cpus_per_node", None)),
                ncpus=_to_int(executor_config.get("ncpus", None)),
                mem=executor_config.get("mem", "4GB"),
                jobfs=executor_config.get("jobfs", "10GB"),
                storage=executor_config.get("storage", ""),
                init_blocks=executor_config.get("init_blocks", 1),
                min_blocks=executor_config.get("min_blocks", 0),
                max_blocks=executor_config.get("max_blocks", 1),
                parallelism=executor_config.get("parallelism", 1),
                scheduler_options=executor_config.get("scheduler_options", ""),
                launcher=SimpleLauncher(),
                walltime=self.walltime,
                cmd_timeout=int(executor_config.get("cmd_timeout", 120)),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")

        executor = HighThroughputExecutor(
            label="ref_hpc_executor",
            cores_per_worker=self.cores_per_worker if self.cores_per_worker else 1,
            mem_per_worker=self.mem_per_worker,
            max_workers_per_node=_to_int(executor_config.get("max_workers_per_node", 16)),
            cpu_affinity=str(executor_config.get("cpu_affinity")),
            provider=provider,
        )

        hpc_config = ParslConfig(
            run_dir=self.log_dir,
            executors=[executor],
            retries=int(executor_config.get("retries", 2)),
        )

        parsl.load(hpc_config)

    def run(
        self,
        definition: ExecutionDefinition,
        execution: Execution | None = None,
    ) -> None:
        """
        Run a diagnostic in process

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic
        execution
            A database model representing the execution of the diagnostic.
            If provided, the result will be updated in the database when completed.
        """
        # Submit the execution to the process pool
        # and track the future so we can wait for it to complete
        future = _process_run(
            definition=definition,
            log_level=self.config.log_level,
        )

        self.parsl_results.append(
            ExecutionFuture(
                future=future,
                definition=definition,
                execution_id=execution.id if execution else None,
            )
        )

    def join(self, timeout: float) -> None:
        """
        Wait for all diagnostics to finish

        This will block until all diagnostics have completed or the timeout is reached.
        If the timeout is reached, the method will return and raise an exception.

        Parameters
        ----------
        timeout
            Timeout in seconds (won't used in HPCExecutor)

        Raises
        ------
        TimeoutError
            If the timeout is reached
        """
        start_time = time.time()
        refresh_time = 0.5

        results = self.parsl_results
        t = tqdm(total=len(results), desc="Waiting for executions to complete", unit="execution")

        try:
            while results:
                # Iterate over a copy of the list and remove finished tasks
                for result in results[:]:
                    if result.future.done():
                        # Cannot catch the execption raised by result.future.result
                        if result.future.exception() is None:
                            try:
                                execution_result = result.future.result(timeout=0)
                            except Exception as e:
                                # Something went wrong when attempting to run the execution
                                # This is likely a failure in the execution itself not the diagnostic
                                raise ExecutionError(
                                    f"Failed to execute {result.definition.execution_slug()!r}"
                                ) from e
                        else:
                            err = result.future.exception()
                            if isinstance(err, DiagnosticError):
                                execution_result = err.result
                            else:
                                execution_result = None

                        assert execution_result is not None, "Execution result should not be None"
                        assert isinstance(execution_result, ExecutionResult), (
                            "Execution result should be of type ExecutionResult"
                        )
                        # Process the result in the main process
                        # The results should be committed after each execution
                        with self.database.session.begin():
                            execution = (
                                self.database.session.get(Execution, result.execution_id)
                                if result.execution_id
                                else None
                            )
                            process_result(self.config, self.database, execution_result, execution)
                        logger.debug(f"Execution completed: {result}")
                        t.update(n=1)
                        results.remove(result)

                # Break early to avoid waiting for one more sleep cycle
                if len(results) == 0:
                    break

                elapsed_time = time.time() - start_time

                if elapsed_time > self.total_minutes * 60:
                    logger.debug(f"Time elasped {elapsed_time} for joining the results")

                # Wait for a short time before checking for completed executions
                time.sleep(refresh_time)
        finally:
            t.close()
            if parsl.dfk():
                parsl.dfk().cleanup()
