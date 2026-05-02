"""
HPC-based Executor to use job schedulers.

If you want to
- run REF under the HPC workflows
- run REF in multiple nodes

The `HPCExecutor` requires the optional `parsl` dependency.
This dependency (and therefore this executor) is not available on Windows.
"""

try:
    import parsl
except ImportError:  # pragma: no cover
    from climate_ref_core.exceptions import InvalidExecutorException

    raise InvalidExecutorException(
        "climate_ref_core.executor.hpc.HPCExecutor", "The HPCExecutor requires the `parsl` package"
    )

import inspect
import os
import re
import resource
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar, cast

import parsl
from loguru import logger
from parsl import python_app
from parsl.config import Config as ParslConfig
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher, SrunLauncher
from parsl.providers import SlurmProvider
from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator
from tqdm import tqdm

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Execution
from climate_ref.slurm import HAS_REAL_SLURM, SlurmChecker
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import DiagnosticError
from climate_ref_core.executor import execute_locally

from .pbs_scheduler import SmartPBSProvider
from .result_handling import ExecutionFuture, mark_execution_failed, process_result

F = TypeVar("F", bound=Callable[..., Any])

T = TypeVar("T")


class SlurmConfig(BaseModel):
    """Slurm Configurations"""

    scheduler: Literal["slurm"]
    account: str
    username: str
    partition: str | None = None
    log_dir: str = "runinfo"
    qos: str | None = None
    req_nodes: Annotated[int, Field(strict=True, ge=1, le=1000)] = 1
    cores_per_worker: Annotated[int, Field(strict=True, ge=1, le=1000)] = 1
    mem_per_worker: Annotated[float, Field(strict=True, gt=0, lt=1000.0)] | None = None
    max_workers_per_node: Annotated[int, Field(strict=True, ge=1, le=1000)] = 16
    validation: StrictBool = False
    walltime: str = "00:30:00"
    scheduler_options: str = ""
    retries: Annotated[int, Field(strict=True, ge=0, le=3)] = 2
    max_blocks: Annotated[int, Field(strict=True, ge=1)] = 1  # one block mean one job?
    worker_init: str = ""
    overrides: str = ""
    cmd_timeout: Annotated[int, Field(strict=True, ge=0)] = 120
    cpu_affinity: str = "none"

    @model_validator(mode="before")
    def _check_parition_qos(cls, data: Any) -> Any:
        if not ("partition" in data or "qos" in data):
            raise ValueError("partition or qos is needed")
        return data

    @field_validator("scheduler_options")
    def _validate_sbatch_syntax(cls, v: str | None) -> Any:
        if not v:
            return v

        sbatch_pattern = re.compile(
            r"^\s*#SBATCH\s+"  # Start with #SBATCH
            r"(?:-\w+\s+[^\s]+"  # Option-value pairs
            r"(?:\s+-\w+\s+[^\s]+)*)"  # Additional options
            r"\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        invalid_lines = [
            line
            for line in v.split("\n")
            if not (line.strip().upper().startswith("#SBATCH") and sbatch_pattern.match(line.strip()))
        ]

        if invalid_lines:
            error_msg = (
                "Invalid SBATCH directives:\n"
                + "\n".join(invalid_lines)
                + "\n"
                + "Expected format: '#SBATCH -option value [-option value ...]'"
            )
            raise ValueError(error_msg)
        return v

    @field_validator("walltime")
    def _validate_walltime(cls, v: str) -> str:
        pattern = r"^(\d+-)?\d{1,5}:[0-5][0-9]:[0-5][0-9]$"
        if not re.match(pattern, v):
            raise ValueError("Walltime must be in `D-HH:MM:SS/HH:MM:SS` format")
        return v


def with_memory_limit(limit_gb: float | Callable[..., float | None]) -> Callable[[F], F]:
    """Set memory limit for a parsl worker"""

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                current_limit = limit_gb(*args, **kwargs) if callable(limit_gb) else limit_gb
            except Exception:
                current_limit = None

            if current_limit is not None and current_limit > 0:
                bytes_limit = int(current_limit * 1024 * 1024 * 1024)
                _, hard0 = resource.getrlimit(resource.RLIMIT_AS)
                soft = min(bytes_limit, hard0) if hard0 > 0 else bytes_limit
                resource.setrlimit(resource.RLIMIT_AS, (soft, hard0))
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def limit_from_env(*args: Any, **kwargs: Any) -> float | None:
    """Get the memory limits from env variables"""
    val = os.getenv("MEMORY_LIMIT_PARSL_JOB_GB")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


@python_app
@with_memory_limit(limit_from_env)
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

        self.extra_slurm_provider: dict[str, Any] = cast(
            dict[str, Any], executor_config.get("extra_slurm_provider") or {}
        )
        self.extra_executor: dict[str, Any] = cast(
            dict[str, Any], executor_config.get("extra_executor") or {}
        )

        self.cores_per_worker = _to_int(executor_config.get("cores_per_worker"))
        self.mem_per_worker = _to_float(executor_config.get("mem_per_worker"))

        self.parsl_provider_base = {
            "account",
            "partition",
            "qos",
            "nodes_per_block",
            "max_blocks",
            "worker_init",
            "walltime",
            "cmd_timeout",
        }

        self.parsl_executor_base = {
            "label",
            "cores_per_worker",
            "mem_per_worker",
            "max_workers_per_node",
            "cpu_affinity",
        }

        self._validate_extras()

        if self.scheduler == "slurm":
            self.slurm_config = SlurmConfig.model_validate(executor_config)
            hours, minutes, seconds = map(int, self.slurm_config.walltime.split(":"))

            if self.slurm_config.validation and HAS_REAL_SLURM:
                self._validate_slurm_params()
        else:
            hours, minutes, seconds = map(int, self.walltime.split(":"))

        total_minutes = hours * 60 + minutes + seconds / 60
        self.total_minutes = total_minutes

        self._initialize_parsl()

        self.parsl_results: list[ExecutionFuture] = []

    def _validate_extras(self) -> None:

        self._provider_extra = {}
        self._executor_extra = {}

        if self.scheduler == "slurm" and self.extra_slurm_provider:
            self._provider_extra = type(self)._filter_extras(
                self.extra_slurm_provider,
                list(self.parsl_provider_base),
                SlurmProvider,
                "SlurmProvider",
            )

        if self.extra_executor:
            self._executor_extra = type(self)._filter_extras(
                self.extra_executor,
                list(self.parsl_executor_base),
                HighThroughputExecutor,
                "HighThroughputExecutor",
            )

    @staticmethod
    def _filter_extras(
        extras: dict[str, Any], used_keys: list[str], cls: type[T], name: str = ""
    ) -> dict[str, Any]:

        # drop duplicates
        duplicates = used_keys & extras.keys()
        if duplicates:
            logger.info(f"Ignoring duplicate {name} params: {sorted(duplicates)}")

        filtered = {k: v for k, v in extras.items() if k not in used_keys}

        # validate keys
        sig = inspect.signature(cls.__init__)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        if not has_var_kw:
            valid = {k for k in sig.parameters if k != "self"}
            invalid = set(filtered) - valid
            if invalid:
                logger.warning(f"Warning: Removing invalid {name} params: {sorted(invalid)}")
                filtered = {k: v for k, v in filtered.items() if k in valid}

        return filtered

    def _validate_slurm_params(self) -> None:
        """Validate the Slurm configuration using SlurmChecker.

        Raises
        ------
            ValueError: If account, partition or QOS are invalid or inaccessible.
        """
        slurm_checker = SlurmChecker()
        if self.slurm_config.account and not slurm_checker.get_account_info(self.slurm_config.account):
            raise ValueError(f"Account: {self.slurm_config.account} not valid")

        partition_limits = None
        node_info = None

        if self.slurm_config.partition:
            if not slurm_checker.get_partition_info(self.slurm_config.partition):
                raise ValueError(f"Partition: {self.slurm_config.partition} not valid")

            if not slurm_checker.can_account_use_partition(
                self.slurm_config.account, self.slurm_config.partition
            ):
                raise ValueError(
                    f"Account: {self.slurm_config.account}"
                    f" cannot access partiton: {self.slurm_config.partition}"
                )

            partition_limits = slurm_checker.get_partition_limits(self.slurm_config.partition)
            node_info = slurm_checker.get_node_from_partition(self.slurm_config.partition)

        qos_limits = None
        if self.slurm_config.qos:
            if not slurm_checker.get_qos_info(self.slurm_config.qos):
                raise ValueError(f"QOS: {self.slurm_config.qos} not valid")

            if not slurm_checker.can_account_use_qos(self.slurm_config.account, self.slurm_config.qos):
                raise ValueError(
                    f"Account: {self.slurm_config.account} cannot access qos: {self.slurm_config.qos}"
                )

            qos_limits = slurm_checker.get_qos_limits(self.slurm_config.qos)

        max_cores_per_node = int(node_info["cpus"]) if node_info else None
        if max_cores_per_node and self.slurm_config.cores_per_worker:
            if self.slurm_config.cores_per_worker > max_cores_per_node:
                raise ValueError(
                    f"cores_per_work:{self.slurm_config.cores_per_worker}"
                    f"larger than the maximum in a node {max_cores_per_node}"
                )

        max_mem_per_node = float(node_info["real_memory"]) if node_info else None
        if max_mem_per_node and self.slurm_config.mem_per_worker:
            if self.slurm_config.mem_per_worker > max_mem_per_node:
                raise ValueError(
                    f"mem_per_work:{self.slurm_config.mem_per_worker}"
                    f"larger than the maximum mem in a node {max_mem_per_node}"
                )

        max_walltime_partition = (
            partition_limits["max_time_minutes"] if partition_limits else self.total_minutes
        )
        max_walltime_qos = qos_limits["max_time_minutes"] if qos_limits else self.total_minutes

        max_walltime_minutes = min(float(max_walltime_partition), float(max_walltime_qos))

        if self.total_minutes > float(max_walltime_minutes):
            raise ValueError(
                f"Walltime: {self.slurm_config.walltime} exceed the maximum time "
                f"{max_walltime_minutes} allowed by {self.slurm_config.partition} and {self.slurm_config.qos}"
            )

    def _initialize_parsl(self) -> None:
        executor_config = self.config.executor.config

        provider: SlurmProvider | SmartPBSProvider
        if self.scheduler == "slurm":
            provider = SlurmProvider(
                account=self.slurm_config.account,
                partition=self.slurm_config.partition,
                qos=self.slurm_config.qos,
                nodes_per_block=self.slurm_config.req_nodes,
                max_blocks=self.slurm_config.max_blocks,
                scheduler_options=self.slurm_config.scheduler_options,
                worker_init=self.slurm_config.worker_init,
                launcher=SrunLauncher(
                    debug=True,
                    overrides=self.slurm_config.overrides,
                ),
                walltime=self.slurm_config.walltime,
                cmd_timeout=self.slurm_config.cmd_timeout,
                **self._provider_extra or {},
            )

            executor = HighThroughputExecutor(
                label="ref_hpc_executor",
                cores_per_worker=self.slurm_config.cores_per_worker,
                mem_per_worker=self.slurm_config.mem_per_worker,
                max_workers_per_node=self.slurm_config.max_workers_per_node,
                cpu_affinity=self.slurm_config.cpu_affinity,
                provider=provider,
                **self._executor_extra or {},
            )

            hpc_config = ParslConfig(
                run_dir=self.slurm_config.log_dir,
                executors=[executor],
                retries=self.slurm_config.retries,
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

            executor = HighThroughputExecutor(
                label="ref_hpc_executor",
                cores_per_worker=self.cores_per_worker if self.cores_per_worker else 1,
                mem_per_worker=self.mem_per_worker,
                max_workers_per_node=_to_int(executor_config.get("max_workers_per_node", 16)),
                cpu_affinity=str(executor_config.get("cpu_affinity")),
                provider=provider,
                **self._executor_extra or {},
            )

            hpc_config = ParslConfig(
                run_dir=self.log_dir,
                executors=[executor],
                retries=int(executor_config.get("retries", 2)),
            )

        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")

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

    def join(self, timeout: float) -> None:  # noqa: PLR0912
        """
        Wait for all diagnostics to finish

        This will block until all diagnostics have completed or the timeout is reached.

        The effective timeout is the smaller of the caller-provided ``timeout`` and
        the configured Slurm/PBS walltime. When it is reached, any outstanding
        executions are marked as failed (retryable) so the next solve can pick
        them up rather than leaving them stuck with ``successful=None``.

        Parameters
        ----------
        timeout
            Timeout in seconds.
            Positive values are capped by the configured Slurm/PBS walltime.
            ``0`` or negative values disable the caller-provided timeout,
            but this method still waits at most until the configured walltime if one is set.

        Raises
        ------
        TimeoutError
            If the timeout is reached before all executions complete
        """
        start_time = time.time()
        refresh_time = 0.5

        walltime_seconds = self.total_minutes * 60
        if timeout and timeout > 0:
            effective_timeout = min(timeout, walltime_seconds) if walltime_seconds else timeout
        else:
            effective_timeout = walltime_seconds

        results = self.parsl_results
        t = tqdm(total=len(results), desc="Waiting for executions to complete", unit="execution")

        try:
            while results:
                # Iterate over a copy of the list and remove finished tasks
                for result in results[:]:
                    if not result.future.done():
                        continue

                    err = result.future.exception()
                    execution_result: ExecutionResult | None
                    if err is None:
                        try:
                            execution_result = result.future.result(timeout=0)
                        except Exception:
                            # Result retrieval failed even though the future
                            # reported done; treat as retryable system failure.
                            logger.exception(
                                f"Failed to retrieve result for {result.definition.execution_slug()!r}"
                            )
                            execution_result = ExecutionResult.build_from_failure(
                                result.definition, retryable=True
                            )
                    elif isinstance(err, DiagnosticError):
                        execution_result = err.result
                    else:
                        # Walltime kill, ExecutionLost, OOM, segfault, etc.
                        # Mark retryable so the next solve picks it up.
                        logger.error(
                            f"System-level failure for {result.definition.execution_slug()!r}: {err!r}"
                        )
                        execution_result = ExecutionResult.build_from_failure(
                            result.definition, retryable=True
                        )

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

                if effective_timeout and elapsed_time > effective_timeout:
                    self._fail_outstanding(results, t)
                    raise TimeoutError(f"Not all HPC executions completed within {effective_timeout}s")

                # Wait for a short time before checking for completed executions
                time.sleep(refresh_time)
        finally:
            t.close()
            if parsl.dfk():
                parsl.dfk().cleanup()

    def _fail_outstanding(self, results: list[ExecutionFuture], progress: Any) -> None:
        """Mark every outstanding execution as failed-retryable before raising."""
        for outstanding in list(results):
            logger.warning(
                f"HPC execution {outstanding.definition.execution_slug()!r} did not complete in time"
            )
            mark_execution_failed(
                self.database,
                self.config,
                outstanding.definition,
                outstanding.execution_id,
                retryable=True,
            )
            progress.update(n=1)
            results.remove(outstanding)
