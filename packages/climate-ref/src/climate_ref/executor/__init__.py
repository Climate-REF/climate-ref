"""
Execute diagnostics in different environments

We support running diagnostics in different environments, such as locally,
in a separate process, or in a container.
These environments are represented by `climate_ref.executor.Executor` classes.

The simplest executor is the `LocalExecutor`, which runs the diagnostic in the same process.
This is useful for local testing and debugging.
"""

from climate_ref_core.exceptions import InvalidExecutorException

try:
    from .hpc import HPCExecutor
except InvalidExecutorException as exc:
    # This exception is reraised when importing the executor as `climate_ref.executors.HPCExecutor`
    HPCExecutor = exc  # type: ignore

from .local import LocalExecutor
from .reingest import ReingestMode, reingest_execution
from .result_handling import handle_execution_result
from .synchronous import SynchronousExecutor

__all__ = [
    "HPCExecutor",
    "LocalExecutor",
    "ReingestMode",
    "SynchronousExecutor",
    "handle_execution_result",
    "reingest_execution",
]
