"""
Execute diagnostics in different environments

We support running diagnostics in different environments, such as locally,
in a separate process, or in a container.
These environments are represented by `climate_ref.executor.Executor` classes.

The simplest executor is the `LocalExecutor`, which runs the diagnostic in the same process.
This is useful for local testing and debugging.
"""

from .hpc import HPCExecutor
from .local import LocalExecutor
from .result_handling import handle_execution_result
from .synchronous import SynchronousExecutor

__all__ = ["HPCExecutor", "LocalExecutor", "SynchronousExecutor", "handle_execution_result"]
