from typing import Any

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.executor.local import process_result
from climate_ref.models import Execution
from climate_ref_core.diagnostics import ExecutionDefinition
from climate_ref_core.executor import execute_locally


class SynchronousExecutor:
    """
    Run a diagnostic synchronously, in-process.

    This is mainly useful for debugging and testing.
    [climate_ref.executor.LocalExecutor][] is a more general purpose executor.
    """

    name = "synchronous"

    def __init__(
        self, *, database: Database | None = None, config: Config | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            config = Config.default()
        if database is None:
            database = Database.from_config(config, run_migrations=False)

        self.database = database
        self.config = config

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
        result = execute_locally(definition, log_level=self.config.log_level)
        process_result(self.config, self.database, result, execution)

    def join(self, timeout: float) -> None:
        """
        Wait for all diagnostics to finish

        This returns immediately because the executor runs diagnostics synchronously.

        Parameters
        ----------
        timeout
            Timeout in seconds (Not used)
        """
        pass
