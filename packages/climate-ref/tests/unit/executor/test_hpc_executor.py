import os
import re
import resource
from unittest.mock import MagicMock, patch

import parsl
import pytest
from parsl.dataflow import futures
from pydantic import ValidationError

from climate_ref.executor.hpc import (
    HPCExecutor,
    SlurmConfig,
    execute_locally,
    limit_from_env,
    with_memory_limit,
)
from climate_ref.executor.local import ExecutionFuture
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.exceptions import DiagnosticError
from climate_ref_core.executor import Executor


def test_execute_locally_failed(definition_factory, mock_diagnostic):
    mock_diagnostic.run = lambda definition: 1 / 0

    # execution raises an exception
    with pytest.raises(DiagnosticError):
        result = execute_locally(
            definition_factory(diagnostic=mock_diagnostic),
            log_level="DEBUG",
            raise_error=True,
        )

        assert result is None


class TestHPCExecutor:
    @pytest.fixture
    def base_config(self, tmp_path):
        """Shared config dictionary for all tests in this class."""
        return {
            "scheduler": "slurm",
            "account": "myaccount",
            "username": "myname",
            "qos": "myqos",
            "partition": "mypartition",
            "log_dir": str(tmp_path / "parsl_runinfo"),
        }

    def test_is_executor(self, base_config):
        executor = HPCExecutor(**base_config)

        assert executor.name == "hpc"
        assert isinstance(executor, Executor)
        parsl.dfk().cleanup()

    def test_run_metric(self, metric_definition, provider, mock_diagnostic, mocker, caplog, base_config):
        with patch.object(HPCExecutor, "run", autospec=True) as mock_run:
            # Configure the mock to behave similarly to the original
            mock_run.side_effect = lambda self, definition, execution=None: (
                self.parsl_results.append(
                    ExecutionFuture(
                        future=MagicMock(),  # Mock the future object
                        definition=definition,
                        execution_id=execution.id if execution else None,
                    )
                )
            )

            executor = HPCExecutor(**base_config)

            # shall have the SerializationError, but not raised
            executor.run(metric_definition, None)
            assert len(executor.parsl_results) == 1
            assert executor.parsl_results[0].definition == metric_definition
            assert executor.parsl_results[0].execution_id is None

        parsl.dfk().cleanup()

    def test_join(self, metric_definition, base_config):
        executor = HPCExecutor(**base_config)
        future = futures.AppFuture(1)
        executor.parsl_results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]

        future.set_result(
            ExecutionResult(
                definition=metric_definition,
                successful=False,
                output_bundle_filename=None,
                metric_bundle_filename=None,
            )
        )
        executor.join(0.1)

        assert len(executor.parsl_results) == 0

    def test_join_diagnostic_exception(self, metric_definition, base_config):
        executor = HPCExecutor(**base_config)
        future = futures.AppFuture(1)
        executor.parsl_results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]

        execution_result = ExecutionResult(
            definition=metric_definition,
            successful=False,
            output_bundle_filename=None,
            metric_bundle_filename=None,
        )

        future.set_exception(DiagnosticError("Some thing bad went wrong", execution_result))
        err_result = executor.parsl_results[0].future.exception().result

        executor.join(0.1)

        assert err_result == ExecutionResult(
            definition=metric_definition,
            successful=False,
            output_bundle_filename=None,
            metric_bundle_filename=None,
        )
        assert len(executor.parsl_results) == 0

    def test_join_other_exception(self, metric_definition, base_config):
        executor = HPCExecutor(**base_config)
        future = futures.AppFuture(1)
        executor.parsl_results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]

        future.set_exception(ValueError("Some thing bad went wrong"))

        with pytest.raises(AssertionError, match=re.escape("Execution result should not be None")):
            executor.join(0.1)

    @pytest.mark.parametrize(
        "field_name, invalid_value",
        [
            ("scheduler", "pbs"),
            ("account", 1234),
            ("username", 0.001),
            ("log_dir", True),
            ("qos", 1234),
            ("req_nodes", 1001),
            ("cores_per_worker", 1001),
            ("mem_per_worker", -1),
            ("max_workers_per_node", 1001),
            ("validation", "true"),
            ("walltime", "3"),
            ("scheduler_options", 10),
            ("scheduler_options", "#SABTCH -C cpu\n$LLLL -C"),
            ("retries", "2"),
            ("max_blocks", "1"),
            ("worker_init", 1),
            ("overrides", 0),
            ("cmd_timeout", -1),
            ("cpu_affinity", 1),
        ],
    )
    def test_hpc_slurm_error_config(self, field_name, invalid_value):
        slurm_cfg_dict = {
            "scheduler": "slurm",
            "qos": "myqos",
            "account": "myaccount",
            "username": "myname",
            "req_nodes": 3,
            "scheduler_options": "#SBATCH -C cpu",
        }
        slurm_cfg_dict[field_name] = invalid_value
        with pytest.raises(ValidationError):
            SlurmConfig.model_validate(slurm_cfg_dict)

    @pytest.mark.parametrize(
        "missing_config",
        [
            ["scheduler"],
            ["account"],
            ["username"],
            ["partition", "qos"],
        ],
    )
    def test_hpc_slurm_missing_required_config(self, missing_config, base_config):
        slurm_cfg_dict = base_config
        [slurm_cfg_dict.pop(m) for m in missing_config]
        with pytest.raises(ValidationError):
            SlurmConfig.model_validate(slurm_cfg_dict)

    def test_memeory_limit_fixed(self):
        """Test with a fixed numeric memory limit"""

        @with_memory_limit(2.0)
        def test_func():
            # Check if memory limit was set
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            expected_bytes = int(2.0 * 1024 * 1024 * 1024)
            assert soft == expected_bytes
            assert hard >= expected_bytes

    def test_memory_limit_func(self):
        # Save original state
        os.environ.pop("MEMORY_LIMIT_PARSL_JOB_GB", None)
        orig_soft, orig_hard = resource.getrlimit(resource.RLIMIT_AS)

        @with_memory_limit(limit_from_env)
        def unset_func():
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            assert soft == orig_soft
            assert hard == orig_hard

        unset_func()
        os.environ["MEMORY_LIMIT_PARSL_JOB_GB"] = "7"

        @with_memory_limit(limit_from_env)
        def set_func():
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            expected_bytes = 7 * 1024 * 1024 * 1024
            assert soft == expected_bytes
            assert hard == -1 or hard >= expected_bytes

        set_func()
