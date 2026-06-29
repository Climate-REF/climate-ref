import datetime

import pytest

from climate_ref.models.execution import Execution, ExecutionGroup, ExecutionOutput, ResultOutputType


class TestMetricExecution:
    def test_should_run_no_results(self, mocker):
        execution = mocker.Mock(spec=ExecutionGroup)
        execution.executions = []

        assert ExecutionGroup.should_run(execution, "dataset_hash")

    def test_should_run_invalid_hash(self, mocker):
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash_old"
        execution.executions = [execution_result]

        assert ExecutionGroup.should_run(execution, "dataset_hash")

    def test_should_run_dirty(self, mocker):
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = True
        execution.executions = [execution_result]
        execution.dirty = True

        assert ExecutionGroup.should_run(execution, "dataset_hash")

    def test_shouldnt_run(self, mocker):
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = True
        execution.executions = [execution_result]
        execution.dirty = False

        assert not ExecutionGroup.should_run(execution, "dataset_hash")

    def test_shouldnt_run_already_in_progress(self, mocker):
        """An in-progress execution with the same hash should not trigger a duplicate"""
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = None
        execution.executions = [execution_result]
        execution.dirty = True

        assert not ExecutionGroup.should_run(execution, "dataset_hash")

    def test_shouldnt_run_failed_not_dirty(self, mocker):
        """A failed execution with dirty=False should not be retried by default"""
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = False
        execution.executions = [execution_result]
        execution.dirty = False

        assert not ExecutionGroup.should_run(execution, "dataset_hash")

    def test_should_run_failed_dirty(self, mocker):
        """A failed execution with dirty=True should be retried (e.g. after flag-dirty or fail-running)"""
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = False
        execution.executions = [execution_result]
        execution.dirty = True

        assert ExecutionGroup.should_run(execution, "dataset_hash")

    def test_should_run_failed_different_hash(self, mocker):
        """A failed execution with a different hash should trigger a new run"""
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "old_hash"
        execution_result.successful = False
        execution.executions = [execution_result]
        execution.dirty = False

        assert ExecutionGroup.should_run(execution, "new_hash")

    def test_should_run_failed_rerun_flag(self, mocker):
        """A failed execution should re-run when rerun_failed=True, even if not dirty"""
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = False
        execution.executions = [execution_result]
        execution.dirty = False

        assert ExecutionGroup.should_run(execution, "dataset_hash", rerun_failed=True)

    def test_should_run_stale_in_progress_dirty(self, mocker):
        """A stale in-progress execution is treated as failed under a dry-run cutoff.

        A real solve reaps it before evaluating should_run, so a dry-run preview must
        report the dirty group as runnable rather than hiding it as in-progress.
        """
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = None
        execution_result.created_at = datetime.datetime(2020, 1, 1)
        execution.executions = [execution_result]
        execution.dirty = True

        cutoff = datetime.datetime(2020, 6, 1)

        # Without a cutoff the in-progress execution suppresses the run...
        assert not ExecutionGroup.should_run(execution, "dataset_hash")
        # ...but with a cutoff the stale execution is treated as reaped.
        assert ExecutionGroup.should_run(execution, "dataset_hash", stale_cutoff=cutoff)

    def test_should_run_stale_in_progress_rerun_flag(self, mocker):
        """A stale in-progress execution re-runs under rerun_failed even if not dirty"""
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = None
        execution_result.created_at = datetime.datetime(2020, 1, 1)
        execution.executions = [execution_result]
        execution.dirty = False

        cutoff = datetime.datetime(2020, 6, 1)

        assert not ExecutionGroup.should_run(execution, "dataset_hash", stale_cutoff=cutoff)
        assert ExecutionGroup.should_run(execution, "dataset_hash", rerun_failed=True, stale_cutoff=cutoff)

    def test_should_run_fresh_in_progress_with_cutoff(self, mocker):
        """An in-progress execution newer than the cutoff is still left alone"""
        execution = mocker.Mock(spec=ExecutionGroup)
        execution_result = mocker.Mock(spec=Execution)

        execution_result.dataset_hash = "dataset_hash"
        execution_result.successful = None
        execution_result.created_at = datetime.datetime(2020, 12, 1)
        execution.executions = [execution_result]
        execution.dirty = True

        cutoff = datetime.datetime(2020, 6, 1)

        assert not ExecutionGroup.should_run(execution, "dataset_hash", stale_cutoff=cutoff)


class TestExecutionOutput:
    @pytest.mark.parametrize(
        "attributes",
        (
            {
                "filename": "test.png",
                "short_name": "test",
                "long_name": "Test Plot",
                "description": "A test plot",
            },
            None,
        ),
    )
    def test_build(self, db_seeded, attributes):
        if attributes is None:
            attributes = {}
        item_orig = ExecutionOutput.build(
            execution_id=1,
            output_type=ResultOutputType.Plot,
            dimensions={"source_id": "test"},
            **attributes,
        )
        db_seeded.session.add(item_orig)
        db_seeded.session.commit()

        item = db_seeded.session.get(ExecutionOutput, item_orig.id)
        for key, value in attributes.items():
            assert getattr(item, key) == value

        assert item.dimensions == {"source_id": "test"}

    def test_invalid_dimension(self, db_seeded):
        exp_msg = "Unknown dimension column 'not_a_dimension'"
        with pytest.raises(KeyError, match=exp_msg):
            ExecutionOutput.build(
                execution_id=1,
                output_type=ResultOutputType.Plot,
                dimensions={"not_a_dimension": "test"},
            )

    def test_register_dimensions(self, cmip7_aft_cv):
        ExecutionOutput._reset_cv_dimensions()
        assert ExecutionOutput._cv_dimensions == []

        with pytest.raises(KeyError):
            ExecutionOutput.build(
                execution_id=1,
                output_type=ResultOutputType.Plot,
                dimensions={"source_id": "test"},
            )

        ExecutionOutput.register_cv_dimensions(cmip7_aft_cv)
        assert ExecutionOutput._cv_dimensions == [d.name for d in cmip7_aft_cv.dimensions]

        # Should work now that the dimension has been registered
        item = ExecutionOutput.build(
            execution_id=1,
            output_type=ResultOutputType.Plot,
            dimensions={"source_id": "test"},
        )

        for k in ExecutionOutput._cv_dimensions:
            assert hasattr(item, k)

    def test_register_dimensions_multiple_times(self, cmip7_aft_cv):
        ExecutionOutput._reset_cv_dimensions()
        assert ExecutionOutput._cv_dimensions == []

        ExecutionOutput.register_cv_dimensions(cmip7_aft_cv)
        assert ExecutionOutput._cv_dimensions == [d.name for d in cmip7_aft_cv.dimensions]

        ExecutionOutput.register_cv_dimensions(cmip7_aft_cv)
        assert ExecutionOutput._cv_dimensions == [d.name for d in cmip7_aft_cv.dimensions]
