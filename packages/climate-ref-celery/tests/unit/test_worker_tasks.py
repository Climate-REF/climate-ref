import pytest
from climate_ref_celery import worker_tasks
from climate_ref_celery.worker_tasks import _worker_database, handle_failure, handle_result
from climate_ref_example import provider as example_provider

from climate_ref.database import Database
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.provider_registry import _register_provider


@pytest.fixture(autouse=True)
def _clear_worker_database():
    worker_tasks._worker_database.close()
    yield
    worker_tasks._worker_database.close()


def test_worker_database_is_reused_across_tasks(config):
    first = _worker_database.get(config)
    second = _worker_database.get(config)

    assert second is first


def test_worker_database_reopens_for_a_different_url(config, tmp_path):
    first = _worker_database.get(config)

    config.db.database_url = f"sqlite:///{tmp_path}/other.db"
    second = _worker_database.get(config)

    assert second is not first
    assert second.url == config.db.database_url


def test_worker_task(mocker, config):
    mock_handle_result = mocker.patch("climate_ref_celery.worker_tasks.handle_execution_result")
    db = Database.from_config(config, run_migrations=True)
    with db.session.begin():
        result = mocker.Mock()

        _register_provider(db, example_provider)
        execution_group = ExecutionGroup(
            diagnostic_id=1,
            key="key",
            dirty=True,
        )
        db.session.add(execution_group)

        metric_execution_result = Execution(
            output_fragment="output_fragment",
            dataset_hash="hash",
            execution_group=execution_group,
        )
        db.session.add(metric_execution_result)

    handle_result(result, metric_execution_result.id)

    mock_handle_result.assert_called_once()


def test_worker_task_missing(mocker, config):
    result = mocker.Mock()
    Database.from_config(config, run_migrations=True)

    assert handle_result(result, 1) is None


def test_handle_failure_marks_execution_failed(config):
    db = Database.from_config(config, run_migrations=True)

    with db.session.begin():
        _register_provider(db, example_provider)
        execution_group = ExecutionGroup(
            diagnostic_id=1,
            key="failure-test-key",
            dirty=True,
        )
        db.session.add(execution_group)

        execution = Execution(
            output_fragment="output_fragment",
            dataset_hash="hash",
            execution_group=execution_group,
        )
        db.session.add(execution)

    # Verify execution starts without a success/failure status
    assert execution.successful is None

    # Simulate a failed task callback (this opens its own DB connection)
    handle_failure("fake-task-uuid-1234", execution_id=execution.id)

    # Expire cached state so the next read hits the DB and sees the
    # changes committed by handle_failure's separate connection.
    db.session.expire_all()

    execution = db.session.get(Execution, execution.id)
    assert execution.successful is False

    # System-level failures (worker crash, OOM, time limit) should leave
    # dirty=True so the execution is retried on the next solve
    assert execution.execution_group.dirty is True


def test_handle_failure_missing_execution(config):
    Database.from_config(config, run_migrations=True)

    # Should not raise - just logs and returns
    assert handle_failure("fake-task-uuid-1234", execution_id=9999) is None
