import io
import time

from climate_ref.progress import (
    DEFAULT_PROGRESS_LOG_INTERVAL,
    PROGRESS_LOG_INTERVAL_ENV,
    LoggedProgress,
    progress_log_interval,
)


def test_interval_from_env(monkeypatch):
    monkeypatch.delenv(PROGRESS_LOG_INTERVAL_ENV, raising=False)
    assert progress_log_interval() == DEFAULT_PROGRESS_LOG_INTERVAL

    monkeypatch.setenv(PROGRESS_LOG_INTERVAL_ENV, "60")
    assert progress_log_interval() == 60.0


def test_counts_updates_when_hidden():
    bar = LoggedProgress(total=4, desc="Waiting", unit="execution", disable=True, log_interval=0)
    bar.update(n=1)
    bar.update(n=2)

    message = bar.progress_message()
    bar.close()

    assert message.startswith("Waiting: 3/4 executions done (75%) after 0:00:00")
    assert "left" in message


def test_counts_iteration_when_hidden():
    bar = LoggedProgress(["a", "b"], desc="Parsing files", unit="file", disable=True, log_interval=0)

    assert list(bar) == ["a", "b"]
    assert bar.progress_message().startswith("Parsing files: 2/2 files done (100%)")


def test_counts_iteration_when_shown():
    bar = LoggedProgress(
        ["a", "b", "c"], desc="Parsing files", unit="file", disable=False, file=io.StringIO(), log_interval=0
    )

    assert list(bar) == ["a", "b", "c"]
    assert bar.progress_message().startswith("Parsing files: 3/3 files done (100%)")


def test_unknown_total():
    bar = LoggedProgress(desc="Parsing", unit="file", disable=True, log_interval=0)
    bar.update()

    assert bar.progress_message() == "Parsing: 1 files done after 0:00:00"
    bar.close()


def test_logs_periodically_until_closed(caplog):
    bar = LoggedProgress(total=2, desc="Waiting", unit="execution", disable=True, log_interval=0.01)
    try:
        deadline = time.monotonic() + 5
        while not caplog.messages and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        bar.close()

    assert any(m.startswith("Waiting: 0/2 executions done") for m in caplog.messages)
