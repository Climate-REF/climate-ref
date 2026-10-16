import contextlib
import logging as std_logging

import pytest
from loguru import logger

from climate_ref_core.diagnostics import ExecutionDefinition
from climate_ref_core.logging import (
    DEFAULT_LOG_FORMAT,
    EXECUTION_LOG_FILENAME,
    _retention_count_ignore_missing,
    capture_logging,
    execution_log_only,
    redirect_logs,
)


@pytest.fixture
def definition(mocker, tmp_path):
    d = mocker.MagicMock(spec=ExecutionDefinition)
    d.output_directory = tmp_path / "output"

    return d, d.output_directory / EXECUTION_LOG_FILENAME


def test_intercept_logs(caplog):
    std_logger = std_logging.getLogger("climate_ref_core")

    capture_logging()

    with caplog.at_level(std_logging.INFO):
        std_logger.error("this is an error log")
        std_logger.debug("this is an debug log")

    assert "this is an error log" in caplog.text
    assert "this is an debug log" not in caplog.text


def test_redirect_logs(definition, caplog):
    log_level = "DEBUG"
    text_outer = "Outer log message"
    text_inner = "Inner log message"
    text_post = "Post log message"

    definition, output_file = definition

    with caplog.at_level(log_level):
        logger.info(text_outer)

        with redirect_logs(definition, log_level):
            logger.info(text_inner)

        logger.info(text_post)

    assert text_outer in caplog.text
    assert text_inner not in caplog.text
    assert text_post in caplog.text

    assert text_inner in output_file.read_text()


def test_redirect_logs_keeps_warnings_on_console(definition, caplog):
    definition, output_file = definition

    with caplog.at_level("DEBUG"):
        with pytest.raises(ValueError):
            with redirect_logs(definition, "DEBUG"):
                logger.warning("Inner warning")
                raise ValueError("boom")

    assert "Inner warning" in caplog.text
    # The executor logs the failure to the console, so the traceback is only written once
    assert "Execution failed" not in caplog.text
    assert "Execution failed" in output_file.read_text()


def test_execution_log_only_skips_console(definition, caplog):
    definition, output_file = definition

    with caplog.at_level("DEBUG"):
        with redirect_logs(definition, "WARNING"):
            execution_log_only().error("bulky command output")

    assert "bulky command output" not in caplog.text
    assert "bulky command output" in output_file.read_text()


def test_redirect_logs_restores_handler_if_setup_fails(definition, caplog, tmp_path):
    definition, _ = definition
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    definition.output_directory = blocker / "output"

    with pytest.raises(OSError):
        with redirect_logs(definition, "INFO"):
            pass

    with caplog.at_level("INFO"):
        logger.info("after failed setup")
    assert "after failed setup" in caplog.text


def test_default_format_names_execution():
    capture_logging()
    lines = []
    handler_id = logger.add(lines.append, format=DEFAULT_LOG_FORMAT, colorize=False)
    try:
        with logger.contextualize(execution="pmp/annual-cycle/cmip6_tas"):
            logger.warning("inside")
        logger.warning("outside")
    finally:
        logger.remove(handler_id)

    assert lines[0].endswith(" - [pmp/annual-cycle/cmip6_tas] inside\n")
    assert lines[1].endswith(" - outside\n")


def test_redirect_logs_without_handler(definition):
    definition, output_file = definition

    assert not hasattr(logger, "default_handler_id")
    with redirect_logs(definition, "INFO"):
        logger.info("inner")
    assert not hasattr(logger, "default_handler_id")

    assert "inner" in output_file.read_text()


def test_redirect_logs_stdlog_captured(definition):
    definition, output_file = definition

    with redirect_logs(definition, "INFO"):
        stdlog = std_logging.getLogger("stdlog")
        stdlog.info("stdlog inner")
        stdlog.debug("stdlog debug")

    assert "stdlog inner" in output_file.read_text()
    assert "stdlog debug" not in output_file.read_text()


def test_retention_count_ignores_missing_files(tmp_path):
    """Log retention cleanup should tolerate files removed by another process."""
    logs = []
    for idx in range(12):
        log = tmp_path / f"log-{idx}.log"
        log.write_text(str(idx))
        logs.append(str(log))
    missing = tmp_path / "already-removed.log"
    logs.append(str(missing))

    _retention_count_ignore_missing(logs)

    assert len(list(tmp_path.glob("*.log"))) == 10


def test_redirect_logs_exception(definition, caplog):
    log_level = "DEBUG"

    orig_handler = logger.default_handler_id  # type: ignore

    with pytest.raises(ValueError):
        with redirect_logs(definition[0], log_level):
            logger.debug("This will raise an exception")
            raise ValueError()

    assert orig_handler != logger.default_handler_id, (  # type: ignore
        "The default handler should have changed during the redirect and cleanup."
    )


@pytest.mark.parametrize("fails", [False, True])
def test_redirect_logs_completion_message(definition, fails):
    definition, output_file = definition

    with contextlib.suppress(ValueError):
        with redirect_logs(definition, "INFO"):
            if fails:
                raise ValueError("boom")

    text = output_file.read_text()
    assert ("Diagnostic execution failed" in text) is fails
    assert ("Diagnostic execution complete" in text) is not fails
