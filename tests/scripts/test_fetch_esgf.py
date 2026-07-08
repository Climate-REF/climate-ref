"""
Tests for the failure handling in `scripts/fetch-esgf.py`.

The script is not an importable module, so it is loaded by path.
No network access is performed: every request's `fetch` is stubbed.
"""

import importlib.util
from pathlib import Path

import pytest
import typer
from intake_esgf.exceptions import DatasetLoadError, NoSearchResults, StalledDownload
from typer.testing import CliRunner

SCRIPT = Path(__file__).parents[2] / "scripts" / "fetch-esgf.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_esgf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture
def no_sleep(script, monkeypatch):
    """Record the retry backoff without actually waiting."""
    delays: list[float] = []
    monkeypatch.setattr(script.time, "sleep", delays.append)
    return delays


def _request(script, request_id="req"):
    return script.Obs4MIPsRequest(id=request_id, facets={"source_id": "X"})


def _stub_fetch(script, monkeypatch, behaviours):
    """
    Replace `Obs4MIPsRequest.fetch` with a stub driven by `behaviours`.

    Each behaviour is either an exception to raise or a value to return.
    The list is consumed one entry per call, and the number of calls is recorded.
    """
    remaining = list(behaviours)
    calls: list[int] = []

    def fake_fetch(self, remove_ensembles=True):
        calls.append(1)
        behaviour = remaining.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour

    monkeypatch.setattr(script.Obs4MIPsRequest, "fetch", fake_fetch)
    return calls


class TestRunRequest:
    def test_success(self, script, monkeypatch, no_sleep):
        _stub_fetch(script, monkeypatch, [{"a": [Path("a.nc")], "b": [Path("b.nc")]}])

        result = script.run_request(_request(script))

        assert result.outcome is script.Outcome.OK
        assert result.dataset_count == 2
        assert result.error is None

    def test_no_search_results_is_not_a_failure(self, script, monkeypatch, no_sleep):
        calls = _stub_fetch(script, monkeypatch, [NoSearchResults()])

        result = script.run_request(_request(script))

        assert result.outcome is script.Outcome.EMPTY
        assert result.dataset_count == 0
        # An empty search is never retried: retrying cannot change the answer.
        assert len(calls) == 1
        assert no_sleep == []

    def test_transient_error_is_retried_then_succeeds(self, script, monkeypatch, no_sleep):
        # This is the observed ORNL THREDDS 504: DatasetLoadError, then a clean retry.
        calls = _stub_fetch(script, monkeypatch, [DatasetLoadError(["k"]), {"a": [Path("a.nc")]}])

        result = script.run_request(_request(script), retry_delay=5.0)

        assert result.outcome is script.Outcome.OK
        assert result.dataset_count == 1
        assert len(calls) == 2
        assert no_sleep == [5.0]

    def test_transient_error_exhausts_attempts_and_fails(self, script, monkeypatch, no_sleep):
        calls = _stub_fetch(script, monkeypatch, [StalledDownload() for _ in range(3)])

        result = script.run_request(_request(script), max_attempts=3, retry_delay=1.0)

        assert result.outcome is script.Outcome.FAILED
        assert "StalledDownload" in (result.error or "")
        assert len(calls) == 3
        # Exponential backoff between attempts.
        assert no_sleep == [1.0, 2.0]

    def test_non_transient_error_is_not_retried(self, script, monkeypatch, no_sleep):
        calls = _stub_fetch(script, monkeypatch, [RuntimeError("boom")])

        result = script.run_request(_request(script))

        assert result.outcome is script.Outcome.FAILED
        assert "RuntimeError: boom" in (result.error or "")
        assert len(calls) == 1
        assert no_sleep == []


class TestExitCode:
    """The regression this guards: a failed fetch must not exit 0."""

    @staticmethod
    def _invoke(script, monkeypatch, behaviours):
        _stub_fetch(script, monkeypatch, behaviours)
        monkeypatch.setattr(script, "requests", [_request(script, "only")])

        app = typer.Typer()
        app.command()(script.main)
        return CliRunner().invoke(app, ["--max-attempts", "1"])

    def test_failure_exits_non_zero(self, script, monkeypatch, no_sleep):
        result = self._invoke(script, monkeypatch, [DatasetLoadError(["k"])])

        assert result.exit_code == 1

    def test_success_exits_zero(self, script, monkeypatch, no_sleep):
        result = self._invoke(script, monkeypatch, [{"a": [Path("a.nc")]}])

        assert result.exit_code == 0

    def test_empty_search_exits_zero(self, script, monkeypatch, no_sleep):
        # A request that legitimately matches nothing must not fail the run.
        result = self._invoke(script, monkeypatch, [NoSearchResults()])

        assert result.exit_code == 0

    def test_one_failure_among_many_exits_non_zero(self, script, monkeypatch, no_sleep):
        _stub_fetch(script, monkeypatch, [{"a": [Path("a.nc")]}, DatasetLoadError(["k"])])
        monkeypatch.setattr(script, "requests", [_request(script, "good"), _request(script, "bad")])

        app = typer.Typer()
        app.command()(script.main)
        result = CliRunner().invoke(app, ["--max-attempts", "1"])

        assert result.exit_code == 1

    def test_unknown_request_id_exits_non_zero(self, script, monkeypatch, no_sleep):
        monkeypatch.setattr(script, "requests", [_request(script, "only")])

        app = typer.Typer()
        app.command()(script.main)
        result = CliRunner().invoke(app, ["--request-id", "does-not-exist"])

        assert result.exit_code == 1
