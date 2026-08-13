import json

import pytest

from climate_ref.doctor import Finding, Severity


def test_doctor_help(invoke_cli):
    result = invoke_cli(["doctor", "--help"])

    assert "Check this deployment for data and configuration problems" in result.stdout


class TestDoctor:
    @pytest.fixture
    def findings(self, monkeypatch):
        """Drive the command's output from a fixed set of findings."""

        def _set(values):
            monkeypatch.setattr("climate_ref.cli.doctor.run_checks", lambda context: values)

        return _set

    def test_no_findings_reports_success(self, findings, invoke_cli):
        findings([])

        result = invoke_cli(["doctor"])

        assert "No problems found" in result.stdout

    def test_warnings_alone_exit_zero(self, findings, invoke_cli):
        findings([Finding(check="c", severity=Severity.WARNING, summary="a warning", detail="do this")])

        result = invoke_cli(["doctor"])

        assert "WARNING" in result.stdout
        assert "a warning" in result.stdout
        assert "do this" in result.stdout

    def test_quiet_omits_the_detail(self, findings, invoke_cli):
        findings([Finding(check="c", severity=Severity.WARNING, summary="a warning", detail="do this")])

        result = invoke_cli(["doctor", "--quiet"])

        assert "a warning" in result.stdout
        assert "do this" not in result.stdout

    def test_strict_exits_non_zero_for_warnings(self, findings, invoke_cli):
        findings([Finding(check="c", severity=Severity.WARNING, summary="a warning")])

        invoke_cli(["doctor", "--strict"], expected_exit_code=1)

    def test_errors_exit_non_zero(self, findings, invoke_cli):
        findings([Finding(check="c", severity=Severity.ERROR, summary="an error")])

        result = invoke_cli(["doctor"], expected_exit_code=1)

        assert "ERROR" in result.stdout

    def test_the_environment_is_omitted_from_the_text_output_by_default(self, findings, invoke_cli):
        findings([])

        result = invoke_cli(["doctor"])

        assert "versions" not in result.stdout

    def test_the_environment_can_be_added_to_the_text_output(self, findings, invoke_cli):
        findings([])

        result = invoke_cli(["doctor", "--environment"])

        assert "versions" in result.stdout
        assert "climate-ref" in result.stdout


class TestListChecks:
    def test_it_lists_the_registered_checks(self, invoke_cli):
        result = invoke_cli(["doctor", "--list"])

        assert "duplicate-coverage" in result.stdout
        assert "built-in" in result.stdout


class TestJsonFormat:
    @pytest.fixture
    def report(self, invoke_cli, monkeypatch):
        def _run(findings, args=()):
            monkeypatch.setattr("climate_ref.cli.doctor.run_checks", lambda context: findings)
            result = invoke_cli(["doctor", "--format", "json", *args])
            return json.loads(result.stdout)

        return _run

    def test_findings_are_machine_readable(self, report):
        parsed = report(
            [Finding(check="c", severity=Severity.WARNING, summary="a warning", detail="do this")]
        )

        assert parsed["findings"] == [
            {"check": "c", "severity": "warning", "summary": "a warning", "detail": "do this"}
        ]
        assert parsed["worst_severity"] == "warning"

    def test_the_environment_is_included_by_default(self, report):
        parsed = report([])

        assert parsed["environment"]["versions"]["climate-ref"]
        assert parsed["findings"] == []
        assert parsed["worst_severity"] is None

    def test_the_environment_can_be_omitted(self, report):
        parsed = report([], args=["--no-environment"])

        assert "environment" not in parsed


class TestMarkdownFormat:
    @pytest.fixture
    def render(self, invoke_cli, monkeypatch):
        def _render(findings, args=()):
            monkeypatch.setattr("climate_ref.cli.doctor.run_checks", lambda context: findings)
            return invoke_cli(["doctor", "--format", "markdown", *args]).stdout

        return _render

    def test_findings_are_rendered_as_a_table(self, render):
        stdout = render(
            [Finding(check="c", severity=Severity.WARNING, summary="a warning", detail="do this")]
        )

        assert "| Severity | Check | Finding |" in stdout
        assert "| warning | `c` | a warning. do this |" in stdout

    def test_a_pipe_in_a_finding_does_not_break_the_table(self, render):
        stdout = render([Finding(check="c", severity=Severity.WARNING, summary="run `a | b`")])

        assert "run `a \\| b`" in stdout

    def test_the_environment_is_included_by_default(self, render):
        stdout = render([])

        assert "No problems found" in stdout
        assert "<details><summary>Environment</summary>" in stdout
        assert "- `climate-ref`:" in stdout
