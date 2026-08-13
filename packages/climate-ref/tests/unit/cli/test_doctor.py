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

        assert "1 warning" in result.stdout
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

        assert "1 error" in result.stdout

    def test_the_environment_is_omitted_from_the_text_output_by_default(self, findings, invoke_cli):
        findings([])

        result = invoke_cli(["doctor"])

        assert "versions" not in result.stdout

    def test_the_environment_can_be_added_to_the_text_output(self, findings, invoke_cli):
        findings([])

        result = invoke_cli(["doctor", "--environment"])

        assert "versions" in result.stdout
        assert "climate-ref" in result.stdout


class TestGrouping:
    """
    The text output states once what a set of findings has in common.

    A deployment missing twenty reference datasets should read as one instruction and twenty
    names, so these guard the report against going back to twenty copies of the instruction.
    """

    @pytest.fixture
    def findings(self, monkeypatch):
        def _set(values):
            monkeypatch.setattr("climate_ref.cli.doctor.run_checks", lambda context: values)

        return _set

    def _missing(self, source_id, remedy="fetch it", command="", check="missing-reference-data"):
        return Finding(
            check=check,
            severity=Severity.WARNING,
            summary=f"{source_id} is not ingested",
            remedy=remedy,
            command=command,
        )

    def test_a_shared_remedy_is_stated_once(self, findings, invoke_cli):
        findings([self._missing("A"), self._missing("B"), self._missing("C")])

        stdout = invoke_cli(["doctor"]).stdout

        assert stdout.count("fetch it") == 1
        for source_id in ("A", "B", "C"):
            assert f"{source_id} is not ingested" in stdout

    def test_findings_with_different_remedies_keep_their_own(self, findings, invoke_cli):
        findings([self._missing("A", remedy="fetch it"), self._missing("B", remedy="ask ESGF")])

        stdout = invoke_cli(["doctor"]).stdout

        assert "fetch it" in stdout
        assert "ask ESGF" in stdout

    def test_the_check_is_named_once_per_group(self, findings, invoke_cli):
        findings([self._missing("A"), self._missing("B")])

        stdout = invoke_cli(["doctor"]).stdout

        assert stdout.count("missing-reference-data") == 1

    def test_the_command_is_printed_unwrapped(self, findings, invoke_cli):
        command = "ref datasets fetch-data --registry obs4ref --output-directory <dir>"
        findings([self._missing("A", command=command)])

        stdout = invoke_cli(["doctor"]).stdout

        # Pasteable only if no wrap was inserted between its arguments.
        assert command in stdout

    def test_a_mixed_group_names_each_severity(self, findings, invoke_cli):
        findings(
            [
                Finding(check="c", severity=Severity.ERROR, summary="an error", remedy="r"),
                Finding(check="c", severity=Severity.WARNING, summary="a warning", remedy="r"),
            ]
        )

        stdout = invoke_cli(["doctor"], expected_exit_code=1).stdout

        assert "error an error" in stdout
        assert "warning a warning" in stdout

    def test_a_uniform_group_does_not_repeat_the_severity(self, findings, invoke_cli):
        findings([self._missing("A"), self._missing("B")])

        stdout = invoke_cli(["doctor"]).stdout

        # Once in the summary line, once in the group heading, and not on either finding.
        assert stdout.count("warning") == 2

    def test_the_summary_comes_before_the_findings(self, findings, invoke_cli):
        findings([self._missing("A")])

        stdout = invoke_cli(["doctor"]).stdout

        assert stdout.index("1 finding from") < stdout.index("A is not ingested")


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
            {
                "check": "c",
                "severity": "warning",
                "summary": "a warning",
                "detail": "do this",
                "remedy": "",
                "command": "",
            }
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
