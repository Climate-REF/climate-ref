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
