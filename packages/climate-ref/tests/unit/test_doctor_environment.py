"""
Tests for the environment report that accompanies the findings.
"""

import climate_ref
from climate_ref.database import REDACTED
from climate_ref.doctor import DoctorContext
from climate_ref.doctor.environment import collect_environment


class TestRedaction:
    def test_a_secret_environment_variable_is_hidden(self, monkeypatch):
        monkeypatch.setenv("REF_NATIVE_STORE_SECRET_ACCESS_KEY", "s3cret")
        monkeypatch.setenv("REF_RESULTS_ROOT", "/data/results")

        variables = collect_environment(DoctorContext.from_catalogs({}, []))["environment_variables"]

        assert variables["REF_NATIVE_STORE_SECRET_ACCESS_KEY"] == REDACTED
        assert variables["REF_RESULTS_ROOT"] == "/data/results"

    def test_a_password_inside_a_url_is_hidden(self, monkeypatch):
        # The name says nothing about a credential, but the value carries one.
        monkeypatch.setenv("REF_DATABASE_URL", "postgresql://ref_user:hunter2@db.example.org:5432/ref")

        variables = collect_environment(DoctorContext.from_catalogs({}, []))["environment_variables"]

        assert variables["REF_DATABASE_URL"] == "postgresql://ref_user:***@db.example.org:5432/ref"


class TestCollectEnvironment:
    def test_it_reports_the_installed_versions(self):
        report = collect_environment(DoctorContext.from_catalogs({}, []))

        assert report["versions"]["climate-ref"] == climate_ref.__version__
        assert "python" in report["versions"]

    def test_it_lists_the_checks_that_would_run(self):
        report = collect_environment(DoctorContext.from_catalogs({}, []))

        assert report["checks"]["duplicate-coverage"] == "built-in"

    def test_it_survives_a_context_with_no_configuration(self):
        # `from_catalogs` has no config and no database, which the report must tolerate.
        report = collect_environment(DoctorContext.from_catalogs({}, []))

        assert report["configuration"] == {"config_file": "unavailable"}
        assert report["paths"] == {}

    def test_it_reports_the_configuration_of_a_real_deployment(self, config):
        report = collect_environment(DoctorContext(config=config, database=None))

        assert report["configuration"]["executor"] == config.executor.executor
        assert report["configuration"]["n_jobs"] == str(config.n_jobs)
        assert str(config.paths.scratch) in report["paths"]["scratch"]

    def test_a_section_that_cannot_be_collected_does_not_stop_the_report(self, monkeypatch):
        def explode():
            raise RuntimeError("no interpreter")

        monkeypatch.setattr("climate_ref.doctor.environment._versions", explode)

        report = collect_environment(DoctorContext.from_catalogs({}, []))

        assert "no interpreter" in report["versions"]["error"]
        # The remaining sections are still collected.
        assert report["platform"]["machine"]
