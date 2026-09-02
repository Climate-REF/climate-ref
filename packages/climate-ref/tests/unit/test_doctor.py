"""
Tests for the deployment health checks behind `ref doctor`.

The checks are driven through a `DoctorContext` with its catalogs and providers supplied directly,
so no database or ingest is needed.
"""

import pandas as pd
import pytest

from climate_ref.doctor import (
    DoctorContext,
    Finding,
    Severity,
    diagnose,
    iter_checks,
    worst_severity,
)
from climate_ref.doctor.checks.data import (
    check_duplicate_coverage,
    check_misfiled_obs4ref,
    check_missing_reference_data,
    check_superseded_obs4ref,
    check_unreachable_source_types,
    check_unsolvable_diagnostics,
)
from climate_ref.doctor.registry import RegisteredCheck, run_checks
from climate_ref_core.datasets import FacetFilter
from climate_ref_core.diagnostics import DataRequirement, Diagnostic
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.source_types import SourceDatasetType


def _catalog(rows):
    """Build a file-level catalog frame of the shape the adapters return."""
    return pd.DataFrame(
        rows,
        columns=["instance_id", "source_id", "variable_id", "start_time", "end_time", "path"],
    ).astype({"start_time": "datetime64[ns]", "end_time": "datetime64[ns]"})


def _context(catalogs=None, providers=None):
    return DoctorContext.from_catalogs(catalogs or {}, providers or [])


def _provider_requiring(source_type: SourceDatasetType, source_id: str, variable_id: str):
    """A provider with one diagnostic requiring a single reference dataset."""

    class _Diagnostic(Diagnostic):
        name = "needs-reference"
        slug = "needs-reference"
        facets = ()
        data_requirements = (
            DataRequirement(
                source_type=source_type,
                filters=(FacetFilter(facets={"source_id": source_id, "variable_id": variable_id}),),
                group_by=None,
            ),
        )

        def run(self, definition, *, capture_regression=False):  # pragma: no cover - never run
            raise NotImplementedError

    provider = DiagnosticProvider("test_provider", "v0.1.0")
    provider.register(_Diagnostic())  # type: ignore[arg-type]
    return provider


class TestDuplicateCoverage:
    def test_overlapping_files_are_reported(self):
        # The real case: one whole-period file from obs4REF and a yearly file from ESGF.
        catalog = _catalog(
            [
                ("obs4MIPs.X.pr", "X", "pr", "1983-01-01", "2023-03-01", "/cache/obs4ref/X/pr.nc"),
                ("obs4MIPs.X.pr", "X", "pr", "1983-01-01", "1983-12-01", "/mnt/esgf/obs4MIPs/X/pr_1983.nc"),
            ]
        )
        findings = check_duplicate_coverage(_context({SourceDatasetType.obs4MIPs: catalog}))

        assert len(findings) == 1
        assert findings[0].severity == Severity.ERROR
        assert "obs4MIPs.X.pr" in findings[0].summary

    def test_contiguous_files_are_not_reported(self):
        # A dataset split into consecutive yearly files is normal and must not be flagged.
        catalog = _catalog(
            [
                ("obs4MIPs.X.pr", "X", "pr", "1983-01-01", "1983-12-01", "/mnt/esgf/pr_1983.nc"),
                ("obs4MIPs.X.pr", "X", "pr", "1984-01-01", "1984-12-01", "/mnt/esgf/pr_1984.nc"),
                ("obs4MIPs.X.pr", "X", "pr", "1985-01-01", "1985-12-01", "/mnt/esgf/pr_1985.nc"),
            ]
        )
        assert check_duplicate_coverage(_context({SourceDatasetType.obs4MIPs: catalog})) == []

    def test_single_file_dataset_is_not_reported(self):
        catalog = _catalog([("obs4MIPs.X.pr", "X", "pr", "1983-01-01", "2023-03-01", "/cache/pr.nc")])
        assert check_duplicate_coverage(_context({SourceDatasetType.obs4MIPs: catalog})) == []

    def test_other_datasets_are_unaffected(self):
        catalog = _catalog(
            [
                ("obs4MIPs.X.pr", "X", "pr", "1983-01-01", "2023-03-01", "/cache/x.nc"),
                ("obs4MIPs.X.pr", "X", "pr", "1983-01-01", "1983-12-01", "/mnt/x_1983.nc"),
                ("obs4MIPs.Y.ts", "Y", "ts", "1983-01-01", "2023-03-01", "/cache/y.nc"),
            ]
        )
        findings = check_duplicate_coverage(_context({SourceDatasetType.obs4MIPs: catalog}))

        assert [f.summary.split()[0] for f in findings] == ["obs4MIPs.X.pr"]


class TestMissingReferenceData:
    def test_missing_dataset_is_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        context = _context({SourceDatasetType.obs4MIPs: _catalog([])}, [provider])

        findings = check_missing_reference_data(context)

        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert "ERA-5" in findings[0].summary
        assert "needs-reference" in findings[0].detail

    def test_datasets_obtained_the_same_way_share_a_remedy(self):
        # The report states a shared remedy once, which only holds if the wording matches exactly.
        providers = [
            _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta"),
            _provider_requiring(SourceDatasetType.obs4MIPs, "C3S-GTO-ECV-9-0", "toz"),
        ]
        context = _context({SourceDatasetType.obs4MIPs: _catalog([])}, providers)

        findings = check_missing_reference_data(context)

        assert len(findings) == 2
        assert findings[0].remedy == findings[1].remedy
        assert "ERA-5" not in findings[0].remedy

    def test_the_summary_counts_the_diagnostics(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        context = _context({SourceDatasetType.obs4MIPs: _catalog([])}, [provider])

        assert "1 diagnostic will not run" in check_missing_reference_data(context)[0].summary

    def test_ingested_dataset_is_not_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("obs4MIPs.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context({SourceDatasetType.obs4MIPs: catalog}, [provider])

        assert check_missing_reference_data(context) == []

    def test_data_under_another_source_type_does_not_satisfy_a_requirement(self):
        # The data is present, but in a table the requirement does not read.
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("pmp.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context(
            {SourceDatasetType.obs4MIPs: _catalog([]), SourceDatasetType.PMPClimatology: catalog},
            [provider],
        )

        findings = check_missing_reference_data(context)

        assert len(findings) == 1
        assert "ERA-5" in findings[0].summary

    def test_obs4mips_requirement_is_met_by_obs4ref_data(self):
        # obs4REF fills in what obs4MIPs has not published, so this is not missing.
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "WECANN-1-0", "gpp")
        catalog = _catalog(
            [("obs4REF.WECANN-1-0.gpp", "WECANN-1-0", "gpp", "2007-01-01", "2015-12-01", "/d/gpp.nc")]
        )
        context = _context({SourceDatasetType.obs4REF: catalog}, [provider])

        assert check_missing_reference_data(context) == []


class TestUnreachableSourceTypes:
    def test_data_no_requirement_asks_for_is_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("pmp.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context({SourceDatasetType.PMPClimatology: catalog}, [provider])

        findings = check_unreachable_source_types(context)

        assert len(findings) == 1
        assert "pmp-climatology" in findings[0].summary
        assert "source type the diagnostics ask for" in findings[0].remedy

    def test_requested_source_type_is_not_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("obs4MIPs.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context({SourceDatasetType.obs4MIPs: catalog}, [provider])

        assert check_unreachable_source_types(context) == []

    def test_obs4ref_data_is_reachable_through_obs4mips_requirements(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "WECANN-1-0", "gpp")
        catalog = _catalog(
            [("obs4REF.WECANN-1-0.gpp", "WECANN-1-0", "gpp", "2007-01-01", "2015-12-01", "/d/gpp.nc")]
        )
        context = _context({SourceDatasetType.obs4REF: catalog}, [provider])

        assert check_unreachable_source_types(context) == []


class TestSupersededObs4ref:
    def test_dataset_in_both_is_reported(self):
        obs4mips = _catalog(
            [
                (
                    "obs4MIPs.obs4MIPs.C.WECANN-1-0.mon.gpp.gn.v2",
                    "WECANN-1-0",
                    "gpp",
                    "2007-01-01",
                    "2015-12-01",
                    "/a",
                )
            ]
        )
        obs4ref = _catalog(
            [
                (
                    "obs4REF.obs4REF.C.WECANN-1-0.mon.gpp.gn.v1",
                    "WECANN-1-0",
                    "gpp",
                    "2007-01-01",
                    "2015-12-01",
                    "/b",
                ),
                (
                    "obs4REF.obs4REF.C.WECANN-1-0.mon.hfls.gn.v1",
                    "WECANN-1-0",
                    "hfls",
                    "2007-01-01",
                    "2015-12-01",
                    "/c",
                ),
            ]
        )
        context = _context({SourceDatasetType.obs4MIPs: obs4mips, SourceDatasetType.obs4REF: obs4ref})

        findings = check_superseded_obs4ref(context)

        assert len(findings) == 1
        assert findings[0].severity == Severity.INFO
        assert findings[0].summary.startswith("obs4REF.obs4REF.C.WECANN-1-0.mon.gpp.gn.v1")

    def test_nothing_ingested(self):
        assert check_superseded_obs4ref(_context()) == []


class TestMisfiledObs4ref:
    def test_obs4ref_layout_under_obs4mips_is_reported(self):
        catalog = _catalog(
            [
                (
                    "obs4MIPs.obs4MIPs.X.gpp",
                    "X-1-0",
                    "gpp",
                    "2007-01-01",
                    "2015-12-01",
                    "/d/obs4REF/X/gpp.nc",
                ),
                (
                    "obs4MIPs.obs4MIPs.ERA-5.ta",
                    "ERA-5",
                    "ta",
                    "2000-01-01",
                    "2000-12-01",
                    "/d/obs4MIPs/ta.nc",
                ),
            ]
        )
        context = _context({SourceDatasetType.obs4MIPs: catalog})

        findings = check_misfiled_obs4ref(context)

        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert findings[0].summary == "obs4MIPs.obs4MIPs.X.gpp is obs4REF data ingested as obs4mips"
        assert findings[0].command == "ref datasets ingest --source-type obs4ref <dir>"

    def test_each_misfiled_dataset_gets_its_own_finding(self):
        catalog = _catalog(
            [
                (
                    "obs4MIPs.obs4MIPs.X.gpp",
                    "X-1-0",
                    "gpp",
                    "2007-01-01",
                    "2015-12-01",
                    "/d/obs4REF/X/gpp.nc",
                ),
                ("obs4MIPs.obs4MIPs.Y.ts", "Y-1-0", "ts", "2007-01-01", "2015-12-01", "/d/obs4REF/Y/ts.nc"),
            ]
        )
        context = _context({SourceDatasetType.obs4MIPs: catalog})

        findings = check_misfiled_obs4ref(context)

        # One remedy shared across them, so the report groups them under it.
        assert len(findings) == 2
        assert len({f.remedy for f in findings}) == 1

    def test_registry_source_id_from_esgf_is_not_reported(self):
        # HadISST-1-1 sits in both collections, so an ESGF copy is legitimately ingested as obs4MIPs.
        catalog = _catalog(
            [("obs4MIPs.obs4MIPs.HadISST.ts", "HadISST-1-1", "ts", "2000-01-01", "2000-12-01", "/d/ts.nc")]
        )
        context = _context({SourceDatasetType.obs4MIPs: catalog})

        assert check_misfiled_obs4ref(context) == []

    def test_esgf_data_is_not_reported(self):
        catalog = _catalog([("obs4MIPs.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context({SourceDatasetType.obs4MIPs: catalog})

        assert check_misfiled_obs4ref(context) == []

    def test_nothing_ingested(self):
        assert check_misfiled_obs4ref(_context()) == []


class TestUnsolvableDiagnostics:
    def test_diagnostic_with_nothing_ingested_is_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        context = _context({}, [provider])

        findings = check_unsolvable_diagnostics(context)

        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert "test_provider/needs-reference has no executions" == findings[0].summary
        assert "nothing is ingested as obs4mips" in findings[0].detail

    def test_unmatched_filter_names_the_facets(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("obs4MIPs.ERA-5.psl", "ERA-5", "psl", "2000-01-01", "2000-12-01", "/d/psl.nc")])
        context = _context({SourceDatasetType.obs4MIPs: catalog}, [provider])

        findings = check_unsolvable_diagnostics(context)

        assert len(findings) == 1
        assert "no obs4mips datasets match source_id=ERA-5, variable_id=ta" in findings[0].detail

    def test_solvable_diagnostic_is_not_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("obs4MIPs.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context({SourceDatasetType.obs4MIPs: catalog}, [provider])

        assert check_unsolvable_diagnostics(context) == []

    def test_obs4mips_requirement_solves_from_obs4ref_data(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "WECANN-1-0", "gpp")
        catalog = _catalog(
            [("obs4REF.WECANN-1-0.gpp", "WECANN-1-0", "gpp", "2007-01-01", "2015-12-01", "/d/gpp.nc")]
        )
        context = _context({SourceDatasetType.obs4REF: catalog}, [provider])

        assert check_unsolvable_diagnostics(context) == []


class TestDiagnose:
    """
    `diagnose` is the way in: one call, one value holding everything needed to report on it.
    """

    def test_it_runs_every_registered_check(self):
        report = diagnose(_context())

        assert report.check_count == len(iter_checks())

    def test_the_environment_is_left_out_unless_asked_for(self):
        assert diagnose(_context()).environment is None

    def test_the_environment_is_collected_on_request(self):
        report = diagnose(_context(), environment=True)

        assert report.environment is not None
        assert report.environment["versions"]["climate-ref"]

    def test_the_worst_severity_is_carried_on_the_report(self, monkeypatch):
        monkeypatch.setattr(
            "climate_ref.doctor.report.run_checks",
            lambda context: [
                Finding(severity=Severity.INFO, summary="info"),
                Finding(severity=Severity.WARNING, summary="warning"),
            ],
        )

        assert diagnose(_context()).worst_severity == Severity.WARNING

    def test_a_clean_deployment_has_no_worst_severity(self, monkeypatch):
        monkeypatch.setattr("climate_ref.doctor.report.run_checks", lambda context: [])

        assert diagnose(_context()).worst_severity is None


def _registered(func, slug="a-check"):
    return RegisteredCheck(slug=slug, description="A check used by the tests", func=func)


class TestRunChecks:
    def test_a_failing_check_becomes_a_finding(self):
        def broken(context):
            raise RuntimeError("boom")

        findings = run_checks(_context(), checks=[_registered(broken, "broken")])

        assert len(findings) == 1
        assert findings[0].severity == Severity.ERROR
        assert findings[0].check == "broken"
        assert "boom" in findings[0].detail

    def test_findings_are_ordered_worst_first(self):
        def mixed(context):
            return [
                Finding(severity=Severity.INFO, summary="info"),
                Finding(severity=Severity.ERROR, summary="error"),
                Finding(severity=Severity.WARNING, summary="warning"),
            ]

        findings = run_checks(_context(), checks=[_registered(mixed)])

        assert [f.severity for f in findings] == [Severity.ERROR, Severity.WARNING, Severity.INFO]

    def test_the_slug_is_stamped_onto_every_finding(self):
        # A check states its slug once, in its registration, so the two cannot drift apart.
        def two_findings(context):
            return [
                Finding(severity=Severity.INFO, summary="one"),
                Finding(severity=Severity.INFO, summary="two"),
            ]

        findings = run_checks(_context(), checks=[_registered(two_findings, "stamped")])

        assert [f.check for f in findings] == ["stamped", "stamped"]

    @pytest.mark.parametrize(
        "severities, expected",
        [
            ([], None),
            ([Severity.INFO], Severity.INFO),
            ([Severity.INFO, Severity.WARNING], Severity.WARNING),
            ([Severity.WARNING, Severity.ERROR], Severity.ERROR),
        ],
    )
    def test_worst_severity(self, severities, expected):
        findings = [Finding(severity=s, summary="s") for s in severities]

        assert worst_severity(findings) == expected
