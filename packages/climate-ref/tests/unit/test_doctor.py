"""
Tests for the deployment health checks behind `ref doctor`.

The checks are driven through a `DoctorContext` with its catalogs and providers supplied
directly, so no database or ingest is needed.
"""

import pandas as pd
import pytest

from climate_ref.doctor import (
    DoctorContext,
    Finding,
    Severity,
    check_duplicate_coverage,
    check_missing_reference_data,
    check_unreachable_source_types,
    run_checks,
    worst_severity,
)
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
        assert findings[0].check == "duplicate-coverage"
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

    def test_ingested_dataset_is_not_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("obs4MIPs.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context({SourceDatasetType.obs4MIPs: catalog}, [provider])

        assert check_missing_reference_data(context) == []

    def test_data_under_another_source_type_does_not_satisfy_a_requirement(self):
        # The obs4ref trap: the data is present, but in a table no requirement reads.
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "WECANN-1-0", "gpp")
        catalog = _catalog(
            [("obs4REF.WECANN-1-0.gpp", "WECANN-1-0", "gpp", "2007-01-01", "2015-12-01", "/d/gpp.nc")]
        )
        context = _context(
            {SourceDatasetType.obs4MIPs: _catalog([]), SourceDatasetType.obs4REF: catalog},
            [provider],
        )

        findings = check_missing_reference_data(context)

        assert len(findings) == 1
        assert "WECANN-1-0" in findings[0].summary


class TestUnreachableSourceTypes:
    def test_data_no_requirement_asks_for_is_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog(
            [("obs4REF.WECANN-1-0.gpp", "WECANN-1-0", "gpp", "2007-01-01", "2015-12-01", "/d/gpp.nc")]
        )
        context = _context({SourceDatasetType.obs4REF: catalog}, [provider])

        findings = check_unreachable_source_types(context)

        assert len(findings) == 1
        assert findings[0].check == "unreachable-source-type"
        assert "obs4ref" in findings[0].summary
        assert "--source-type obs4mips" in findings[0].detail

    def test_requested_source_type_is_not_reported(self):
        provider = _provider_requiring(SourceDatasetType.obs4MIPs, "ERA-5", "ta")
        catalog = _catalog([("obs4MIPs.ERA-5.ta", "ERA-5", "ta", "2000-01-01", "2000-12-01", "/d/ta.nc")])
        context = _context({SourceDatasetType.obs4MIPs: catalog}, [provider])

        assert check_unreachable_source_types(context) == []


class TestRunChecks:
    def test_a_failing_check_becomes_a_finding(self):
        def broken(context):
            raise RuntimeError("boom")

        findings = run_checks(_context(), checks=[broken])

        assert len(findings) == 1
        assert findings[0].severity == Severity.ERROR
        assert "boom" in findings[0].detail

    def test_findings_are_ordered_worst_first(self):
        def mixed(context):
            return [
                Finding(check="c", severity=Severity.INFO, summary="info"),
                Finding(check="a", severity=Severity.ERROR, summary="error"),
                Finding(check="b", severity=Severity.WARNING, summary="warning"),
            ]

        findings = run_checks(_context(), checks=[mixed])

        assert [f.severity for f in findings] == [Severity.ERROR, Severity.WARNING, Severity.INFO]

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
        findings = [Finding(check="c", severity=s, summary="s") for s in severities]

        assert worst_severity(findings) == expected
