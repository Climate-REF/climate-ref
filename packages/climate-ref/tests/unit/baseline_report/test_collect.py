"""Tests for collecting the manifest changes on a branch."""

import pytest
from git import Repo

from climate_ref.baseline_report.collect import (
    FileKind,
    case_label,
    classify,
    collect,
)
from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest, NativeEntry

MANIFEST_PATH = "packages/climate-ref-example/tests/test-data/global-mean-timeseries/default/manifest.json"


def _digest(char: str) -> str:
    """Build a valid sha256 digest from a single repeated hex character."""
    return char * 64


def _write_manifest(repo_dir, rel_path, *, version, native, committed=None, catalog_hash=None):
    """Write a manifest at ``rel_path`` inside ``repo_dir``."""
    path = repo_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Manifest(
        schema=SCHEMA_VERSION,
        test_case_version=version,
        diagnostic_version=1,
        committed=committed or {},
        native=native,
        catalog_hash=catalog_hash,
    ).dump(path)
    return path


@pytest.fixture
def repo(tmp_path):
    """A git repository with one manifest committed twice, the second time with changes."""
    repo = Repo.init(tmp_path)
    with repo.config_writer() as writer:
        writer.set_value("user", "name", "test")
        writer.set_value("user", "email", "test@example.com")

    _write_manifest(
        tmp_path,
        MANIFEST_PATH,
        version=3,
        native={
            "kept.png": NativeEntry(sha256=_digest("1"), size=10),
            "changed.json": NativeEntry(sha256=_digest("2"), size=20),
            "removed.nc": NativeEntry(sha256=_digest("3"), size=30),
        },
        committed={"series.json": _digest("a")},
    )
    repo.git.add("-A")
    repo.index.commit("base")

    _write_manifest(
        tmp_path,
        MANIFEST_PATH,
        version=4,
        native={
            "kept.png": NativeEntry(sha256=_digest("1"), size=10),
            "changed.json": NativeEntry(sha256=_digest("4"), size=25),
            "added.bin": NativeEntry(sha256=_digest("5"), size=40),
        },
        committed={"series.json": _digest("b")},
    )
    repo.git.add("-A")
    repo.index.commit("head")
    return repo


class TestClassify:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("plot.png", FileKind.IMAGE),
            ("plot.JPG", FileKind.IMAGE),
            ("plot.jpeg", FileKind.IMAGE),
            ("plot.gif", FileKind.IMAGE),
            ("plot.svg", FileKind.IMAGE),
            ("a.json", FileKind.TEXT),
            ("a.csv", FileKind.TEXT),
            ("a.yml", FileKind.TEXT),
            ("a.yaml", FileKind.TEXT),
            ("a.html", FileKind.TEXT),
            ("a.txt", FileKind.TEXT),
            ("a.md", FileKind.TEXT),
            ("a.log", FileKind.TEXT),
            ("out.nc", FileKind.NETCDF),
            ("blob.bin", FileKind.OTHER),
            ("noextension", FileKind.OTHER),
        ],
    )
    def test_suffixes(self, name, expected):
        assert classify(name) is expected


class TestCaseLabel:
    def test_strips_the_provider_prefix(self):
        assert case_label(MANIFEST_PATH) == "example/global-mean-timeseries/default"

    def test_short_path_keeps_only_the_provider(self):
        assert case_label("packages/climate-ref-pmp") == "pmp"


class TestCollect:
    def test_pairs_native_entries(self, repo):
        report = collect(repo, "HEAD~1")

        assert report.base_ref == "HEAD~1"
        assert report.head_sha == repo.head.commit.hexsha
        assert len(report.cases) == 1

        case = report.cases[0]
        assert case.label == "example/global-mean-timeseries/default"
        assert not case.is_new
        assert not case.is_removed

        # Unchanged entries are dropped, and the rest are in name order.
        assert [(f.name, f.status, f.kind) for f in case.files] == [
            ("added.bin", "added", FileKind.OTHER),
            ("changed.json", "changed", FileKind.TEXT),
            ("removed.nc", "removed", FileKind.NETCDF),
        ]

    def test_reports_metadata_and_committed_changes(self, repo):
        case = collect(repo, "HEAD~1").cases[0]

        assert case.metadata == ("test_case_version: 3 -> 4",)
        assert case.committed == ("series.json",)

    def test_new_case_has_no_base(self, tmp_path):
        repo = Repo.init(tmp_path)
        with repo.config_writer() as writer:
            writer.set_value("user", "name", "test")
            writer.set_value("user", "email", "test@example.com")
        (tmp_path / "README.md").write_text("seed")
        repo.git.add("-A")
        repo.index.commit("seed")

        _write_manifest(
            tmp_path,
            MANIFEST_PATH,
            version=7,
            native={"plot.png": NativeEntry(sha256=_digest("1"), size=10)},
        )
        repo.git.add("-A")
        repo.index.commit("add case")

        case = collect(repo, "HEAD~1").cases[0]

        assert case.base is None
        assert case.is_new
        assert case.metadata == ("new test case at test_case_version 7",)
        assert [f.status for f in case.files] == ["added"]

    def test_no_changes_gives_no_cases(self, repo):
        assert collect(repo, "HEAD").cases == ()

    def test_an_unreadable_manifest_is_skipped(self, repo, tmp_path):
        # A manifest written by an incompatible version should cost its own case, not the report.
        (tmp_path / MANIFEST_PATH).write_text('{"schema": 1}')
        repo.git.add("-A")
        repo.index.commit("break the manifest")

        assert collect(repo, "HEAD~1").cases == ()
