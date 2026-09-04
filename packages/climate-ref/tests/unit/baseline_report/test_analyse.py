"""Tests for fetching blobs and building the diffs the templates render."""

import json
from unittest.mock import MagicMock

import pytest
from attrs import evolve

from climate_ref.baseline_report.analyse import MAX_FETCH_BYTES, analyse, blob_url, text_diff
from climate_ref.baseline_report.collect import (
    CaseChange,
    FileChange,
    FileKind,
    Report,
    classify,
)
from climate_ref_core.regression.manifest import NativeEntry
from climate_ref_core.regression.store import NativeStore


def _kinds(diff):
    """Return the kind of every line in a diff."""
    return [line.kind for line in diff.lines]


def _file_change(name, old, new):
    """Build a file change from two entries."""
    return FileChange(name=name, old=old, new=new, kind=classify(name))


def _report(files):
    """Wrap file changes in a single-case report."""
    case = CaseChange(
        label="example/diag/case",
        rel_path="packages/climate-ref-example/tests/test-data/diag/case/manifest.json",
        base=None,
        head=None,
        files=tuple(files),
        committed=(),
        metadata=(),
    )
    return Report(base_ref="origin/main", head_sha="a" * 40, cases=(case,))


class TestBlobUrl:
    def test_a_remote_store_serves_blobs_flat(self):
        store = NativeStore(url="https://store/")

        assert blob_url(store, "a" * 64) == f"https://store/{'a' * 64}"

    def test_a_local_store_keeps_its_two_level_fan_out(self, tmp_path):
        store = NativeStore(url=str(tmp_path / "store"))
        digest = "a" * 64

        url = blob_url(store, digest)

        assert url.startswith("file://")
        assert url.endswith(f"/aa/{digest}")


class TestTextDiff:
    def test_json_key_order_is_not_a_difference(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        old.write_text(json.dumps({"b": 2, "a": 1}))
        new.write_text(json.dumps({"a": 1, "b": 2}))

        diff = text_diff(old, new, "series.json")

        assert diff.note == "identical after decoding"
        assert diff.lines == ()

    def test_one_changed_line(self, tmp_path):
        old = tmp_path / "old.csv"
        new = tmp_path / "new.csv"
        old.write_text("header\nvalue 1\ntail\n")
        new.write_text("header\nvalue 2\ntail\n")

        diff = text_diff(old, new, "series.csv")

        assert diff.note is None
        kinds = _kinds(diff)
        assert kinds.count("add") == 1
        assert kinds.count("remove") == 1
        assert "hunk" in kinds
        assert kinds.count("header") == 2

    def test_added_file_is_all_additions(self, tmp_path):
        new = tmp_path / "new.txt"
        new.write_text("one\ntwo\n")

        diff = text_diff(None, new, "new.txt")

        assert [line.kind for line in diff.lines if line.kind not in ("header", "hunk")] == [
            "add",
            "add",
        ]


class TestAnalyse:
    def test_no_fetch_notes_every_text_file(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        report = _report(
            [
                _file_change("plot.png", None, NativeEntry(sha256="1" * 64, size=10)),
                _file_change("series.json", None, NativeEntry(sha256="2" * 64, size=10)),
            ]
        )

        analysed = analyse(report, store, fetch=False, workdir=tmp_path)

        files = {f.change.name: f for f in analysed.cases[0].files}
        assert files["series.json"].text.note == "fetching disabled"
        assert files["plot.png"].text is None
        store.fetch.assert_not_called()

    def test_counts_are_tallied_per_kind(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        entry = NativeEntry(sha256="1" * 64, size=10)
        other = NativeEntry(sha256="2" * 64, size=10)
        report = _report(
            [
                _file_change("a.png", None, entry),
                _file_change("b.png", entry, other),
                _file_change("c.nc", entry, None),
            ]
        )

        rows = analyse(report, store, fetch=False, workdir=tmp_path).cases[0].counts
        counts = {row.label: row for row in rows}

        assert (counts["image"].added, counts["image"].changed, counts["image"].removed) == (1, 1, 0)
        assert (counts["netcdf"].added, counts["netcdf"].changed, counts["netcdf"].removed) == (0, 0, 1)
        assert (counts["text"].added, counts["text"].changed, counts["text"].removed) == (0, 0, 0)
        # Every kind gets a column, in the enum's order, so the index header cannot drift.
        assert [
            row.label for row in analyse(report, store, fetch=False, workdir=tmp_path).cases[0].counts
        ] == [kind.value for kind in FileKind]

    def test_partitions_and_back_link(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        entry = NativeEntry(sha256="1" * 64, size=10)
        report = _report(
            [
                _file_change("a.png", None, entry),
                _file_change("b.json", None, entry),
                _file_change("c.nc", None, entry),
                _file_change("d.bin", None, entry),
            ]
        )

        case = analyse(report, store, fetch=False, workdir=tmp_path).cases[0]

        assert [f.change.name for f in case.images] == ["a.png"]
        assert [f.change.name for f in case.texts] == ["b.json"]
        assert [f.change.name for f in case.binaries] == ["c.nc", "d.bin"]
        # The label has three segments, so the index sits three levels up.
        assert case.back_link == "../../../index.html"

    def test_the_back_link_follows_the_label_depth(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        report = _report([])
        shallow = evolve(report.cases[0], label="pmp")
        report = evolve(report, cases=(shallow,))

        case = analyse(report, store, fetch=False, workdir=tmp_path).cases[0]

        assert case.back_link == "../index.html"

    def test_size_delta_is_signed_and_absent_on_one_sided_files(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        report = _report(
            [
                _file_change(
                    "grew.png", NativeEntry(sha256="1" * 64, size=10), NativeEntry(sha256="2" * 64, size=25)
                ),
                _file_change("added.png", None, NativeEntry(sha256="3" * 64, size=25)),
            ]
        )

        files = {
            f.change.name: f for f in analyse(report, store, fetch=False, workdir=tmp_path).cases[0].files
        }

        assert files["grew.png"].size_delta == 15
        assert files["added.png"].size_delta is None

    def test_local_store_produces_a_real_diff(self, tmp_path):
        store = NativeStore(url=str(tmp_path / "store"))
        old_file = tmp_path / "old.csv"
        new_file = tmp_path / "new.csv"
        old_file.write_text("a\nb\n")
        new_file.write_text("a\nc\n")
        old_digest = store.put(old_file)
        new_digest = store.put(new_file)

        report = _report(
            [
                _file_change(
                    "series.csv",
                    NativeEntry(sha256=old_digest, size=old_file.stat().st_size),
                    NativeEntry(sha256=new_digest, size=new_file.stat().st_size),
                )
            ]
        )

        analysed = analyse(report, store, fetch=True, workdir=tmp_path / "work")

        diff = analysed.cases[0].files[0].text
        assert diff.note is None
        assert [line.text for line in diff.lines if line.kind == "add"] == ["+c"]
        assert [line.text for line in diff.lines if line.kind == "remove"] == ["-b"]

    def test_oversized_blob_is_noted_not_fetched(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        report = _report(
            [
                _file_change(
                    "big.json",
                    None,
                    NativeEntry(sha256="1" * 64, size=MAX_FETCH_BYTES + 1),
                )
            ]
        )

        analysed = analyse(report, store, fetch=True, workdir=tmp_path)

        assert "too large to diff" in analysed.cases[0].files[0].text.note
        store.fetch.assert_not_called()

    def test_a_failed_fetch_becomes_a_note(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        store.fetch.side_effect = FileNotFoundError("gone")
        report = _report([_file_change("series.json", None, NativeEntry(sha256="1" * 64, size=10))])

        analysed = analyse(report, store, fetch=True, workdir=tmp_path)

        assert "could not fetch" in analysed.cases[0].files[0].text.note

    @pytest.mark.parametrize("fetch", [True, False])
    def test_urls_follow_the_entries_that_exist(self, tmp_path, fetch):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store/"
        store.root = None
        report = _report([_file_change("plot.png", NativeEntry(sha256="1" * 64, size=10), None)])

        analysed = analyse(report, store, fetch=fetch, workdir=tmp_path)

        file = analysed.cases[0].files[0]
        assert file.old_url == f"https://store/{'1' * 64}"
        assert file.new_url is None
        assert analysed.store_url == "https://store"
