"""Tests for fetching blobs and building the diffs the templates render."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from attrs import evolve

from climate_ref.baseline_report.analyse import (
    MAX_FETCH_BYTES,
    NETCDF_FETCH_BYTES,
    analyse,
    baseline_tree,
    blob_url,
    committed_diff,
    netcdf_diff,
    text_diff,
)
from climate_ref.baseline_report.collect import (
    CaseChange,
    CommittedChange,
    FileChange,
    FileKind,
    Report,
    classify,
)
from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest, NativeEntry
from climate_ref_core.regression.store import NativeStore


def _kinds(diff):
    """Return the kind of every line in a diff."""
    return [line.kind for line in diff.lines]


def _file_change(name, old, new):
    """Build a file change from two entries."""
    return FileChange(name=name, old=old, new=new, kind=classify(name))


def _report(files, committed=(), base=None, head=None):
    """Wrap file changes in a single-case report."""
    case = CaseChange(
        label="example/diag/case",
        rel_path="packages/climate-ref-example/tests/test-data/diag/case/manifest.json",
        base=base,
        head=head,
        files=tuple(files),
        committed=tuple(committed),
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
        assert [f.change.name for f in case.netcdfs] == ["c.nc"]
        assert [f.change.name for f in case.others] == ["d.bin"]
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

        analysed = analyse(report, store, fetch=True, workdir=tmp_path)

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


def _write_nc(path, data, attrs=None, name="tas"):
    """Write a single-variable dataset to a NetCDF file and return its path."""
    values = np.asarray(data)
    dataset = xr.Dataset(
        {name: (("lat", "lon"), values)},
        coords={"lat": np.arange(values.shape[0], dtype=float), "lon": np.arange(values.shape[1])},
        attrs=attrs or {"title": "a"},
    )
    dataset.to_netcdf(path)
    return path


@pytest.fixture
def base_nc(tmp_path):
    """A two by two dataset both sides of every pair start from."""
    return _write_nc(tmp_path / "old.nc", [[1.0, 2.0], [3.0, 4.0]])


class TestNetcdfDiff:
    def test_identical_files_show_the_whole_header_as_context(self, tmp_path, base_nc):
        new = _write_nc(tmp_path / "new.nc", [[1.0, 2.0], [3.0, 4.0]])

        diff = netcdf_diff(base_nc, new)

        assert diff.note is None
        assert diff.header_changed is False
        assert {line.kind for line in diff.header} == {"context"}
        assert len(diff.header) == len(diff.header_old) == len(diff.header_new)
        assert diff.rows[0].moved is False
        assert diff.rows[0].severity == "same"
        assert diff.rows[0].cells_differ == 0
        assert diff.rows[0].max_abs_diff == 0.0

    def test_a_changed_attribute_shows_in_the_header_only(self, tmp_path, base_nc):
        new = _write_nc(tmp_path / "new.nc", [[1.0, 2.0], [3.0, 4.0]], attrs={"title": "b"})

        diff = netcdf_diff(base_nc, new)

        assert sum(1 for line in diff.header if line.kind == "add") == 1
        assert sum(1 for line in diff.header if line.kind == "remove") == 1
        assert diff.header_changed is True
        assert any(line.kind == "context" for line in diff.header)  # the rest is still shown
        assert all(row.moved is False for row in diff.rows)

    def test_one_changed_value_is_counted_and_measured(self, tmp_path, base_nc):
        new = _write_nc(tmp_path / "new.nc", [[1.0, 2.0], [3.0, 4.5]])

        row = netcdf_diff(base_nc, new).rows[0]

        assert row.cells_differ == 1
        assert row.max_abs_diff == pytest.approx(0.5)
        assert row.max_rel_diff == pytest.approx(0.125)
        assert row.moved is True
        assert row.differs is True
        assert row.maximum.changed is True  # 4.0 -> 4.5
        assert row.minimum.changed is False
        assert row.severity == "changed"

    def test_a_last_bit_difference_reads_as_noise(self, tmp_path, base_nc):
        new = _write_nc(tmp_path / "new.nc", [[1.0, 2.0], [3.0, 4.0 + 1e-14]])

        row = netcdf_diff(base_nc, new).rows[0]

        assert row.cells_differ == 1
        assert row.moved is True
        assert row.severity == "noise"
        assert row.differs is False
        assert row.maximum.changed is False  # the move is inside atol

    def test_a_difference_just_above_atol_still_counts(self, tmp_path, base_nc):
        new = _write_nc(tmp_path / "new.nc", [[1.0, 2.0], [3.0, 4.0 + 1e-7]])

        row = netcdf_diff(base_nc, new).rows[0]

        assert row.severity == "changed"
        assert row.differs is True
        assert row.maximum.changed is True

    def test_a_changed_shape_cannot_be_compared_cell_by_cell(self, tmp_path, base_nc):
        new = _write_nc(tmp_path / "new.nc", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        row = netcdf_diff(base_nc, new).rows[0]

        assert row.max_abs_diff is None
        assert row.cells_differ is None
        assert row.shape.old == "2x2"
        assert row.shape.new == "2x3"
        assert row.shape.changed is True
        assert row.moved is True
        assert row.severity == "changed"  # an unmeasurable move is never called noise

    def test_a_nan_in_the_same_cell_on_both_sides_is_not_a_change(self, tmp_path):
        old = _write_nc(tmp_path / "old.nc", [[1.0, np.nan], [3.0, 4.0]])
        new = _write_nc(tmp_path / "new.nc", [[1.0, np.nan], [3.0, 4.0]])

        row = netcdf_diff(old, new).rows[0]

        assert row.cells_differ == 0
        assert row.moved is False
        assert row.nan.old == 1
        assert row.nan.new == 1
        assert row.nan.changed is False

    def test_an_all_nan_array_reports_no_statistics(self, tmp_path):
        old = _write_nc(tmp_path / "old.nc", [[np.nan, np.nan], [np.nan, np.nan]])
        new = _write_nc(tmp_path / "new.nc", [[np.nan, np.nan], [np.nan, np.nan]])

        row = netcdf_diff(old, new).rows[0]

        assert (row.minimum.old, row.maximum.old, row.mean.old) == (None, None, None)
        assert row.nan.old == 4
        assert row.moved is False

    def test_an_absent_old_side_leaves_every_old_statistic_unset(self, tmp_path, base_nc):
        diff = netcdf_diff(None, base_nc)
        row = diff.rows[0]

        assert diff.header_old == ()
        assert diff.header_new

        assert (row.shape.old, row.minimum.old, row.maximum.old, row.mean.old, row.nan.old) == (
            None,
            None,
            None,
            None,
            None,
        )
        assert row.shape.new == "2x2"
        assert row.shape.changed is True
        assert row.moved is True

    def test_adjacent_integers_past_the_float_mantissa_still_compare(self, tmp_path):
        # 2**53 and the next integer collide once cast to float64.
        old_values = np.array([[9007199254740992, 1], [2, 3]], dtype=np.int64)
        new_values = np.array([[9007199254740993, 1], [2, 3]], dtype=np.int64)
        paths = []
        for name, values in (("old.nc", old_values), ("new.nc", new_values)):
            path = tmp_path / name
            xr.Dataset({"count": (("lat", "lon"), values)}).to_netcdf(path)
            paths.append(path)

        row = netcdf_diff(*paths).rows[0]

        assert row.cells_differ == 1
        assert row.moved is True
        assert row.max_abs_diff == 1.0
        assert row.severity == "changed"
        assert row.differs is True

    def test_a_large_integer_moving_by_one_is_never_noise(self, tmp_path):
        paths = []
        for name, cell in (("old.nc", 1_000_000_000), ("new.nc", 1_000_000_001)):
            path = tmp_path / name
            values = np.array([[cell, 1], [2, 3]], dtype=np.int64)
            xr.Dataset({"count": (("lat", "lon"), values)}).to_netcdf(path)
            paths.append(path)

        row = netcdf_diff(*paths).rows[0]

        assert row.atol == 0.0
        assert row.severity == "changed"

    def test_an_infinite_base_value_does_not_swallow_a_real_change(self, tmp_path):
        paths = []
        for name, cell in (("old.nc", np.inf), ("new.nc", 5.0)):
            path = tmp_path / name
            values = np.array([[cell, 1.0], [2.0, 3.0]])
            xr.Dataset({"tas": (("lat", "lon"), values)}).to_netcdf(path)
            paths.append(path)

        row = netcdf_diff(*paths).rows[0]

        assert row.atol == pytest.approx(3.0 * 1e-9)  # the largest finite base magnitude
        assert row.severity == "changed"

    @pytest.mark.parametrize(
        ("old_cell", "new_cell"),
        [(np.nan, 5.0), (5.0, np.nan)],
        ids=["nan_to_number", "number_to_nan"],
    )
    def test_a_cell_moving_between_nan_and_a_number_has_no_measurable_difference(
        self, tmp_path, old_cell, new_cell
    ):
        old = _write_nc(tmp_path / "old.nc", [[old_cell, 1.0], [2.0, 3.0]])
        new = _write_nc(tmp_path / "new.nc", [[new_cell, 1.0], [2.0, 3.0]])

        row = netcdf_diff(old, new).rows[0]

        assert row.cells_differ == 1
        assert row.max_abs_diff is None  # never 0.0, which would read as no change
        assert row.max_rel_diff is None
        assert row.moved is True

    def test_a_file_that_decodes_to_too_much_is_not_reduced(self, tmp_path, base_nc, monkeypatch):
        monkeypatch.setattr("climate_ref.baseline_report.analyse.MAX_DECODED_BYTES", 8)

        diff = netcdf_diff(base_nc, base_nc)

        assert diff.note.startswith("decodes to too much to analyse")
        assert diff.rows == ()

    def test_a_file_that_is_not_netcdf_becomes_a_note(self, tmp_path, base_nc):
        broken = tmp_path / "broken.nc"
        broken.write_text("not a netcdf file at all")

        diff = netcdf_diff(base_nc, broken)

        assert diff.note.startswith("could not open")
        assert diff.rows == ()
        assert diff.header_old == ()

    def test_a_string_variable_gets_shapes_but_no_statistics(self, tmp_path):
        for path in (tmp_path / "old.nc", tmp_path / "new.nc"):
            xr.Dataset({"label": (("i",), np.array(["a", "b"], dtype=object))}).to_netcdf(path)

        row = netcdf_diff(tmp_path / "old.nc", tmp_path / "new.nc").rows[0]

        assert row.shape.old == "2"
        assert row.shape.new == "2"
        assert (row.minimum.old, row.maximum.old, row.nan.old, row.cells_differ) == (
            None,
            None,
            None,
            None,
        )
        assert row.moved is False

    def test_a_variable_added_on_one_side_only_moves(self, tmp_path, base_nc):
        new = tmp_path / "new.nc"
        xr.Dataset(
            {
                "tas": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]])),
                "pr": (("lat", "lon"), np.array([[0.0, 0.0], [0.0, 0.0]])),
            },
            coords={"lat": np.arange(2, dtype=float), "lon": np.arange(2)},
            attrs={"title": "a"},
        ).to_netcdf(new)

        rows = {row.name: row for row in netcdf_diff(base_nc, new).rows}

        assert sorted(rows) == ["pr", "tas"]
        assert rows["pr"].moved is True
        assert rows["pr"].shape.old is None
        assert rows["tas"].moved is False


class TestAnalyseNetcdf:
    def test_a_netcdf_file_is_fetched_and_analysed(self, tmp_path, base_nc):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        store.fetch.side_effect = lambda digest, dest: dest.write_bytes(base_nc.read_bytes())
        entry = NativeEntry(sha256="1" * 64, size=base_nc.stat().st_size)
        report = _report([_file_change("out.nc", entry, evolve(entry, sha256="2" * 64))])

        file = analyse(report, store, fetch=True, workdir=tmp_path).cases[0].files[0]

        assert file.netcdf.note is None
        assert [row.name for row in file.netcdf.rows] == ["tas"]
        assert file.text is None

    def test_an_oversized_netcdf_file_is_not_fetched(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        entry = NativeEntry(sha256="1" * 64, size=NETCDF_FETCH_BYTES + 1)
        report = _report([_file_change("out.nc", None, entry)])

        file = analyse(report, store, fetch=True, workdir=tmp_path).cases[0].files[0]

        assert "too large to analyse" in file.netcdf.note
        store.fetch.assert_not_called()

    def test_fetching_disabled_leaves_a_note(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        report = _report([_file_change("out.nc", None, NativeEntry(sha256="1" * 64, size=10))])

        file = analyse(report, store, fetch=False, workdir=tmp_path).cases[0].files[0]

        assert file.netcdf.note == "fetching disabled"


def _committed(**kwargs):
    """Build one committed artefact change, defaulting to a changed ``series.json``."""
    fields = {
        "name": "series.json",
        "rel_path": "packages/climate-ref-example/tests/test-data/diag/case/regression/series.json",
        "old": "a" * 64,
        "new": "b" * 64,
        "old_text": '{"value": 1}',
        "new_text": '{"value": 2}',
    }
    fields.update(kwargs)
    return CommittedChange(**fields)


class TestCommittedDiff:
    def test_json_is_pretty_printed_before_diffing(self):
        diff = committed_diff(_committed())

        assert diff.note is None
        # A minified bundle would otherwise diff as one unreadable line.
        assert '-  "value": 1' in [line.text for line in diff.lines]
        assert '+  "value": 2' in [line.text for line in diff.lines]

    def test_an_added_artefact_diffs_against_nothing(self):
        diff = committed_diff(_committed(old=None, old_text=None))

        assert diff.note is None
        assert diff.lines[0].text == "--- (absent)"

    def test_an_unreadable_base_side_leaves_a_note(self):
        diff = committed_diff(_committed(old_text=None))

        assert diff.note == "could not read the base version from git"

    def test_an_unreadable_head_side_leaves_a_note(self):
        diff = committed_diff(_committed(new_text=None))

        assert diff.note == "could not read the working tree version"

    def test_the_diff_reaches_the_analysed_case(self, tmp_path):
        store = MagicMock(spec=NativeStore)
        store.url = "https://store"
        store.root = None
        report = _report([], committed=[_committed()])

        case = analyse(report, store, fetch=False, workdir=tmp_path).cases[0]

        assert [item.change.name for item in case.committed] == ["series.json"]
        assert case.committed[0].text.note is None


def _manifest(native, committed=None):
    """Build a manifest carrying only what a tree is built from."""
    return Manifest(
        schema=SCHEMA_VERSION,
        test_case_version=1,
        diagnostic_version=1,
        committed=committed or {},
        native=native,
    )


class TestBaselineTree:
    def test_directories_are_emitted_once_before_their_files(self):
        entry = NativeEntry(sha256="1" * 64, size=10)
        added = NativeEntry(sha256="2" * 64, size=10)
        case = _report(
            [_file_change("plots/b.png", None, added)],
            base=_manifest({"plots/a.png": entry}),
            head=_manifest({"plots/a.png": entry, "plots/b.png": added}),
        ).cases[0]

        assert [(node.name, node.depth, node.is_dir, node.status) for node in baseline_tree(case)] == [
            ("plots", 0, True, None),
            ("a.png", 1, False, None),
            ("b.png", 1, False, "added"),
        ]

    def test_a_nested_directory_is_only_reopened_when_it_changes(self):
        entry = NativeEntry(sha256="1" * 64, size=10)
        case = _report(
            [],
            head=_manifest({"a/b/one.nc": entry, "a/c/two.nc": entry, "top.png": entry}),
        ).cases[0]

        assert [(node.name, node.depth, node.is_dir) for node in baseline_tree(case)] == [
            ("a", 0, True),
            ("b", 1, True),
            ("one.nc", 2, False),
            ("c", 1, True),
            ("two.nc", 2, False),
            ("top.png", 0, False),
        ]

    def test_a_removed_file_keeps_its_place_in_the_listing(self):
        entry = NativeEntry(sha256="1" * 64, size=10)
        case = _report(
            [_file_change("gone.png", entry, None)],
            base=_manifest({"gone.png": entry}),
            head=_manifest({}),
        ).cases[0]

        assert [(node.name, node.status, node.size) for node in baseline_tree(case)] == [
            ("gone.png", "removed", 10)
        ]

    def test_a_case_with_no_manifests_has_no_tree(self):
        assert baseline_tree(_report([]).cases[0]) == ()

    def test_the_committed_bundle_is_listed_under_regression(self):
        entry = NativeEntry(sha256="1" * 64, size=10)
        case = _report(
            [],
            committed=[_committed()],
            base=_manifest({"plot.png": entry}, {"series.json": "a" * 64}),
            head=_manifest({"plot.png": entry}, {"series.json": "b" * 64}),
        ).cases[0]

        assert [(node.name, node.depth, node.is_dir, node.status) for node in baseline_tree(case)] == [
            ("plot.png", 0, False, None),
            ("regression", 0, True, None),
            ("series.json", 1, False, "changed"),
        ]

    def test_a_committed_artefact_has_no_size(self):
        case = _report([], head=_manifest({}, {"series.json": "a" * 64})).cases[0]

        assert baseline_tree(case)[-1].size is None
