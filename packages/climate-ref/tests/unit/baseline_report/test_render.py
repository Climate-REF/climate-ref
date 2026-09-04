"""Tests for the static HTML the report is written as."""

from html.parser import HTMLParser
from unittest.mock import MagicMock

import pytest
from attrs import evolve

from climate_ref.baseline_report.analyse import (
    AnalysedReport,
    DiffLine,
    NetcdfDiff,
    StatRow,
    TextDiff,
    analyse,
)
from climate_ref.baseline_report.collect import CaseChange, FileChange, FileKind, Report, classify
from climate_ref.baseline_report.render import render_case, render_index, write_site
from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest, NativeEntry
from climate_ref_core.regression.store import NativeStore

STORE_URL = "https://store.example"


class Collector(HTMLParser):
    """Collect the tags and attributes of a rendered page."""

    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _parse(html: str) -> Collector:
    """Parse a page and return its collected tags."""
    collector = Collector()
    collector.feed(html)
    return collector


def _tags(html: str, name: str) -> list[dict[str, str]]:
    """Return the attributes of every ``name`` tag in a page."""
    return [attrs for tag, attrs in _parse(html).tags if tag == name]


def _hrefs(html: str) -> list[str]:
    """Return every href in a page."""
    return [attrs["href"] for _, attrs in _parse(html).tags if "href" in attrs]


def _manifest(version: int) -> Manifest:
    """Build a manifest carrying only the fields the templates read."""
    return Manifest(
        schema=SCHEMA_VERSION,
        test_case_version=version,
        diagnostic_version=1,
        committed={},
        native={},
    )


def _entry(char: str, size: int = 10) -> NativeEntry:
    """Build a manifest entry from a single repeated hex character."""
    return NativeEntry(sha256=char * 64, size=size)


def _change(name: str, old: NativeEntry | None, new: NativeEntry | None) -> FileChange:
    """Build one file change with the kind its name implies."""
    return FileChange(name=name, old=old, new=new, kind=classify(name))


def _case_change(changes, label="example/diag/case", base=None, head=None) -> CaseChange:
    """Build one collected case."""
    return CaseChange(
        label=label,
        rel_path=f"packages/climate-ref-{label.split('/')[0]}/tests/test-data/manifest.json",
        base=base,
        head=head,
        files=tuple(changes),
        committed=("series.json",),
        metadata=("test_case_version: 3 -> 4",),
    )


def _analysed(cases, tmp_path, diffs=None) -> AnalysedReport:
    """
    Analyse collected cases with a store that is never read.

    Running the real :func:`analyse` keeps these tests pinned to the counts, partitions and
    back links the pages are actually built from. ``diffs`` replaces the placeholder note on a
    named file, which is how a specific diff shape is put in front of the templates.
    """
    store = MagicMock(spec=NativeStore)
    store.url = STORE_URL
    store.root = None
    report = analyse(
        Report(base_ref="origin/main", head_sha="a" * 40, cases=tuple(cases)),
        store,
        fetch=False,
        workdir=tmp_path,
    )
    if not diffs:
        return report

    def _replace(file):
        """Swap in the fixture diff for a named file, leaving the rest as analysed."""
        found = diffs.get(file.change.name)
        if found is None:
            return file
        if isinstance(found, NetcdfDiff):
            return evolve(file, netcdf=found)
        return evolve(file, text=found)

    return evolve(
        report,
        cases=tuple(
            evolve(
                case,
                files=tuple(_replace(file) for file in case.files),
                texts=tuple(_replace(file) for file in case.texts),
                netcdfs=tuple(_replace(file) for file in case.netcdfs),
            )
            for case in report.cases
        ),
    )


@pytest.fixture
def changed_image_case(tmp_path):
    """A report whose single case has one changed image."""
    return _analysed(
        [
            _case_change(
                [_change("plot.png", _entry("1"), _entry("2", 20))], base=_manifest(3), head=_manifest(4)
            )
        ],
        tmp_path,
    )


class TestIndex:
    def test_one_row_per_case(self, tmp_path):
        report = _analysed(
            [
                _case_change([], label="example/diag/a", base=_manifest(1), head=_manifest(2)),
                _case_change([], label="pmp/diag/b", base=_manifest(1), head=_manifest(2)),
            ],
            tmp_path,
        )

        html = render_index(report)

        assert html.count("<tr>") == 3  # one header row plus one per case
        assert "example/diag/a" in html
        assert "pmp/diag/b" in html

    def test_every_link_ends_in_index_html(self, tmp_path):
        report = _analysed([_case_change([], base=_manifest(1), head=_manifest(2))], tmp_path)

        assert _hrefs(render_index(report)) == ["example/diag/case/index.html"]

    def test_versions_column(self, tmp_path):
        report = _analysed([_case_change([], base=_manifest(3), head=_manifest(4))], tmp_path)

        assert "v3 -&gt; v4" in render_index(report)

    def test_a_column_header_per_kind(self, tmp_path):
        report = _analysed([_case_change([], base=_manifest(1), head=_manifest(2))], tmp_path)

        headers = render_index(report)

        for kind in FileKind:
            assert f"<th>{kind.value}</th>" in headers

    def test_counts_appear_per_kind(self, tmp_path):
        report = _analysed(
            [
                _case_change(
                    [_change("a.png", None, _entry("1")), _change("b.nc", _entry("2"), None)],
                    base=_manifest(1),
                    head=_manifest(2),
                )
            ],
            tmp_path,
        )

        html = render_index(report)

        assert '<span class="added">+1</span>' in html
        assert '<span class="removed">-1</span>' in html

    def test_an_empty_report_says_so(self, tmp_path):
        html = render_index(_analysed([], tmp_path))

        assert "No baseline manifests changed" in html
        assert "<tbody>" not in html


class TestCasePage:
    def test_a_changed_image_renders_two_images(self, changed_image_case):
        images = _tags(render_case(changed_image_case, changed_image_case.cases[0]), "img")

        assert len(images) == 2
        assert all(image["src"].startswith(STORE_URL) for image in images)

    def test_an_added_image_renders_one_image_and_a_placeholder(self, tmp_path):
        report = _analysed([_case_change([_change("plot.png", None, _entry("2"))])], tmp_path)

        html = render_case(report, report.cases[0])

        assert len(_tags(html, "img")) == 1
        assert 'class="absent"' in html

    def test_a_text_diff_renders_one_span_per_line(self, tmp_path):
        diff = TextDiff(
            lines=(
                DiffLine(kind="header", text="--- old"),
                DiffLine(kind="hunk", text="@@ -1 +1 @@"),
                DiffLine(kind="remove", text="-a"),
                DiffLine(kind="add", text="+b"),
            ),
            note=None,
            elided=0,
        )
        report = _analysed(
            [_case_change([_change("series.csv", _entry("1"), _entry("2"))])],
            tmp_path,
            diffs={"series.csv": diff},
        )

        html = render_case(report, report.cases[0])
        spans = [
            attrs["class"]
            for attrs in _tags(html, "span")
            if attrs.get("class") in {"header", "hunk", "remove", "add"}
        ]

        assert spans == ["header", "hunk", "remove", "add"]

    def test_a_note_replaces_the_diff(self, tmp_path):
        report = _analysed([_case_change([_change("series.csv", None, _entry("2"))])], tmp_path)

        html = render_case(report, report.cases[0])

        assert "fetching disabled" in html
        assert '<pre class="diff">' not in html

    def test_elided_lines_are_reported(self, tmp_path):
        diff = TextDiff(lines=(DiffLine(kind="add", text="+a"),), note=None, elided=7)
        report = _analysed(
            [_case_change([_change("series.csv", None, _entry("2"))])],
            tmp_path,
            diffs={"series.csv": diff},
        )

        assert "7 further diff line(s) elided" in render_case(report, report.cases[0])

    def test_netcdf_renders_as_a_row(self, tmp_path):
        report = _analysed([_case_change([_change("out.nc", _entry("1"), None)])], tmp_path)

        html = render_case(report, report.cases[0])

        assert "out.nc" in html
        assert "was 10 B" in html
        assert not _tags(html, "img")

    def test_the_back_link_matches_the_label_depth(self, tmp_path):
        report = _analysed([_case_change([], label="pmp/diag/one")], tmp_path)

        assert "../../../index.html" in _hrefs(render_case(report, report.cases[0]))

    def test_a_shallow_label_gets_a_shallow_back_link(self, tmp_path):
        report = _analysed([_case_change([], label="pmp")], tmp_path)

        assert "../index.html" in _hrefs(render_case(report, report.cases[0]))

    def test_a_changed_file_shows_its_signed_size_delta(self, changed_image_case):
        assert "(+10)" in render_case(changed_image_case, changed_image_case.cases[0])

    def test_a_text_diff_names_both_digests(self, tmp_path):
        diff = TextDiff(lines=(DiffLine(kind="add", text="+a"),), note=None, elided=0)
        report = _analysed(
            [_case_change([_change("series.csv", _entry("1"), _entry("2"))])],
            tmp_path,
            diffs={"series.csv": diff},
        )

        html = render_case(report, report.cases[0])

        assert "1" * 12 in html
        assert "2" * 12 in html

    def test_links_are_internal_index_pages_or_store_blobs(self, changed_image_case):
        for href in _hrefs(render_case(changed_image_case, changed_image_case.cases[0])):
            assert href.endswith("index.html") or href.startswith(STORE_URL)

    def test_metadata_and_committed_are_listed(self, changed_image_case):
        html = render_case(changed_image_case, changed_image_case.cases[0])

        assert "test_case_version: 3 -&gt; 4" in html
        assert "series.json" in html


class TestWriteSite:
    def test_writes_an_index_and_a_page_per_case(self, tmp_path, changed_image_case):
        index = write_site(changed_image_case, tmp_path / "out")

        assert index == tmp_path / "out" / "index.html"
        assert index.exists()
        assert (tmp_path / "out" / "example" / "diag" / "case" / "index.html").exists()

    def test_an_empty_report_still_writes_an_index(self, tmp_path):
        index = write_site(_analysed([], tmp_path), tmp_path / "out")

        assert index.exists()


def _stat_row(name, *, moved, **overrides):
    """Build a stats row with every field set, so a template cannot pass on a missing one."""
    fields = dict(
        shape_old="2x2",
        shape_new="2x2",
        min_old=1.0,
        min_new=1.0,
        max_old=4.0,
        max_new=4.0,
        mean_old=2.5,
        mean_new=2.5,
        nan_old=0,
        nan_new=0,
        max_abs_diff=0.0,
        max_rel_diff=0.0,
        cells_differ=0,
    )
    fields.update(overrides)
    return StatRow(name=name, moved=moved, **fields)


def _netcdf_case(tmp_path, diff, name="out.nc"):
    """A report whose single case has one changed NetCDF file carrying ``diff``."""
    return _analysed(
        [_case_change([_change(name, _entry("1"), _entry("2", 20))], base=_manifest(3), head=_manifest(4))],
        tmp_path,
        diffs={name: diff},
    )


class TestNetcdfBlock:
    def test_only_the_moved_row_is_shaded(self, tmp_path):
        diff = NetcdfDiff(
            header=(DiffLine(kind="add", text="+title: b"),),
            rows=(
                _stat_row("tas", moved=True, max_abs_diff=0.5, max_rel_diff=0.125, cells_differ=1),
                _stat_row("pr", moved=False),
            ),
            note=None,
        )
        report = _netcdf_case(tmp_path, diff)

        html = render_case(report, report.cases[0])

        assert html.count('<tr class="moved">') == 1
        assert [attrs.get("class") for attrs in _tags(html, "table")] == ["stats"]
        assert "0.5" in html
        assert "0.125" in html

    def test_a_note_replaces_the_table(self, tmp_path):
        report = _netcdf_case(tmp_path, NetcdfDiff(header=(), rows=(), note="could not open: boom"))

        html = render_case(report, report.cases[0])

        assert "could not open: boom" in html
        assert "<table" not in html

    def test_an_absent_statistic_renders_as_an_ascii_hyphen(self, tmp_path):
        diff = NetcdfDiff(
            header=(),
            rows=(
                _stat_row(
                    "tas",
                    moved=True,
                    shape_old=None,
                    min_old=None,
                    max_old=None,
                    mean_old=None,
                    nan_old=None,
                    max_abs_diff=None,
                    max_rel_diff=None,
                    cells_differ=None,
                ),
            ),
            note=None,
        )
        report = _netcdf_case(tmp_path, diff)

        html = render_case(report, report.cases[0])

        assert "- -> 2x2" in html
        assert "<td>-</td>" in html

    def test_the_page_carries_no_dash_that_is_not_ascii(self, tmp_path):
        diff = NetcdfDiff(header=(), rows=(_stat_row("tas", moved=False),), note=None)
        report = _netcdf_case(tmp_path, diff)

        html = render_case(report, report.cases[0])

        assert "\u2013" not in html
        assert "\u2014" not in html

    def test_a_wide_table_is_wrapped_so_it_can_scroll(self, tmp_path):
        diff = NetcdfDiff(header=(), rows=(_stat_row("tas", moved=False),), note=None)
        report = _netcdf_case(tmp_path, diff, name="a" * 120 + ".nc")

        html = render_case(report, report.cases[0])

        assert '<div class="scroll-x">' in html
        assert "a" * 120 in html

    def test_matching_headers_say_so_rather_than_showing_an_empty_diff(self, tmp_path):
        diff = NetcdfDiff(header=(), rows=(_stat_row("tas", moved=False),), note=None)
        report = _netcdf_case(tmp_path, diff)

        html = render_case(report, report.cases[0])

        assert "The headers match." in html
        assert '<pre class="diff">' not in html
