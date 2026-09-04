"""Tests for the static HTML the report is written as."""

from html.parser import HTMLParser

import pytest

from climate_ref.baseline_report.analyse import AnalysedCase, AnalysedFile, AnalysedReport, DiffLine, TextDiff
from climate_ref.baseline_report.collect import CaseChange, FileChange, FileKind, Report
from climate_ref.baseline_report.render import render_case, render_index, write_site
from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest, NativeEntry

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


def _analysed_file(name, kind, old, new, text=None) -> AnalysedFile:
    """Build one analysed file with URLs derived from its entries."""
    change = FileChange(name=name, old=old, new=new, kind=kind)
    return AnalysedFile(
        change=change,
        old_url=f"{STORE_URL}/{old.sha256}" if old else None,
        new_url=f"{STORE_URL}/{new.sha256}" if new else None,
        text=text,
        size_delta=new.size - old.size if old and new else None,
    )


def _case(files, *, label="example/diag/case", base=None, head=None) -> AnalysedCase:
    """Build one analysed case with tallied counts."""
    counts = {kind.value: {"added": 0, "changed": 0, "removed": 0} for kind in FileKind}
    for file in files:
        counts[file.change.kind.value][file.change.status] += 1
    change = CaseChange(
        label=label,
        slug=label,
        rel_path=f"packages/climate-ref-{label.split('/')[0]}/tests/test-data/manifest.json",
        base=base,
        head=head,
        files=tuple(file.change for file in files),
        committed=("series.json",),
        metadata=("test_case_version: 3 -> 4",),
    )
    return AnalysedCase(
        change=change,
        files=tuple(files),
        counts=counts,
        images=tuple(f for f in files if f.change.kind is FileKind.IMAGE),
        texts=tuple(f for f in files if f.change.kind is FileKind.TEXT),
        binaries=tuple(f for f in files if f.change.kind in (FileKind.NETCDF, FileKind.OTHER)),
        back_link="/".join([*[".."] * len(label.split("/")), "index.html"]),
    )


def _report(cases) -> AnalysedReport:
    """Wrap analysed cases in a report."""
    return AnalysedReport(
        report=Report(base_ref="origin/main", head_sha="a" * 40, cases=tuple(c.change for c in cases)),
        store_url=STORE_URL,
        cases=tuple(cases),
    )


@pytest.fixture
def changed_image_case():
    """A case whose single image changed."""
    return _case(
        [_analysed_file("plot.png", FileKind.IMAGE, _entry("1"), _entry("2", 20))],
        base=_manifest(3),
        head=_manifest(4),
    )


class TestIndex:
    def test_one_row_per_case(self):
        report = _report(
            [
                _case([], label="example/diag/a", base=_manifest(1), head=_manifest(2)),
                _case([], label="pmp/diag/b", base=_manifest(1), head=_manifest(2)),
            ]
        )

        html = render_index(report)

        assert html.count("<tr>") == 3  # one header row plus one per case
        assert "example/diag/a" in html
        assert "pmp/diag/b" in html

    def test_every_link_ends_in_index_html(self):
        report = _report([_case([], base=_manifest(1), head=_manifest(2))])

        assert _hrefs(render_index(report)) == ["example/diag/case/index.html"]

    def test_versions_column(self):
        report = _report([_case([], base=_manifest(3), head=_manifest(4))])

        assert "v3 -&gt; v4" in render_index(report)

    def test_an_empty_report_says_so(self):
        html = render_index(_report([]))

        assert "No baseline manifests changed" in html
        assert "<tbody>" not in html


class TestCasePage:
    def test_a_changed_image_renders_two_images(self, changed_image_case):
        report = _report([changed_image_case])

        images = _tags(render_case(report, changed_image_case), "img")

        assert len(images) == 2
        assert all(image["src"].startswith(STORE_URL) for image in images)

    def test_an_added_image_renders_one_image_and_a_placeholder(self):
        case = _case([_analysed_file("plot.png", FileKind.IMAGE, None, _entry("2"))])
        report = _report([case])

        html = render_case(report, case)

        assert len(_tags(html, "img")) == 1
        assert 'class="absent"' in html

    def test_a_text_diff_renders_one_span_per_line(self):
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
        case = _case([_analysed_file("series.csv", FileKind.TEXT, _entry("1"), _entry("2"), text=diff)])
        report = _report([case])

        html = render_case(report, case)
        spans = [
            attrs["class"]
            for attrs in _tags(html, "span")
            if attrs.get("class") in {"header", "hunk", "remove", "add"}
        ]

        assert spans == ["header", "hunk", "remove", "add"]

    def test_a_note_replaces_the_diff(self):
        diff = TextDiff(lines=(), note="fetching disabled", elided=0)
        case = _case([_analysed_file("series.csv", FileKind.TEXT, None, _entry("2"), text=diff)])
        report = _report([case])

        html = render_case(report, case)

        assert "fetching disabled" in html
        assert '<pre class="diff">' not in html

    def test_elided_lines_are_reported(self):
        diff = TextDiff(lines=(DiffLine(kind="add", text="+a"),), note=None, elided=7)
        case = _case([_analysed_file("series.csv", FileKind.TEXT, None, _entry("2"), text=diff)])
        report = _report([case])

        assert "7 further diff line(s) elided" in render_case(report, case)

    def test_netcdf_renders_as_a_row(self):
        case = _case([_analysed_file("out.nc", FileKind.NETCDF, _entry("1"), None)])
        report = _report([case])

        html = render_case(report, case)

        assert "out.nc" in html
        assert "was 10 B" in html
        assert not _tags(html, "img")

    def test_the_back_link_matches_the_slug_depth(self):
        case = _case([], label="pmp/diag/one")
        report = _report([case])

        assert "../../../index.html" in _hrefs(render_case(report, case))

    def test_a_shallow_slug_gets_a_shallow_back_link(self):
        case = _case([], label="pmp")
        report = _report([case])

        assert "../index.html" in _hrefs(render_case(report, case))

    def test_a_changed_file_shows_its_signed_size_delta(self, changed_image_case):
        assert "(+10)" in render_case(_report([changed_image_case]), changed_image_case)

    def test_a_text_diff_names_both_digests(self):
        diff = TextDiff(lines=(DiffLine(kind="add", text="+a"),), note=None, elided=0)
        case = _case([_analysed_file("series.csv", FileKind.TEXT, _entry("1"), _entry("2"), text=diff)])

        html = render_case(_report([case]), case)

        assert "1" * 12 in html
        assert "2" * 12 in html

    def test_links_are_internal_index_pages_or_store_blobs(self, changed_image_case):
        report = _report([changed_image_case])

        for href in _hrefs(render_case(report, changed_image_case)):
            assert href.endswith("index.html") or href.startswith(STORE_URL)

    def test_metadata_and_committed_are_listed(self, changed_image_case):
        html = render_case(_report([changed_image_case]), changed_image_case)

        assert "test_case_version: 3 -&gt; 4" in html
        assert "series.json" in html


class TestWriteSite:
    def test_writes_an_index_and_a_page_per_case(self, tmp_path, changed_image_case):
        report = _report([changed_image_case])

        index = write_site(report, tmp_path / "out")

        assert index == tmp_path / "out" / "index.html"
        assert index.exists()
        assert (tmp_path / "out" / "example" / "diag" / "case" / "index.html").exists()

    def test_an_empty_report_still_writes_an_index(self, tmp_path):
        index = write_site(_report([]), tmp_path / "out")

        assert index.exists()
