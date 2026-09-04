"""Tests for pushing a rendered report into the report store."""

from pathlib import Path

import pytest
from attrs import define, field

from climate_ref.baseline_report.upload import DEFAULT_CONTENT_TYPE, content_type_for, upload_site

PREFIX = "912/0c7e1d4abc12"


@define
class RecordingStore:
    """A report store that records what was put rather than storing it."""

    url: str = "https://reports.example"
    puts: list[tuple[str, Path, str]] = field(factory=list)

    def put(self, key: str, path: Path, content_type: str) -> str:
        self.puts.append((key, path, content_type))
        return self.url_for(key)

    def url_for(self, key: str) -> str:
        return f"{self.url}/{key}"


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A three-file site with one page nested a directory deep."""
    out = tmp_path / "site"
    (out / "example" / "diag").mkdir(parents=True)
    (out / "index.html").write_text("<p>index</p>", encoding="utf-8")
    (out / "report.css").write_text("body {}", encoding="utf-8")
    (out / "example" / "diag" / "index.html").write_text("<p>case</p>", encoding="utf-8")
    return out


@pytest.mark.parametrize(
    "name, expected",
    [
        ("index.html", "text/html; charset=utf-8"),
        ("report.CSS", "text/css; charset=utf-8"),
        ("report.js", "text/javascript; charset=utf-8"),
        ("plot.png", "image/png"),
        ("plot.svg", "image/svg+xml"),
        ("blob.bin", DEFAULT_CONTENT_TYPE),
        ("no-extension", DEFAULT_CONTENT_TYPE),
    ],
)
def test_content_type_for(name, expected):
    assert content_type_for(Path(name)) == expected


class TestUploadSite:
    def test_every_file_lands_under_the_prefix(self, site):
        store = RecordingStore()

        upload_site(site, store, PREFIX)

        assert sorted(key for key, _, _ in store.puts) == [
            f"{PREFIX}/example/diag/index.html",
            f"{PREFIX}/index.html",
            f"{PREFIX}/report.css",
        ]

    def test_content_types_are_set(self, site):
        store = RecordingStore()

        upload_site(site, store, PREFIX)

        by_key = {key: content_type for key, _, content_type in store.puts}
        assert by_key[f"{PREFIX}/index.html"] == "text/html; charset=utf-8"
        assert by_key[f"{PREFIX}/report.css"] == "text/css; charset=utf-8"

    def test_returns_the_index_url(self, site):
        store = RecordingStore()

        assert upload_site(site, store, PREFIX) == f"https://reports.example/{PREFIX}/index.html"

    def test_directories_are_not_uploaded(self, site):
        store = RecordingStore()

        upload_site(site, store, PREFIX)

        assert all(path.is_file() for _, path, _ in store.puts)
