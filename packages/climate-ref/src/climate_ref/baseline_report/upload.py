"""
Push a rendered report into the report store.

The store is a plain object store with no directory semantics,
so every file is uploaded under an explicit key.
The content type has to be set per object, or a browser will download the page instead of rendering it.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from climate_ref_core.regression.report_store import ReportStore

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}
"""
Content types for the file kinds a report is made of.

A deliberate allowlist rather than :mod:`mimetypes`, whose answers vary with the host's registry,
and which would not add the charset a browser needs on the text types.
"""

DEFAULT_CONTENT_TYPE = "application/octet-stream"
"""Served for anything else, which a browser will offer as a download."""


def content_type_for(path: Path) -> str:
    """
    Return the content type a report file should be served with.

    Parameters
    ----------
    path
        The file, which is classified by its extension.

    Returns
    -------
    :
        The MIME type, or :data:`DEFAULT_CONTENT_TYPE` for an unrecognised extension.
    """
    return CONTENT_TYPES.get(path.suffix.lower(), DEFAULT_CONTENT_TYPE)


def upload_site(out_dir: Path, store: ReportStore, prefix: str) -> str:
    """
    Upload every file under ``out_dir`` as ``prefix/<relative path>``.

    Parameters
    ----------
    out_dir
        The rendered site.
    store
        The store to upload into.
    prefix
        The key prefix the report is published under, for example ``912/0c7e1d4abc12``.
        Validated by :meth:`~climate_ref_core.regression.report_store.ReportStore.put`.

    Returns
    -------
    :
        The URL of the report's index page, whether or not that page exists.
    """
    count = 0
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        key = str(PurePosixPath(prefix, *path.relative_to(out_dir).parts))
        store.put(key, path, content_type_for(path))
        count += 1
    logger.info(f"Uploaded {count} report file(s) under {prefix}")
    return store.url_for(f"{prefix}/index.html")
