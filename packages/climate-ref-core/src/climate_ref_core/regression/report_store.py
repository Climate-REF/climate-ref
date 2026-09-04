"""
Named-key store for hosted baseline diff reports.

A diff report is a small static site, so its objects are addressed by path rather than by digest.
That is the one thing it does not share with :class:`~climate_ref_core.regression.store.NativeStore`,
which is content-addressed and must stay that way.
Everything below the key (where the store lives, how it authenticates, preflight, putting a file)
is the shared transport in :mod:`~climate_ref_core.regression.store`.

Reports live in their own public bucket, served at ``{url}/{key}``.

Write credentials are resolved from the following in order:

- ``REF_REPORT_STORE_ACCESS_KEY_ID`` / ``REF_REPORT_STORE_SECRET_ACCESS_KEY``,
- ``REF_REPORT_STORE_PROFILE``,
- boto3's default chain
"""

import re
import shutil
from pathlib import Path

from attrs import field, frozen
from loguru import logger

from climate_ref_core.paths import safe_path

from .store import (
    S3WriteConfig,
    _local_root,
    _preflight_store,
    _StoreConfigProtocol,
    _write_config_from_env,
)

_KEY_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")


def _validate_key(key: str, base: Path | None = None) -> Path:
    """
    Check a report key is a safe relative path and return the path it names.

    The character class is the object-key rule, narrower than a filesystem path because the key
    also has to survive a URL. Containment is left to :func:`safe_path`, which is what the rest
    of the regression package validates paths with.

    Parameters
    ----------
    key
        The key to check, for example ``912/0c7e1d4abc12/index.html``.
    base
        The local store root the key is joined onto, or ``None`` for a remote store,
        where there is no directory to escape.

    Returns
    -------
    :
        ``base / key`` when a base is given, otherwise the key as a relative path.

    Raises
    ------
    ValueError
        If the key uses characters outside ``[A-Za-z0-9._/-]``, is empty, absolute, or
        contains a ``..`` segment, or if it escapes ``base``.
    """
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError(
            f"Unsafe report store key {key!r}: may only contain letters, digits, dot, "
            "underscore, hyphen and forward slash."
        )
    return safe_path(key, base, label="report store key")


@frozen
class ReportStore:
    """
    Named-key store for hosted baseline diff reports.

    A local store (a ``file://`` URL or a bare path) writes under its root and needs no credentials.
    A remote store writes through an :class:`S3WriteConfig` and is served at ``{url}/{key}``.

    Parameters
    ----------
    url
        Where reports are served from: an ``http(s)://`` base URL, a ``file:///absolute/path``
        URL, or a bare filesystem path.
    write
        The S3 endpoint, bucket and credentials a remote store writes with.
        ``None`` leaves a remote store read-only.
        A local store ignores it and is always writable.
    """

    url: str
    write: S3WriteConfig | None = None
    root: Path | None = field(init=False)
    """The local store root, or ``None`` when this store is remote."""

    @root.default
    def _resolve_root(self) -> Path | None:
        """Resolve the local root once at construction, which also validates the URL."""
        return _local_root(self.url)

    def url_for(self, key: str) -> str:
        """
        Return the public URL a key is served at.

        Parameters
        ----------
        key
            The key, relative and slash-separated.

        Returns
        -------
        :
            An ``http(s)://`` URL for a remote store, or a ``file://`` URL for a local one.

        Raises
        ------
        ValueError
            If the key is not a safe relative path.
        """
        root = self.root
        if root is not None:
            return _validate_key(key, root).absolute().as_uri()
        _validate_key(key)
        return f"{self.url.rstrip('/')}/{key}"

    def put(self, key: str, path: Path, content_type: str) -> str:
        """
        Store ``path`` under ``key``, overwriting whatever was there.

        Reports are rewritten whenever a pull request is re-minted, so a put always replaces.

        Parameters
        ----------
        key
            The key to store under, relative and slash-separated.
        path
            The local file to store.
        content_type
            The MIME type the object is served with. A browser will not render an
            uploaded page without it, since R2 defaults to a binary type.

        Returns
        -------
        :
            The URL the key is served at.

        Raises
        ------
        ValueError
            If the key is not a safe relative path.
        NotImplementedError
            If this is an anonymous remote store, which cannot write.
        """
        root = self.root
        if root is not None:
            dest = _validate_key(key, root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(dest))
        else:
            _validate_key(key)
            write = self.write
            if write is None:
                raise NotImplementedError(
                    f"Report store {self.url} is a public-read store, so put() is not supported. "
                    "Upload against a local path or a credentialed remote store."
                )
            write.client().upload_file(str(path), write.bucket, key, ExtraArgs={"ContentType": content_type})
        logger.debug(f"ReportStore.put: {path} -> {key}")
        return self.url_for(key)

    def preflight(self) -> None:
        """
        Verify the store is reachable and usable before uploading to it.

        Raises
        ------
        NativeStoreUnavailableError
            If the store cannot be reached or used, with an operator-facing message.
        """
        _preflight_store(self.root, self.write, "Report store")


def build_report_store(config: _StoreConfigProtocol, *, writable: bool) -> ReportStore:
    """
    Build a :class:`ReportStore` from a report-store config object.

    With ``writable=False`` the returned store is anonymous and credential-free.
    With ``writable=True`` and a remote URL the S3 endpoint and bucket come from the config,
    and authentication is read from the environment
    (``REF_REPORT_STORE_ACCESS_KEY_ID`` / ``REF_REPORT_STORE_SECRET_ACCESS_KEY``,
    else ``REF_REPORT_STORE_PROFILE``, else boto3's default chain),
    so secrets never live in the persisted config.
    A local store is always readable and writable, so ``writable`` makes no difference to it.

    Parameters
    ----------
    config
        A config object providing ``url``, ``s3_endpoint_url`` and ``bucket``.
        Typically ``app_config.report_store``.
    writable
        Whether the store must be able to write.

    Returns
    -------
    :
        The configured store.

    Raises
    ------
    ValueError
        If the URL scheme is unrecognised, or a writable remote store is requested
        without an S3 endpoint / bucket configured.
    """
    store = ReportStore(url=config.url)
    if not writable or store.root is not None:
        # A local store is already writable, and a read-only store needs no credentials.
        return store
    return ReportStore(
        url=config.url,
        write=_write_config_from_env(config.s3_endpoint_url, config.bucket, "REF_REPORT_STORE"),
    )
