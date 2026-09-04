"""
Named-key store for hosted baseline diff reports.

A diff report is a small static site, so its objects are addressed by path rather than by digest.
That makes it the opposite of :class:`~climate_ref_core.regression.store.NativeStore`, which is
content-addressed and must stay that way, so this is a sibling class rather than a new method on it.

Reports live in their own public bucket, served at ``{url}/{key}``.
The bucket is separate from the baselines bucket because an R2 token cannot be scoped to a prefix,
so a token that can write reports must not also be able to overwrite baseline blobs.

Write credentials are resolved from ``REF_REPORT_STORE_ACCESS_KEY_ID`` /
``REF_REPORT_STORE_SECRET_ACCESS_KEY``, then ``REF_REPORT_STORE_PROFILE``, then boto3's default
chain. They are never read from the persisted config.
"""

import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Protocol

from attrs import field, frozen
from loguru import logger

from .store import (
    _AUTH_REJECTED_STATUSES,
    _HTTP_FORBIDDEN,
    _HTTP_NOT_FOUND,
    _PREFLIGHT_PROBE_KEY,
    NativeStoreUnavailableError,
    S3WriteConfig,
    _http_status,
    _local_root,
)

_KEY_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")


def _validate_key(key: str) -> str:
    """
    Check a report key is a safe relative path and return it.

    A key becomes a path under the local root, so a key that escapes the root or carries
    shell-hostile characters is rejected before it is ever joined.

    Parameters
    ----------
    key
        The key to check, for example ``912/0c7e1d4abc12/index.html``.

    Returns
    -------
    :
        The key, unchanged.

    Raises
    ------
    ValueError
        If the key is empty, absolute, contains a ``..`` segment, or uses characters
        outside ``[A-Za-z0-9._/-]``.
    """
    if not key or not _KEY_PATTERN.fullmatch(key):
        raise ValueError(
            f"Invalid report store key {key!r}: keys must be non-empty and may only contain "
            "letters, digits, dot, underscore, hyphen and forward slash."
        )
    if key.startswith("/"):
        raise ValueError(f"Invalid report store key {key!r}: keys must be relative, not absolute.")
    if ".." in PurePosixPath(key).parts:
        raise ValueError(f"Invalid report store key {key!r}: keys must not contain a '..' segment.")
    return key


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
        _validate_key(key)
        root = self.root
        if root is not None:
            return (root / key).absolute().as_uri()
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
        _validate_key(key)
        root = self.root
        if root is not None:
            dest = root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(dest))
        else:
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

        A local store's root is created if needed and checked for writability.
        A writable remote store performs a cheap authenticated ``HEAD`` on a sentinel key,
        which is expected to be absent, so a bad credential is caught before the upload starts.

        Raises
        ------
        NativeStoreUnavailableError
            If the store cannot be reached or used, with an operator-facing message.
        """
        root = self.root
        if root is not None:
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise NativeStoreUnavailableError(
                    f"Local report store root {root} could not be created: {exc}"
                ) from exc
            if not os.access(root, os.W_OK):
                raise NativeStoreUnavailableError(f"Local report store root {root} is not writable.")
            logger.debug(f"Local report store ready at {root}")
            return
        write = self.write
        if write is None:
            return

        from botocore.exceptions import ClientError  # noqa: PLC0415 - optional dependency

        try:
            write.client().head_object(Bucket=write.bucket, Key=_PREFLIGHT_PROBE_KEY)
        except ClientError as exc:
            status = _http_status(exc)
            if status == _HTTP_NOT_FOUND:
                pass  # authenticated, and the probe object is simply absent, so the store is usable
            elif status in _AUTH_REJECTED_STATUSES:
                raise NativeStoreUnavailableError(
                    f"Report store authentication failed (HTTP {status}) for bucket {write.bucket!r} at "
                    f"{write.endpoint_url}: the credentials were rejected or malformed. Check "
                    f"REF_REPORT_STORE_PROFILE, or REF_REPORT_STORE_ACCESS_KEY_ID / "
                    f"REF_REPORT_STORE_SECRET_ACCESS_KEY."
                ) from exc
            elif status == _HTTP_FORBIDDEN:
                raise NativeStoreUnavailableError(
                    f"Report store access denied (HTTP 403) for bucket {write.bucket!r} at "
                    f"{write.endpoint_url}: the request was forbidden. The secret key may be wrong, "
                    f"or the token may lack object read and write on this bucket. Check the "
                    f"credentials and the token's permissions."
                ) from exc
            else:
                raise NativeStoreUnavailableError(
                    f"Report store preflight failed (HTTP {status}) for bucket {write.bucket!r} at "
                    f"{write.endpoint_url}: {exc}"
                ) from exc
        logger.info(f"Report store authenticated: bucket {write.bucket!r} at {write.endpoint_url}")


class _ReportStoreConfigProtocol(Protocol):
    """
    Structural protocol for the report-store config object expected by :func:`build_report_store`.

    Keeps ``climate_ref_core`` free of any import dependency on ``climate_ref``, so both
    :class:`climate_ref.config.ReportStoreConfig` and test doubles satisfy it.

    ``s3_endpoint_url`` and ``bucket`` are non-secret routing config. Write credentials are
    intentionally **not** part of this protocol, and are read from the environment instead.
    """

    @property
    def url(self) -> str: ...

    @property
    def s3_endpoint_url(self) -> str: ...

    @property
    def bucket(self) -> str: ...


def build_report_store(config: _ReportStoreConfigProtocol, *, writable: bool) -> ReportStore:
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
        write=S3WriteConfig(
            endpoint_url=config.s3_endpoint_url,
            bucket=config.bucket,
            access_key_id=os.environ.get("REF_REPORT_STORE_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("REF_REPORT_STORE_SECRET_ACCESS_KEY", ""),
            profile=os.environ.get("REF_REPORT_STORE_PROFILE", ""),
        ),
    )
