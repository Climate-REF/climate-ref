"""
Data store for native bundles.

This module provides the :class:`NativeStore` Protocol and three implementations:

- :class:`LocalFilesystemStore`: local filesystem store for tests and development.
  Supports both read and write.
- :class:`PoochReadStore`: anonymous public-read store backed by a URL,
  using :mod:`pooch` for caching, retry, and hash verification.
  Write is intentionally unsupported (``put`` raises :class:`NotImplementedError`).
- :class:`R2WriteStore`: credentialed S3-compatible write backend for Cloudflare R2.
  Used by the ``mint`` verb to upload native blobs; reads go through
  :class:`PoochReadStore` against the public read URL.

The factory :func:`build_native_store` selects the appropriate implementation
based on the application :class:`~climate_ref.config.Config` and the ``writable`` flag.
``writable=False`` never requires credentials.

Write credentials are **never** read from the persisted config: the S3 endpoint and
bucket are non-secret routing config, while authentication is resolved at client-build time
only, in precedence order: explicit ``REF_NATIVE_STORE_ACCESS_KEY_ID`` /
``REF_NATIVE_STORE_SECRET_ACCESS_KEY`` env vars, then a named ``REF_NATIVE_STORE_PROFILE``,
then boto3's default credential chain (which honours an ambient ``AWS_PROFILE``).

Blobs are keyed by their **sha256 hex digest**.
The :class:`LocalFilesystemStore` uses a two-level directory layout
``<root>/<digest[:2]>/<digest>`` similar to git's object storage.
The remote stores (:class:`PoochReadStore`, :class:`R2WriteStore`) use a flat layout
(object key == digest), so a blob is served at ``{public_url}/{digest}``.
"""

import os
import shutil
from functools import cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import unquote, urlsplit

import pooch
from attrs import frozen
from loguru import logger

from climate_ref_core.dataset_registry import _verify_hash_matches

from .manifest import _validate_digest, sha256_file

# S3 error codes / HTTP status that denote a missing object on a HEAD/GET.
_MISSING_OBJECT_CODES = ("404", "NoSuchKey", "NotFound")
_HTTP_NOT_FOUND = 404
_HTTP_BAD_REQUEST = 400
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
# On a trivial preflight HEAD (bucket + sentinel key + signature), these all mean the
# credentials could not be authenticated: malformed key (400), unknown/revoked key (401).
_AUTH_REJECTED_STATUSES = (_HTTP_BAD_REQUEST, _HTTP_UNAUTHORIZED)


class NativeStoreUnavailableError(RuntimeError):
    """
    Raised when a native store cannot be reached or used.

    Covers rejected credentials, a missing bucket, or an unwritable local directory.
    The message is operator-facing and actionable (it names the env vars / path to check),
    so callers can surface it directly.
    """


@cache
def _pooch_manager(base_url: str, cache_dir: str) -> pooch.Pooch:
    """
    Build (and cache) a pooch manager for a ``(base_url, cache_dir)`` pair.

    ``pooch.create`` rebuilds the whole manager and registry, so doing it per
    ``fetch`` is wasteful when many blobs are pulled from the same store.
    The manager is keyed by its immutable inputs and reused across fetches;
    per-blob registry entries are added on the shared instance at fetch time.
    """
    return pooch.create(
        path=Path(cache_dir),
        base_url=base_url + "/",
        retry_if_failed=10,
    )


@cache
def _s3_client(endpoint_url: str, access_key_id: str, secret_access_key: str, profile: str) -> Any:
    """
    Build (and cache) an S3-compatible client for a Cloudflare R2 endpoint.

    boto3 is imported lazily so the read/replay paths (and any environment without the
    optional ``aws`` extra installed) never pull in boto3. The client is cached by its
    immutable inputs so many ``put`` calls in a single ``mint`` run reuse one client.

    Authentication precedence (each empty value falls through to the next):

    1. Explicit ``access_key_id`` / ``secret_access_key`` (from the REF cred env vars).
    2. A named ``profile`` from ``~/.aws/{config,credentials}``.
    3. boto3's default credential chain (ambient ``AWS_PROFILE`` / ``AWS_ACCESS_KEY_ID`` /
       instance profile, etc.).

    R2 requires SigV4 and a fixed ``auto`` region; path-style addressing avoids virtual-host
    DNS requirements against the account endpoint.

    Parameters
    ----------
    endpoint_url
        The S3 API endpoint of the R2 bucket's account
        (e.g. ``https://<account>.eu.r2.cloudflarestorage.com``), without the bucket.
    access_key_id
        R2 access-key id, or ``""`` to fall through to the profile / default chain.
    secret_access_key
        R2 secret-access-key, or ``""`` to fall through to the profile / default chain.
    profile
        Named AWS/R2 profile to load credentials from, or ``""`` for the default session
        (which still honours an ambient ``AWS_PROFILE``).

    Returns
    -------
    :
        A configured boto3 S3 client (typed ``Any``; boto3 ships no inline types).
    """
    try:
        import boto3  # noqa: PLC0415 - optional dependency, imported lazily
        from botocore.config import Config as BotoConfig  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Minting to a remote native store requires boto3, which is an optional "
            "dependency. Install it with the 'aws' extra, e.g. "
            "`uv pip install 'climate-ref-core[aws]'`."
        ) from exc

    session = boto3.Session(profile_name=profile or None)
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id or None,
        aws_secret_access_key=secret_access_key or None,
        region_name="auto",
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


@runtime_checkable
class NativeStore(Protocol):
    """
    Protocol for a content-addressed native-bundle blob store.

    Blobs are keyed by their sha256 hex digest.
    Read operations (``has``, ``fetch``) are anonymous and credential-free.
    ``put`` requires write credentials and raises :class:`NotImplementedError` on read-only implementations.
    """

    def has(self, digest: str) -> bool:
        """
        Return ``True`` if the blob identified by ``digest`` is available in the store.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob.

        Returns
        -------
        :
            ``True`` when the blob is present, ``False`` when it is not.
        """
        ...

    def fetch(self, digest: str, dest: Path) -> None:
        """
        Fetch the blob identified by ``digest`` and write it to ``dest``.

        The sha256 of the written file is verified to equal ``digest``.
        Raises :class:`ValueError` on hash mismatch and
        :class:`FileNotFoundError` if the blob is not found.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob to fetch.
        dest
            Destination path to write the blob to.
            Parent directories are created if they do not exist.
        """
        ...

    def put(self, path: Path) -> str:
        """
        Store the file at ``path`` and return its sha256 hex digest.

        Requires write credentials.
        Read-only implementations raise :class:`NotImplementedError`.

        Parameters
        ----------
        path
            Path to the file to store.

        Returns
        -------
        :
            The sha256 hex digest of the stored blob.
        """
        ...

    def preflight(self) -> None:
        """
        Verify the store is reachable and usable before relying on it.

        For writable stores this checks the credentials and target (bucket, or that the local
        directory is writable); for anonymous read-only stores it may be a no-op. Intended to
        be called once up front (e.g. before a slow ``mint`` run) so a misconfiguration is
        caught early.

        Raises
        ------
        NativeStoreUnavailableError
            If the store cannot be reached or used, with an operator-facing message.
        """
        ...


@frozen
class LocalFilesystemStore:
    """
    Content-addressed blob store backed by a local filesystem directory.

    Intended for tests and development.
    Supports both read and write without credentials.

    Blobs are stored under ``root`` using a two-level layout::

        <root>/<digest[:2]>/<digest>

    This mirrors the git object-store convention,
    keeping individual subdirectories manageable for large collections.

    Parameters
    ----------
    root
        Root directory for the content-addressed store.
        Created on first write if it does not exist.
    """

    root: Path

    def _blob_path(self, digest: str) -> Path:
        """Return the canonical on-disk path for a blob with the given digest.

        The digest is validated as 64-character lowercase hex first,
        so a malformed or hostile digest cannot be used to construct a path outside the store root.
        """
        _validate_digest(digest)
        return self.root / digest[:2] / digest

    def has(self, digest: str) -> bool:
        """
        Return ``True`` if the blob is present on disk.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob.

        Returns
        -------
        :
            ``True`` when the blob file exists at its canonical path.
        """
        return self._blob_path(digest).exists()

    def fetch(self, digest: str, dest: Path) -> None:
        """
        Copy the blob to ``dest`` and verify its sha256 matches ``digest``.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob to fetch.
        dest
            Destination path to write the blob to.
            Parent directories are created if they do not exist.

        Raises
        ------
        FileNotFoundError
            If the blob is not present in the store.
        ValueError
            If the blob's sha256 does not match ``digest``.
        """
        blob = self._blob_path(digest)
        if not blob.exists():
            raise FileNotFoundError(f"Blob {digest!r} not found in local store at {self.root}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(blob), str(dest))
        _verify_hash_matches(dest, digest)
        logger.debug(f"LocalFilesystemStore.fetch: {digest} -> {dest}")

    def put(self, path: Path) -> str:
        """
        Store the file at ``path`` in the content-addressed store.

        Computes the sha256 digest, copies the file to its canonical location, and returns the digest.
        If a blob with the same digest already exists, the copy is skipped.

        Parameters
        ----------
        path
            Path to the file to store.

        Returns
        -------
        :
            The sha256 hex digest of the stored blob.
        """
        digest = sha256_file(path)
        blob = self._blob_path(digest)
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(blob))
            logger.debug(f"LocalFilesystemStore.put: {path} -> {blob}")
        else:
            logger.debug(f"LocalFilesystemStore.put: {digest} already present, skipping copy")
        return digest

    def preflight(self) -> None:
        """
        Verify the store root exists (creating it if needed) and is writable.

        Raises
        ------
        NativeStoreUnavailableError
            If the root cannot be created or is not writable.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise NativeStoreUnavailableError(
                f"Local native store root {self.root} could not be created: {exc}"
            ) from exc
        if not os.access(self.root, os.W_OK):
            raise NativeStoreUnavailableError(f"Local native store root {self.root} is not writable.")
        logger.debug(f"Local native store ready at {self.root}")


@frozen
class PoochReadStore:
    """
    Anonymous public-read blob store backed by a remote URL.

    Uses :mod:`pooch` for caching, retry, and hash verification,
    mirroring the pattern in :mod:`climate_ref_core.dataset_registry`.

    Blobs are fetched from ``{base_url}/{digest}`` and cached under ``cache_dir``.
    ``put`` is intentionally unsupported; minting uses the write backend.

    Parameters
    ----------
    base_url
        Base URL from which blobs are served (no trailing slash).
        Example: ``https://baselines.climate-ref.org``.
    cache_dir
        Local directory used by pooch to cache downloaded blobs.
    """

    base_url: str
    cache_dir: Path

    def _cached_blob_path(self, digest: str) -> Path:
        """Return the pooch cache path for a blob."""
        return self.cache_dir / digest

    def has(self, digest: str) -> bool:
        """
        Return ``True`` if the blob is already in the local pooch cache.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob.

        Returns
        -------
        :
            ``True`` when the cached file exists.
        """
        return self._cached_blob_path(digest).exists()

    def fetch(self, digest: str, dest: Path) -> None:
        """
        Download the blob from ``{base_url}/{digest}`` and write it to ``dest``.

        Uses pooch for caching and retry.
        Verifies the sha256 of the downloaded file against ``digest``.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob to fetch.
        dest
            Destination path to write the blob to.
            Parent directories are created if they do not exist.

        Raises
        ------
        ValueError
            If the downloaded blob's sha256 does not match ``digest``.
        """
        registry = _pooch_manager(self.base_url, str(self.cache_dir))
        registry.registry[digest] = digest  # content-addressed: hash == name

        cached = registry.fetch(digest)
        # pooch should already verify the hash against the registry entry
        _verify_hash_matches(cached, digest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, str(dest))
        logger.debug(f"PoochReadStore.fetch: {digest} -> {dest}")

    def put(self, path: Path) -> str:
        """
        Not supported on this read-only store.

        Raises
        ------
        NotImplementedError
            Always; minting uses the write backend, not the read store.
        """
        raise NotImplementedError(
            "PoochReadStore is a public-read store; put() is not supported. "
            "Use a writable store (LocalFilesystemStore or R2WriteStore) for minting."
        )

    def preflight(self) -> None:
        """
        No-op: an anonymous public-read store has nothing to verify up front.

        It has no credentials, and every read is hash-checked per blob; this exists only to
        satisfy the :class:`NativeStore` protocol.
        """
        return None


@frozen
class R2WriteStore:
    """
    Credentialed S3-compatible write backend for a Cloudflare R2 bucket.

    Used by the ``mint`` verb to upload native blobs. Reads in CI and for local replay go
    through :class:`PoochReadStore` against the public read URL, so this store's ``fetch`` /
    ``has`` exist mainly for mint-time idempotence and verification.

    Blobs are content-addressed with a **flat** key layout (object key == ``key_prefix`` +
    digest), so a blob is served at ``{public_url}/{key_prefix}{digest}``. The public read
    domain is expected to map to the bucket root, so ``key_prefix`` defaults to ``""``.

    boto3 is imported lazily (see :func:`_s3_client`); constructing this store does not
    require boto3, only the endpoint and bucket. Credentials are passed in explicitly and
    are never sourced from the persisted config.

    Parameters
    ----------
    endpoint_url
        S3 API endpoint for the bucket's account, without the bucket
        (e.g. ``https://<account>.eu.r2.cloudflarestorage.com``).
    bucket
        Name of the R2 bucket (e.g. ``ref-baselines``).
    access_key_id
        R2 access-key id, or ``""`` to fall through to ``profile`` / boto3's default chain.
    secret_access_key
        R2 secret-access-key, or ``""`` to fall through to ``profile`` / boto3's default chain.
    profile
        Named AWS/R2 profile to authenticate with, or ``""`` for the default session.
        Ignored when explicit ``access_key_id`` / ``secret_access_key`` are supplied.
    key_prefix
        Optional object-key prefix. Defaults to ``""`` (flat, bucket-root layout).
    """

    endpoint_url: str
    bucket: str
    access_key_id: str = ""
    secret_access_key: str = ""
    profile: str = ""
    key_prefix: str = ""

    def __attrs_post_init__(self) -> None:
        """Fail fast at construction (mint startup) when routing config is missing."""
        if not self.endpoint_url:
            raise ValueError(
                "R2 native store requires an S3 endpoint URL; set REF_NATIVE_STORE_S3_ENDPOINT_URL "
                "(e.g. https://<account>.eu.r2.cloudflarestorage.com)."
            )
        if not self.bucket:
            raise ValueError(
                "R2 native store requires a bucket name; set REF_NATIVE_STORE_BUCKET (e.g. ref-baselines)."
            )

    def _key(self, digest: str) -> str:
        """Return the object key for a blob, validating the digest first.

        The digest is validated as 64-character lowercase hex, so a malformed or hostile
        digest cannot inject an unexpected object key.
        """
        _validate_digest(digest)
        return f"{self.key_prefix}{digest}"

    def _client(self) -> Any:
        """Return the cached boto3 S3 client for this store's endpoint and credentials."""
        return _s3_client(self.endpoint_url, self.access_key_id, self.secret_access_key, self.profile)

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        """Return ``True`` when a botocore ``ClientError`` denotes a missing object (404)."""
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return False
        code = response.get("Error", {}).get("Code")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in _MISSING_OBJECT_CODES or status == _HTTP_NOT_FOUND

    def has(self, digest: str) -> bool:
        """
        Return ``True`` if the blob is present in the bucket.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob.

        Returns
        -------
        :
            ``True`` when a ``HEAD`` on the object succeeds, ``False`` on a 404.
        """
        from botocore.exceptions import ClientError  # noqa: PLC0415 - optional dependency

        try:
            self._client().head_object(Bucket=self.bucket, Key=self._key(digest))
        except ClientError as exc:
            if self._is_missing(exc):
                return False
            raise
        return True

    def fetch(self, digest: str, dest: Path) -> None:
        """
        Download the blob to ``dest`` and verify its sha256 matches ``digest``.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob to fetch.
        dest
            Destination path to write the blob to.
            Parent directories are created if they do not exist.

        Raises
        ------
        FileNotFoundError
            If the blob is not present in the bucket.
        ValueError
            If the downloaded blob's sha256 does not match ``digest``.
        """
        from botocore.exceptions import ClientError  # noqa: PLC0415 - optional dependency

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client().download_file(self.bucket, self._key(digest), str(dest))
        except ClientError as exc:
            if self._is_missing(exc):
                raise FileNotFoundError(f"Blob {digest!r} not found in R2 bucket {self.bucket!r}") from exc
            raise
        _verify_hash_matches(dest, digest)
        logger.debug(f"R2WriteStore.fetch: {digest} -> {dest}")

    def put(self, path: Path) -> str:
        """
        Upload the file at ``path`` to the bucket and return its sha256 hex digest.

        The blob is content-addressed: the upload is skipped when an object with the same
        digest already exists, so minting is idempotent and re-mints are cheap.

        Parameters
        ----------
        path
            Path to the file to store.

        Returns
        -------
        :
            The sha256 hex digest of the stored blob.
        """
        digest = sha256_file(path)
        if self.has(digest):
            logger.debug(f"R2WriteStore.put: {digest} already present, skipping upload")
            return digest
        self._client().upload_file(str(path), self.bucket, self._key(digest))
        logger.debug(f"R2WriteStore.put: {path} -> s3://{self.bucket}/{self._key(digest)}")
        return digest

    @staticmethod
    def _http_status(exc: Exception) -> int | None:
        """Return the HTTP status code from a botocore ``ClientError``, if present."""
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if isinstance(status, int):
                return status
        return None

    def preflight(self) -> None:
        """
        Verify the bucket is reachable and the credentials are accepted, before any upload.

        Performs a cheap authenticated ``HEAD`` on a sentinel key (expected to be absent). A
        ``404`` means the request authenticated and the store is usable; ``401`` / ``403`` are
        translated into actionable :class:`NativeStoreUnavailableError` messages so a
        misconfigured credential is caught before the (slow) diagnostic run rather than after.

        ``head_object`` is used rather than ``head_bucket`` so the check works with
        least-privilege, object-scoped tokens (which cannot perform bucket-level operations).

        Raises
        ------
        NativeStoreUnavailableError
            If the credentials are rejected (401), access is denied (403), or the probe
            otherwise fails.
        """
        from botocore.exceptions import ClientError  # noqa: PLC0415 - optional dependency

        probe_key = f"{self.key_prefix}.ref-preflight-probe"
        try:
            self._client().head_object(Bucket=self.bucket, Key=probe_key)
        except ClientError as exc:
            status = self._http_status(exc)
            if status == _HTTP_NOT_FOUND:
                pass  # authenticated; the probe object is simply absent — store is usable
            elif status in _AUTH_REJECTED_STATUSES:
                raise NativeStoreUnavailableError(
                    f"Native store authentication failed (HTTP {status}) for bucket {self.bucket!r} at "
                    f"{self.endpoint_url}: the credentials were rejected or malformed. Check "
                    f"REF_NATIVE_STORE_PROFILE, or REF_NATIVE_STORE_ACCESS_KEY_ID / "
                    f"REF_NATIVE_STORE_SECRET_ACCESS_KEY."
                ) from exc
            elif status == _HTTP_FORBIDDEN:
                raise NativeStoreUnavailableError(
                    f"Native store access denied (HTTP 403) for bucket {self.bucket!r} at "
                    f"{self.endpoint_url}: the request was forbidden — the secret key may be wrong, or "
                    f"the token may lack object read & write on this bucket. Check the credentials and "
                    f"the token's permissions."
                ) from exc
            else:
                raise NativeStoreUnavailableError(
                    f"Native store preflight failed (HTTP {status}) for bucket {self.bucket!r} at "
                    f"{self.endpoint_url}: {exc}"
                ) from exc
        logger.info(f"Native store authenticated: bucket {self.bucket!r} at {self.endpoint_url}")


class _NativeStoreConfigProtocol(Protocol):
    """
    Structural protocol for the native-store config object expected by :func:`build_native_store`.

    Both :class:`climate_ref.config.NativeStoreConfig` and test doubles satisfy
    this interface without an import dependency on the app package.

    This keeps ``climate_ref_core`` free of any import dependency on ``climate_ref``.

    ``s3_endpoint_url`` and ``bucket`` are non-secret routing config consumed only by the
    writable (R2) backend. Write credentials are intentionally **not** part of this protocol
    — they are read from the environment at client-build time, never from the config object.
    """

    @property
    def url(self) -> str: ...

    @property
    def cache_dir(self) -> Path: ...

    @property
    def s3_endpoint_url(self) -> str: ...

    @property
    def bucket(self) -> str: ...


def build_native_store(config: _NativeStoreConfigProtocol, *, writable: bool) -> NativeStore:
    """
    Build an appropriate :class:`NativeStore` from a native-store config object.

    Accepts any object that exposes ``url: str`` and ``cache_dir: Path``
    (satisfying :class:`_NativeStoreConfigProtocol`), so callers pass
    ``config.native_store`` rather than the full :class:`~climate_ref.config.Config`.

    With ``writable=False`` the returned store is always anonymous and
    credential-free (suitable for CI read/replay paths).
    With ``writable=True`` and a local URL/path a :class:`LocalFilesystemStore` is returned;
    with a remote (``http(s)``) URL a credentialed :class:`R2WriteStore` is returned. The S3
    endpoint and bucket come from the config; authentication is read from the environment
    (``REF_NATIVE_STORE_ACCESS_KEY_ID`` / ``REF_NATIVE_STORE_SECRET_ACCESS_KEY``, else
    ``REF_NATIVE_STORE_PROFILE``, else boto3's default chain), so secrets never live in the
    persisted config.

    Parameters
    ----------
    config
        A config object providing ``url``, ``cache_dir``, ``s3_endpoint_url`` and ``bucket``.
        Typically ``app_config.native_store``.
    writable
        When ``False``, return a read-only store (no credentials required).
        When ``True``, return a writable store (``LocalFilesystemStore`` for local
        paths, or a :class:`R2WriteStore` for remote URLs).

    Returns
    -------
    :
        A :class:`NativeStore` implementation appropriate for the configuration.

    Raises
    ------
    ValueError
        If the URL scheme is unrecognised, or a writable remote store is requested
        without an S3 endpoint / bucket configured.
    """
    url: str = config.url
    cache_dir: Path = config.cache_dir

    parts = urlsplit(url)
    scheme = parts.scheme

    if scheme in ("http", "https"):
        if writable:
            return R2WriteStore(
                endpoint_url=config.s3_endpoint_url,
                bucket=config.bucket,
                access_key_id=os.environ.get("REF_NATIVE_STORE_ACCESS_KEY_ID", ""),
                secret_access_key=os.environ.get("REF_NATIVE_STORE_SECRET_ACCESS_KEY", ""),
                profile=os.environ.get("REF_NATIVE_STORE_PROFILE", ""),
            )
        return PoochReadStore(base_url=url.rstrip("/"), cache_dir=cache_dir)

    if scheme == "file":
        # Parse properly so malformed variants fail loudly instead of
        # silently producing a wrong (e.g. relative) path.
        if parts.netloc not in ("", "localhost"):
            raise ValueError(
                f"Unsupported file URL {url!r}: a host component ({parts.netloc!r}) is not "
                "supported. Use the file:///absolute/path form (three slashes)."
            )
        return LocalFilesystemStore(root=Path(unquote(parts.path)))

    if scheme == "":
        return LocalFilesystemStore(root=Path(url))

    raise ValueError(
        f"Unsupported native store URL {url!r}: scheme {scheme!r} is not recognised. "
        "Use http(s):// for a remote store, or file:///absolute/path or a bare filesystem "
        "path for a local store."
    )
