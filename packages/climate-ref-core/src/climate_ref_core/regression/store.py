"""
Data store for native bundles.

A single :class:`NativeStore` serves both ends of the baseline lifecycle.
It is constructed from a read URL, plus the S3 routing config and credentials when it must
also write:

- A ``file://`` URL or a bare filesystem path is a local store, readable and writable
  without credentials, laid out two levels deep (``<root>/<digest[:2]>/<digest>``) like
  git's object store. Used by tests and local development.
- An ``http(s)://`` URL is the public read store. Blobs are pulled with :mod:`pooch` for
  caching, retry and hash verification, and are served flat at ``{url}/{digest}``.
- The same ``http(s)://`` URL plus ``writable=True`` adds the credentialed S3-compatible
  (Cloudflare R2) write path used by the ``mint`` verb.

The factory :func:`build_native_store` builds the store from the application
:class:`~climate_ref.config.Config` and the ``writable`` flag.
``writable=False`` never requires credentials.

Write credentials are **never** read from the persisted config: the S3 endpoint and
bucket are non-secret routing config, while authentication is resolved at client-build time
only, in precedence order: explicit ``REF_NATIVE_STORE_ACCESS_KEY_ID`` /
``REF_NATIVE_STORE_SECRET_ACCESS_KEY`` env vars, then a named ``REF_NATIVE_STORE_PROFILE``,
then boto3's default credential chain (which honours an ambient ``AWS_PROFILE``).

Blobs are keyed by their **sha256 hex digest**.
"""

import os
import shutil
from functools import cache
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import pooch
from attrs import field, frozen
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

_PREFLIGHT_PROBE_KEY = ".ref-preflight-probe"


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


def _local_root(url: str) -> Path | None:
    """
    Return the filesystem root a store URL names, or ``None`` when it is remote.

    ``file://`` URLs are parsed properly so malformed variants fail loudly instead of
    silently producing a wrong (for example relative) path.

    Raises
    ------
    ValueError
        If the URL scheme is unrecognised, or a ``file://`` URL carries a host component.
    """
    parts = urlsplit(url)
    if parts.scheme in ("http", "https"):
        return None
    if parts.scheme == "file":
        if parts.netloc not in ("", "localhost"):
            raise ValueError(
                f"Unsupported file URL {url!r}: a host component ({parts.netloc!r}) is not "
                "supported. Use the file:///absolute/path form (three slashes)."
            )
        return Path(unquote(parts.path))
    if parts.scheme == "":
        return Path(url)
    raise ValueError(
        f"Unsupported native store URL {url!r}: scheme {parts.scheme!r} is not recognised. "
        "Use http(s):// for a remote store, or file:///absolute/path or a bare filesystem "
        "path for a local store."
    )


def _is_missing(exc: Exception) -> bool:
    """Return ``True`` when a botocore ``ClientError`` denotes a missing object (404)."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = response.get("Error", {}).get("Code")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in _MISSING_OBJECT_CODES or status == _HTTP_NOT_FOUND


def _http_status(exc: Exception) -> int | None:
    """Return the HTTP status code from a botocore ``ClientError``, if present."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(status, int):
            return status
    return None


@frozen
class NativeStore:
    """
    Content-addressed store for native baseline blobs, keyed by sha256 hex digest.

    A local store (a ``file://`` URL or a bare path) always reads and writes without
    credentials. A remote store reads anonymously through :mod:`pooch`, and writes only
    when ``writable`` is set and the S3 routing config and credentials are supplied.

    Parameters
    ----------
    url
        Where blobs are read from: an ``http(s)://`` base URL (no trailing slash needed),
        a ``file:///absolute/path`` URL, or a bare filesystem path.
    cache_dir
        Local directory pooch caches remote downloads in. Unused by a local store.
    writable
        Whether the remote backend is credentialed for writes. A local store ignores this
        and is always writable.
    s3_endpoint_url
        S3 API endpoint for the bucket's account, without the bucket
        (e.g. ``https://<account>.eu.r2.cloudflarestorage.com``). Required for a writable
        remote store.
    bucket
        Name of the R2 bucket (e.g. ``ref-baselines``). Required for a writable remote store.
    access_key_id
        R2 access-key id, or ``""`` to fall through to ``profile`` / boto3's default chain.
    secret_access_key
        R2 secret-access-key, or ``""`` to fall through to ``profile`` / boto3's default chain.
    profile
        Named AWS/R2 profile to authenticate with, or ``""`` for the default session.
        Ignored when explicit ``access_key_id`` / ``secret_access_key`` are supplied.
    """

    url: str
    cache_dir: Path = field(default=Path(), converter=Path)
    writable: bool = False
    s3_endpoint_url: str = ""
    bucket: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    profile: str = ""

    def __attrs_post_init__(self) -> None:
        """Fail fast at construction (mint startup) when the URL or routing config is unusable."""
        if _local_root(self.url) is not None or not self.writable:
            return
        if not self.s3_endpoint_url:
            raise ValueError(
                "R2 native store requires an S3 endpoint URL; set REF_NATIVE_STORE_S3_ENDPOINT_URL "
                "(e.g. https://<account>.eu.r2.cloudflarestorage.com)."
            )
        if not self.bucket:
            raise ValueError(
                "R2 native store requires a bucket name; set REF_NATIVE_STORE_BUCKET (e.g. ref-baselines)."
            )

    @property
    def root(self) -> Path | None:
        """The local store root, or ``None`` when this store is remote."""
        return _local_root(self.url)

    def _blob_path(self, digest: str, root: Path) -> Path:
        """
        Return the canonical on-disk path for a blob in a local store.

        The digest is validated as 64-character lowercase hex first,
        so a malformed or hostile digest cannot be used to construct a path outside the root.
        """
        _validate_digest(digest)
        return root / digest[:2] / digest

    def _key(self, digest: str) -> str:
        """
        Return the object key for a blob, validating the digest first.

        The digest is validated as 64-character lowercase hex, so a malformed or hostile
        digest cannot inject an unexpected object key.
        """
        _validate_digest(digest)
        return digest

    def _client(self) -> Any:
        """Return the cached boto3 S3 client for this store's endpoint and credentials."""
        return _s3_client(self.s3_endpoint_url, self.access_key_id, self.secret_access_key, self.profile)

    def has(self, digest: str) -> bool:
        """
        Return ``True`` if the blob identified by ``digest`` is available.

        A local store checks its canonical path, a writable remote store ``HEAD``s the
        object, and an anonymous remote store checks the pooch cache.

        Parameters
        ----------
        digest
            The sha256 hex digest of the blob.

        Returns
        -------
        :
            ``True`` when the blob is present, ``False`` when it is not.
        """
        root = self.root
        if root is not None:
            return self._blob_path(digest, root).exists()
        if not self.writable:
            return (self.cache_dir / digest).exists()

        from botocore.exceptions import ClientError  # noqa: PLC0415 - optional dependency

        try:
            self._client().head_object(Bucket=self.bucket, Key=self._key(digest))
        except ClientError as exc:
            if _is_missing(exc):
                return False
            raise
        return True

    def fetch(self, digest: str, dest: Path) -> None:
        """
        Fetch the blob identified by ``digest`` and write it to ``dest``.

        The sha256 of the written file is verified to equal ``digest``.

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
            If the fetched blob's sha256 does not match ``digest``.
        """
        root = self.root
        dest.parent.mkdir(parents=True, exist_ok=True)
        if root is not None:
            blob = self._blob_path(digest, root)
            if not blob.exists():
                raise FileNotFoundError(f"Blob {digest!r} not found in local store at {root}")
            shutil.copy2(str(blob), str(dest))
        elif not self.writable:
            registry = _pooch_manager(self.url.rstrip("/"), str(self.cache_dir))
            registry.registry[digest] = digest  # content-addressed: hash == name
            shutil.copy2(registry.fetch(digest), str(dest))
        else:
            from botocore.exceptions import ClientError  # noqa: PLC0415 - optional dependency

            try:
                self._client().download_file(self.bucket, self._key(digest), str(dest))
            except ClientError as exc:
                if _is_missing(exc):
                    raise FileNotFoundError(
                        f"Blob {digest!r} not found in R2 bucket {self.bucket!r}"
                    ) from exc
                raise
        _verify_hash_matches(dest, digest)
        logger.debug(f"NativeStore.fetch: {digest} -> {dest}")

    def put(self, path: Path) -> str:
        """
        Store the file at ``path`` and return its sha256 hex digest.

        The blob is content-addressed, so storing one that is already present is skipped and
        re-minting is cheap.

        Parameters
        ----------
        path
            Path to the file to store.

        Returns
        -------
        :
            The sha256 hex digest of the stored blob.

        Raises
        ------
        NotImplementedError
            If this is an anonymous remote store, which cannot write.
        """
        digest = sha256_file(path)
        root = self.root
        if root is not None:
            blob = self._blob_path(digest, root)
            if blob.exists():
                logger.debug(f"NativeStore.put: {digest} already present, skipping copy")
                return digest
            blob.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(blob))
        else:
            if not self.writable:
                raise NotImplementedError(
                    f"Native store {self.url} is a public-read store; put() is not supported. "
                    "Mint against a local path or a credentialed remote store."
                )
            if self.has(digest):
                logger.debug(f"NativeStore.put: {digest} already present, skipping upload")
                return digest
            self._client().upload_file(str(path), self.bucket, self._key(digest))
        logger.debug(f"NativeStore.put: {path} -> {digest}")
        return digest

    def preflight(self) -> None:
        """
        Verify the store is reachable and usable before relying on it.

        A local store's root is created if needed and checked for writability. A writable
        remote store performs a cheap authenticated ``HEAD`` on a sentinel key (expected to
        be absent): a ``404`` means the request authenticated and the store is usable, while
        ``401`` / ``403`` become actionable errors so a misconfigured credential is caught
        before the (slow) diagnostic run rather than after. ``head_object`` is used rather
        than ``head_bucket`` so the check works with least-privilege, object-scoped tokens.

        An anonymous remote store has nothing to verify up front: it has no credentials, and
        every read is hash-checked per blob.

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
                    f"Local native store root {root} could not be created: {exc}"
                ) from exc
            if not os.access(root, os.W_OK):
                raise NativeStoreUnavailableError(f"Local native store root {root} is not writable.")
            logger.debug(f"Local native store ready at {root}")
            return
        if not self.writable:
            return

        from botocore.exceptions import ClientError  # noqa: PLC0415 - optional dependency

        try:
            self._client().head_object(Bucket=self.bucket, Key=_PREFLIGHT_PROBE_KEY)
        except ClientError as exc:
            status = _http_status(exc)
            if status == _HTTP_NOT_FOUND:
                pass  # authenticated, and the probe object is simply absent, so the store is usable
            elif status in _AUTH_REJECTED_STATUSES:
                raise NativeStoreUnavailableError(
                    f"Native store authentication failed (HTTP {status}) for bucket {self.bucket!r} at "
                    f"{self.s3_endpoint_url}: the credentials were rejected or malformed. Check "
                    f"REF_NATIVE_STORE_PROFILE, or REF_NATIVE_STORE_ACCESS_KEY_ID / "
                    f"REF_NATIVE_STORE_SECRET_ACCESS_KEY."
                ) from exc
            elif status == _HTTP_FORBIDDEN:
                raise NativeStoreUnavailableError(
                    f"Native store access denied (HTTP 403) for bucket {self.bucket!r} at "
                    f"{self.s3_endpoint_url}: the request was forbidden. The secret key may be wrong, "
                    f"or the token may lack object read and write on this bucket. Check the "
                    f"credentials and the token's permissions."
                ) from exc
            else:
                raise NativeStoreUnavailableError(
                    f"Native store preflight failed (HTTP {status}) for bucket {self.bucket!r} at "
                    f"{self.s3_endpoint_url}: {exc}"
                ) from exc
        logger.info(f"Native store authenticated: bucket {self.bucket!r} at {self.s3_endpoint_url}")


class _NativeStoreConfigProtocol(Protocol):
    """
    Structural protocol for the native-store config object expected by :func:`build_native_store`.

    Both :class:`climate_ref.config.NativeStoreConfig` and test doubles satisfy
    this interface without an import dependency on the app package.

    This keeps ``climate_ref_core`` free of any import dependency on ``climate_ref``.

    ``s3_endpoint_url`` and ``bucket`` are non-secret routing config consumed only by a
    writable remote store. Write credentials are intentionally **not** part of this protocol.
    They are read from the environment at client-build time, never from the config object.
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
    Build a :class:`NativeStore` from a native-store config object.

    Accepts any object that exposes ``url``, ``cache_dir``, ``s3_endpoint_url`` and ``bucket``
    (satisfying :class:`_NativeStoreConfigProtocol`), so callers pass ``config.native_store``
    rather than the full :class:`~climate_ref.config.Config`.

    With ``writable=False`` the returned store is anonymous and credential-free (suitable for
    CI read/replay paths). With ``writable=True`` and a remote URL the S3 endpoint and bucket
    come from the config, and authentication is read from the environment
    (``REF_NATIVE_STORE_ACCESS_KEY_ID`` / ``REF_NATIVE_STORE_SECRET_ACCESS_KEY``, else
    ``REF_NATIVE_STORE_PROFILE``, else boto3's default chain), so secrets never live in the
    persisted config. A local store is always readable and writable, so ``writable`` makes no
    difference to it.

    Parameters
    ----------
    config
        A config object providing ``url``, ``cache_dir``, ``s3_endpoint_url`` and ``bucket``.
        Typically ``app_config.native_store``.
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
    if not writable:
        return NativeStore(url=config.url, cache_dir=config.cache_dir)
    return NativeStore(
        url=config.url,
        cache_dir=config.cache_dir,
        writable=True,
        s3_endpoint_url=config.s3_endpoint_url,
        bucket=config.bucket,
        access_key_id=os.environ.get("REF_NATIVE_STORE_ACCESS_KEY_ID", ""),
        secret_access_key=os.environ.get("REF_NATIVE_STORE_SECRET_ACCESS_KEY", ""),
        profile=os.environ.get("REF_NATIVE_STORE_PROFILE", ""),
    )
