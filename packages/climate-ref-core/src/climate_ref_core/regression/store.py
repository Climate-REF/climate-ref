"""
Data store for native bundles.

This module provides the :class:`NativeStore` Protocol and three implementations:

- :class:`LocalFilesystemStore`: local filesystem store for tests and development.
  Supports both read and write.
- :class:`PoochReadStore`: anonymous public-read store backed by a URL,
  using :mod:`pooch` for caching, retry, and hash verification.
  Write is intentionally unsupported (``put`` raises :class:`NotImplementedError`).
- :class:`R2WriteStore`: stub for the future Cloudflare R2 write backend.
  Construction raises :class:`NotImplementedError` until implemented.

The factory :func:`build_native_store` selects the appropriate implementation
based on the application :class:`~climate_ref.config.Config` and the ``writable`` flag.
``writable=False`` never requires credentials.

Blobs are keyed by their **sha256 hex digest**.
The :class:`LocalFilesystemStore` uses a two-level directory layout
``<root>/<digest[:2]>/<digest>`` similar to git's object storage.
"""

import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

import pooch
from attrs import frozen
from loguru import logger

from climate_ref_core.dataset_registry import _verify_hash_matches
from climate_ref_core.regression.manifest import sha256_file


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
        """Return the canonical on-disk path for a blob with the given digest."""
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
        # Build a single-file pooch registry: the URL path component is the digest
        # and the expected hash IS the digest (content-addressed).
        registry = pooch.create(
            path=self.cache_dir,
            base_url=self.base_url + "/",
            retry_if_failed=10,
        )
        registry.registry[digest] = digest  # content-addressed: hash == name

        cached = registry.fetch(digest)
        # Verify the cached copy against the expected digest.
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


@frozen
class R2WriteStore:
    """
    Stub for the future Cloudflare R2 write backend.

    This class documents the seam for the credentialed write backend
    and will be implemented in a follow-up once the R2 credentials and
    bucket lifecycle policy are in place.

    Construction raises :class:`NotImplementedError` with a deferral message.
    """

    def __attrs_post_init__(self) -> None:
        raise NotImplementedError(
            "Remote writable native store (R2 backend) is deferred to a follow-up PR. "
            "Anonymous public-read URL + credentialed S3-compatible PUT will be wired here. "
            "Use a local store URL (file:// or a filesystem path) for minting in the meantime."
        )

    def has(self, digest: str) -> bool:  # pragma: no cover
        """Not implemented — R2 backend deferred."""
        raise NotImplementedError("R2 backend deferred")

    def fetch(self, digest: str, dest: Path) -> None:  # pragma: no cover
        """Not implemented — R2 backend deferred."""
        raise NotImplementedError("R2 backend deferred")

    def put(self, path: Path) -> str:  # pragma: no cover
        """Not implemented — R2 backend deferred."""
        raise NotImplementedError("R2 backend deferred")


class _NativeStoreConfigProtocol(Protocol):
    """
    Structural protocol for the native-store config object expected by :func:`build_native_store`.

    Both :class:`climate_ref.config.NativeStoreConfig` and test doubles satisfy
    this interface without an import dependency on the app package.

    This keeps ``climate_ref_core`` free of any import dependency on ``climate_ref``.

    """

    @property
    def url(self) -> str: ...

    @property
    def cache_dir(self) -> Path: ...


def build_native_store(config: _NativeStoreConfigProtocol, *, writable: bool) -> NativeStore:
    """
    Build an appropriate :class:`NativeStore` from a native-store config object.

    Accepts any object that exposes ``url: str`` and ``cache_dir: Path``
    (satisfying :class:`_NativeStoreConfigProtocol`), so callers pass
    ``config.native_store`` rather than the full :class:`~climate_ref.config.Config`.

    With ``writable=False`` the returned store is always anonymous and
    credential-free (suitable for CI read/replay paths).
    With ``writable=True`` and a local URL/path a :class:`LocalFilesystemStore`
    or with a remote URL a :class:`R2WriteStore` is attempted
    (currently deferred — raises :class:`NotImplementedError`).

    Parameters
    ----------
    config
        A config object providing ``url`` and ``cache_dir``.
        Typically ``app_config.native_store``.
    writable
        When ``False``, return a read-only store (no credentials required).
        When ``True``, return a writable store (``LocalFilesystemStore`` for local
        paths, or a :class:`R2WriteStore` for remote URLs — deferred).

    Returns
    -------
    :
        A :class:`NativeStore` implementation appropriate for the configuration.
    """
    url: str = config.url
    cache_dir: Path = config.cache_dir

    parts = urlsplit(url)
    is_local = parts.scheme not in ("http", "https")

    if is_local:
        if parts.scheme == "file":
            # Parse properly so malformed variants fail loudly instead of
            # silently producing a wrong (e.g. relative) path.
            if parts.netloc not in ("", "localhost"):
                raise ValueError(
                    f"Unsupported file URL {url!r}: a host component ({parts.netloc!r}) is not "
                    "supported. Use the file:///absolute/path form (three slashes)."
                )
            local_root = Path(parts.path)
        else:
            # A bare filesystem path.
            local_root = Path(url)
        return LocalFilesystemStore(root=local_root)
    elif writable:
        # Remote writable store — R2 backend is deferred.
        # Construction raises NotImplementedError until the follow-up lands.
        return R2WriteStore()
    else:
        return PoochReadStore(base_url=url.rstrip("/"), cache_dir=cache_dir)
