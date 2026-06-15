import hashlib
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from climate_ref_core.regression.store import (
    LocalFilesystemStore,
    NativeStore,
    PoochReadStore,
    R2WriteStore,
    build_native_store,
)


def _patch_pooch_create(
    mocker: MockerFixture,
    served: dict[str, bytes],
) -> None:
    """
    Replace ``pooch.create`` so ``PoochReadStore.fetch`` runs without any network.

    The returned registry double writes the bytes registered in ``served`` to the
    pooch cache directory and returns the cached path, leaving the store's own
    ``_verify_hash_matches`` to do the real hash check.

    Parameters
    ----------
    mocker
        The pytest-mock fixture.
    served
        Mapping of blob name (the sha256 digest) to the bytes the fake server returns.
    """

    def _create(*, path: Path, base_url: str, retry_if_failed: int = 0) -> object:
        registry: dict[str, str] = {}

        def _fetch(name: str) -> str:
            cache_dir = Path(path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / name
            cached.write_bytes(served[name])
            return str(cached)

        fake = mocker.MagicMock()
        fake.registry = registry
        fake.fetch.side_effect = _fetch
        return fake

    mocker.patch("climate_ref_core.regression.store.pooch.create", side_effect=_create)


@pytest.fixture()
def blob_content() -> bytes:
    return b"native bundle blob content for testing"


@pytest.fixture()
def blob_file(tmp_path: Path, blob_content: bytes) -> Path:
    p = tmp_path / "blob.nc"
    p.write_bytes(blob_content)
    return p


@pytest.fixture()
def blob_digest(blob_content: bytes) -> str:
    return hashlib.sha256(blob_content).hexdigest()


@pytest.fixture()
def local_store(tmp_path: Path) -> LocalFilesystemStore:
    return LocalFilesystemStore(root=tmp_path / "store")


class TestLocalFilesystemStore:
    def test_satisfies_protocol(self, local_store: LocalFilesystemStore) -> None:
        assert isinstance(local_store, NativeStore)

    def test_put_returns_sha256(
        self, local_store: LocalFilesystemStore, blob_file: Path, blob_digest: str
    ) -> None:
        result = local_store.put(blob_file)
        assert result == blob_digest

    def test_has_true_after_put(
        self, local_store: LocalFilesystemStore, blob_file: Path, blob_digest: str
    ) -> None:
        local_store.put(blob_file)
        assert local_store.has(blob_digest) is True

    def test_has_false_before_put(self, local_store: LocalFilesystemStore, blob_digest: str) -> None:
        assert local_store.has(blob_digest) is False

    def test_fetch_byte_identical(
        self,
        local_store: LocalFilesystemStore,
        blob_file: Path,
        blob_content: bytes,
        blob_digest: str,
        tmp_path: Path,
    ) -> None:
        local_store.put(blob_file)
        dest = tmp_path / "fetched.nc"
        local_store.fetch(blob_digest, dest)
        assert dest.read_bytes() == blob_content

    def test_fetch_creates_parent_dirs(
        self,
        local_store: LocalFilesystemStore,
        blob_file: Path,
        blob_digest: str,
        tmp_path: Path,
    ) -> None:
        local_store.put(blob_file)
        dest = tmp_path / "deep" / "nested" / "file.nc"
        local_store.fetch(blob_digest, dest)
        assert dest.exists()

    def test_fetch_missing_raises_file_not_found(
        self, local_store: LocalFilesystemStore, blob_digest: str, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            local_store.fetch(blob_digest, tmp_path / "out.nc")

    @pytest.mark.parametrize(
        "bad_digest",
        [
            "../../etc/passwd",
            "not-hex",
            "ABCDEF" + "0" * 58,  # uppercase rejected
            "0" * 63,  # too short
            "0" * 65,  # too long
        ],
    )
    def test_blob_path_rejects_bad_digest(
        self, local_store: LocalFilesystemStore, bad_digest: str
    ) -> None:
        with pytest.raises(ValueError, match="Invalid sha256 digest"):
            local_store.has(bad_digest)

    def test_fetch_digest_mismatch_raises_value_error(
        self,
        local_store: LocalFilesystemStore,
        blob_file: Path,
        blob_digest: str,
        tmp_path: Path,
    ) -> None:
        """Corrupt the stored blob; fetch should detect the mismatch."""
        local_store.put(blob_file)
        # Corrupt the stored blob directly
        blob_path = local_store._blob_path(blob_digest)
        blob_path.write_bytes(b"corrupted content")
        dest = tmp_path / "out.nc"
        with pytest.raises(ValueError):
            local_store.fetch(blob_digest, dest)

    def test_put_idempotent(
        self,
        local_store: LocalFilesystemStore,
        blob_file: Path,
        blob_digest: str,
    ) -> None:
        """Putting the same blob twice must succeed and return the same digest."""
        d1 = local_store.put(blob_file)
        d2 = local_store.put(blob_file)
        assert d1 == d2 == blob_digest

    def test_two_level_layout(
        self, local_store: LocalFilesystemStore, blob_file: Path, blob_digest: str
    ) -> None:
        """Blobs must be stored at root/<digest[:2]>/<digest>."""
        local_store.put(blob_file)
        expected = local_store.root / blob_digest[:2] / blob_digest
        assert expected.exists()


class TestPoochReadStore:
    def test_satisfies_protocol(self, tmp_path: Path) -> None:
        store = PoochReadStore(base_url="https://example.com", cache_dir=tmp_path)
        assert isinstance(store, NativeStore)

    def test_put_raises_not_implemented(self, tmp_path: Path, blob_file: Path) -> None:
        store = PoochReadStore(base_url="https://example.com", cache_dir=tmp_path)
        with pytest.raises(NotImplementedError):
            store.put(blob_file)

    def test_fetch_byte_identical(
        self,
        mocker: MockerFixture,
        blob_content: bytes,
        blob_digest: str,
        tmp_path: Path,
    ) -> None:
        """
        PoochReadStore.fetch must download a blob and produce byte-identical content.

        The pooch download is mocked so the test needs no network or local server.
        """
        _patch_pooch_create(mocker, {blob_digest: blob_content})

        store = PoochReadStore(base_url="https://example.com", cache_dir=tmp_path / "cache")
        dest = tmp_path / "fetched.nc"
        store.fetch(blob_digest, dest)

        assert dest.read_bytes() == blob_content

    def test_fetch_hash_verified(
        self,
        mocker: MockerFixture,
        blob_digest: str,
        tmp_path: Path,
    ) -> None:
        """
        A blob served with corrupt content under the correct digest name
        must be detected by ``_verify_hash_matches`` and raise.
        """
        # The fake server returns corrupt bytes under the correct digest name.
        _patch_pooch_create(mocker, {blob_digest: b"corrupt bytes -- digest will not match"})

        store = PoochReadStore(base_url="https://example.com", cache_dir=tmp_path / "cache")
        dest = tmp_path / "out.nc"
        with pytest.raises(ValueError, match="does not match"):
            store.fetch(blob_digest, dest)


class TestR2WriteStore:
    def test_construction_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="deferred to a follow-up PR"):
            R2WriteStore()


class _StubConfig:
    """Minimal config double satisfying _NativeStoreConfigProtocol."""

    def __init__(self, url: str, cache_dir: Path) -> None:
        self._url = url
        self._cache_dir = cache_dir

    @property
    def url(self) -> str:
        return self._url

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir


class TestBuildNativeStore:
    def test_writable_false_local_path_returns_local_store(self, tmp_path: Path) -> None:
        cfg = _StubConfig(url=str(tmp_path / "store"), cache_dir=tmp_path / "cache")
        store = build_native_store(cfg, writable=False)
        assert isinstance(store, LocalFilesystemStore)

    def test_writable_false_file_url_returns_local_store(self, tmp_path: Path) -> None:
        cfg = _StubConfig(
            url=(tmp_path / "store").as_uri(),
            cache_dir=tmp_path / "cache",
        )
        store = build_native_store(cfg, writable=False)
        assert isinstance(store, LocalFilesystemStore)

    def test_writable_false_remote_url_returns_pooch_store(self, tmp_path: Path) -> None:
        cfg = _StubConfig(
            url="https://baselines.example.com",
            cache_dir=tmp_path / "cache",
        )
        store = build_native_store(cfg, writable=False)
        assert isinstance(store, PoochReadStore)

    def test_writable_false_never_requires_creds(self, tmp_path: Path) -> None:
        """
        ``writable=False`` must always return a credential-free store.

        Local stores are inherently creds-free.
        Remote read stores (PoochReadStore) are anonymous public-read.
        This test asserts no :class:`NotImplementedError` is raised.
        """
        for url in [
            str(tmp_path / "store"),
            (tmp_path / "store").as_uri(),
            "https://baselines.example.com",
        ]:
            cfg = _StubConfig(url=url, cache_dir=tmp_path / "cache")
            # Must not raise — no credentials required.
            store = build_native_store(cfg, writable=False)
            assert isinstance(store, NativeStore)

    def test_file_url_resolves_to_absolute_root(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        for url in [root.as_uri(), f"file:{root}"]:  # file:///abs and single-slash file:/abs
            cfg = _StubConfig(url=url, cache_dir=tmp_path / "cache")
            store = build_native_store(cfg, writable=False)
            assert isinstance(store, LocalFilesystemStore)
            assert store.root == root

    def test_file_url_with_host_raises_value_error(self, tmp_path: Path) -> None:
        cfg = _StubConfig(url="file://store/blobs", cache_dir=tmp_path / "cache")
        with pytest.raises(ValueError, match="host component"):
            build_native_store(cfg, writable=False)

    def test_writable_true_remote_url_raises_not_implemented(self, tmp_path: Path) -> None:
        cfg = _StubConfig(url="https://baselines.example.com", cache_dir=tmp_path / "cache")
        with pytest.raises(NotImplementedError, match="deferred to a follow-up PR"):
            build_native_store(cfg, writable=True)

    def test_unsupported_scheme_raises_value_error(self, tmp_path: Path) -> None:
        # An unrecognised remote scheme must fail loudly, not be coerced to a local path.
        for url in ["s3://bucket/blobs", "gs://bucket/blobs", "ftp://host/blobs"]:
            cfg = _StubConfig(url=url, cache_dir=tmp_path / "cache")
            with pytest.raises(ValueError, match="not recognised"):
                build_native_store(cfg, writable=False)
