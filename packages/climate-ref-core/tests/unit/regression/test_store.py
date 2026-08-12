import hashlib
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from pytest_mock import MockerFixture

from climate_ref_core.regression.store import (
    NativeStore,
    NativeStoreUnavailableError,
    S3WriteConfig,
    build_native_store,
)

REMOTE_URL = "https://baselines.example.com"
S3_ENDPOINT = "https://account.r2.cloudflarestorage.com"
BUCKET = "ref-baselines"


def _patch_pooch_create(mocker: MockerFixture, served: dict[str, bytes]) -> None:
    """
    Replace ``pooch.create`` so an anonymous remote fetch runs without any network.

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
        def _fetch(name: str) -> str:
            cache_dir = Path(path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / name
            cached.write_bytes(served[name])
            return str(cached)

        fake = mocker.MagicMock()
        fake.registry = {}
        fake.fetch.side_effect = _fetch
        return fake

    mocker.patch("climate_ref_core.regression.store.pooch.create", side_effect=_create)


def _client_error(code: str, status: int, operation: str = "HeadObject") -> ClientError:
    """Build a botocore ``ClientError`` with the given S3 error code / HTTP status."""
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


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
def local_store(tmp_path: Path) -> NativeStore:
    return NativeStore(url=str(tmp_path / "store"))


class _StubConfig:
    """Minimal config double satisfying _NativeStoreConfigProtocol."""

    def __init__(
        self,
        url: str,
        cache_dir: Path,
        s3_endpoint_url: str = S3_ENDPOINT,
        bucket: str = BUCKET,
    ) -> None:
        self._url = url
        self._cache_dir = cache_dir
        self._s3_endpoint_url = s3_endpoint_url
        self._bucket = bucket

    @property
    def url(self) -> str:
        return self._url

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def s3_endpoint_url(self) -> str:
        return self._s3_endpoint_url

    @property
    def bucket(self) -> str:
        return self._bucket


class TestLocalStore:
    """A ``file://`` URL or bare path reads and writes on the local filesystem."""

    def test_put_returns_sha256(self, local_store: NativeStore, blob_file: Path, blob_digest: str) -> None:
        assert local_store.put(blob_file) == blob_digest

    def test_has_true_after_put(self, local_store: NativeStore, blob_file: Path, blob_digest: str) -> None:
        local_store.put(blob_file)
        assert local_store.has(blob_digest) is True

    def test_has_false_before_put(self, local_store: NativeStore, blob_digest: str) -> None:
        assert local_store.has(blob_digest) is False

    def test_fetch_byte_identical_into_nested_dest(
        self,
        local_store: NativeStore,
        blob_file: Path,
        blob_content: bytes,
        blob_digest: str,
        tmp_path: Path,
    ) -> None:
        local_store.put(blob_file)
        dest = tmp_path / "deep" / "nested" / "fetched.nc"
        local_store.fetch(blob_digest, dest)
        assert dest.read_bytes() == blob_content

    def test_fetch_missing_raises_file_not_found(
        self, local_store: NativeStore, blob_digest: str, tmp_path: Path
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
    def test_blob_path_rejects_bad_digest(self, local_store: NativeStore, bad_digest: str) -> None:
        with pytest.raises(ValueError, match="Invalid sha256 digest"):
            local_store.has(bad_digest)

    def test_fetch_digest_mismatch_raises_value_error(
        self, local_store: NativeStore, blob_file: Path, blob_digest: str, tmp_path: Path
    ) -> None:
        """Corrupt the stored blob; fetch should detect the mismatch."""
        local_store.put(blob_file)
        root = local_store.root
        assert root is not None
        (root / blob_digest[:2] / blob_digest).write_bytes(b"corrupted content")
        with pytest.raises(ValueError):
            local_store.fetch(blob_digest, tmp_path / "out.nc")

    def test_put_idempotent(self, local_store: NativeStore, blob_file: Path, blob_digest: str) -> None:
        """Putting the same blob twice must succeed and return the same digest."""
        assert local_store.put(blob_file) == local_store.put(blob_file) == blob_digest

    def test_two_level_layout(self, local_store: NativeStore, blob_file: Path, blob_digest: str) -> None:
        """Blobs must be stored at root/<digest[:2]>/<digest>."""
        local_store.put(blob_file)
        root = local_store.root
        assert root is not None
        assert (root / blob_digest[:2] / blob_digest).exists()

    def test_preflight_creates_and_accepts_root(self, tmp_path: Path) -> None:
        store = NativeStore(url=str(tmp_path / "new-store"))
        assert store.root is not None
        assert not store.root.exists()
        store.preflight()  # must not raise; creates the root
        assert store.root.is_dir()

    def test_preflight_raises_when_not_writable(self, tmp_path: Path) -> None:
        root = tmp_path / "ro-store"
        root.mkdir()
        root.chmod(0o500)  # read+execute, not writable
        try:
            store = NativeStore(url=str(root))
            with pytest.raises(NativeStoreUnavailableError, match="not writable"):
                store.preflight()
        finally:
            root.chmod(0o700)  # restore so tmp cleanup can remove it

    def test_local_store_needs_no_write_config(self, tmp_path: Path) -> None:
        """A local store is always read/write, so it never needs routing config."""
        store = NativeStore(url=str(tmp_path / "store"))
        assert store.write is None
        blob = tmp_path / "blob.bin"
        blob.write_bytes(b"abc")
        assert store.has(store.put(blob))

    def test_file_url_resolves_to_absolute_root(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        for url in [root.as_uri(), f"file:{root}"]:  # file:///abs and single-slash file:/abs
            assert NativeStore(url=url).root == root

    def test_windows_drive_letter_is_a_local_path(self) -> None:
        """A single-character scheme is a drive letter, not an unrecognised URL scheme."""
        assert NativeStore(url="C:/native-store").root == Path("C:/native-store")

    def test_file_url_with_host_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="host component"):
            NativeStore(url="file://store/blobs")

    def test_unsupported_scheme_raises_value_error(self) -> None:
        # An unrecognised remote scheme must fail loudly, not be coerced to a local path.
        for url in ["s3://bucket/blobs", "gs://bucket/blobs", "ftp://host/blobs"]:
            with pytest.raises(ValueError, match="not recognised"):
                NativeStore(url=url)


class TestAnonymousRemoteStore:
    """An ``http(s)`` URL without credentials reads through pooch and cannot write."""

    def test_put_raises_not_implemented(self, tmp_path: Path, blob_file: Path) -> None:
        store = NativeStore(url=REMOTE_URL, cache_dir=tmp_path)
        with pytest.raises(NotImplementedError, match="public-read"):
            store.put(blob_file)

    def test_has_reports_the_pooch_cache(self, tmp_path: Path, blob_digest: str) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        store = NativeStore(url=REMOTE_URL, cache_dir=cache_dir)
        assert store.has(blob_digest) is False
        (cache_dir / blob_digest).write_bytes(b"cached")
        assert store.has(blob_digest) is True

    def test_fetch_byte_identical(
        self, mocker: MockerFixture, blob_content: bytes, blob_digest: str, tmp_path: Path
    ) -> None:
        """A fetch downloads a blob and produces byte-identical content."""
        _patch_pooch_create(mocker, {blob_digest: blob_content})

        store = NativeStore(url=REMOTE_URL, cache_dir=tmp_path / "cache")
        dest = tmp_path / "fetched.nc"
        store.fetch(blob_digest, dest)

        assert dest.read_bytes() == blob_content

    def test_fetch_hash_verified(self, mocker: MockerFixture, blob_digest: str, tmp_path: Path) -> None:
        """Corrupt content served under the correct digest name must be detected."""
        _patch_pooch_create(mocker, {blob_digest: b"corrupt bytes, the digest will not match"})

        store = NativeStore(url=REMOTE_URL, cache_dir=tmp_path / "cache")
        with pytest.raises(ValueError, match="does not match"):
            store.fetch(blob_digest, tmp_path / "out.nc")

    def test_reading_without_a_cache_dir_raises(self, blob_digest: str, tmp_path: Path) -> None:
        """An anonymous read needs somewhere to cache, so a store without one fails loudly."""
        store = NativeStore(url=REMOTE_URL)
        with pytest.raises(ValueError, match="needs a cache directory"):
            store.has(blob_digest)
        with pytest.raises(ValueError, match="needs a cache directory"):
            store.fetch(blob_digest, tmp_path / "out.nc")

    @pytest.mark.parametrize("bad_digest", ["../../etc/passwd", "not-hex", "0" * 63])
    def test_rejects_bad_digest_before_touching_the_cache(self, tmp_path: Path, bad_digest: str) -> None:
        """A hostile digest cannot escape the cache directory or reach the network."""
        store = NativeStore(url=REMOTE_URL, cache_dir=tmp_path / "cache")
        with pytest.raises(ValueError, match="Invalid sha256 digest"):
            store.has(bad_digest)
        with pytest.raises(ValueError, match="Invalid sha256 digest"):
            store.fetch(bad_digest, tmp_path / "out.nc")

    def test_credentials_are_not_in_the_repr(self) -> None:
        """A logged or asserted store must not leak its keys."""
        store = NativeStore(
            url=REMOTE_URL,
            write=S3WriteConfig(
                endpoint_url=S3_ENDPOINT,
                bucket=BUCKET,
                access_key_id="akid",
                secret_access_key="shhh",  # noqa: S106 - a fixture value, not a real secret
            ),
        )
        assert "akid" not in repr(store)
        assert "shhh" not in repr(store)

    def test_preflight_is_a_no_op(self, tmp_path: Path) -> None:
        NativeStore(url=REMOTE_URL, cache_dir=tmp_path).preflight()  # must not raise


class TestWritableRemoteStore:
    """
    The credentialed R2 write path, with a mocked boto3 client.

    The boto3 client is replaced by patching :func:`_s3_client`, so these tests neither
    touch the network nor exercise the ``@cache`` on the real factory.
    """

    def _store(self, mocker: MockerFixture, client, **kwargs) -> NativeStore:
        mocker.patch("climate_ref_core.regression.store._s3_client", return_value=client)
        write = {
            "endpoint_url": S3_ENDPOINT,
            "bucket": BUCKET,
            "access_key_id": "akid",
            "secret_access_key": "secret",
        }
        write.update(kwargs)
        return NativeStore(url=REMOTE_URL, write=S3WriteConfig(**write))

    def test_construct_requires_endpoint(self) -> None:
        with pytest.raises(ValueError, match="S3 endpoint URL"):
            S3WriteConfig(endpoint_url="", bucket=BUCKET)

    def test_construct_requires_bucket(self) -> None:
        with pytest.raises(ValueError, match="bucket name"):
            S3WriteConfig(endpoint_url=S3_ENDPOINT, bucket="")

    def test_client_threads_profile_to_factory(self, mocker: MockerFixture) -> None:
        factory = mocker.patch(
            "climate_ref_core.regression.store._s3_client", return_value=mocker.MagicMock()
        )
        store = NativeStore(
            url=REMOTE_URL,
            write=S3WriteConfig(endpoint_url=S3_ENDPOINT, bucket=BUCKET, profile="cf-ref"),
        )
        store.has("a" * 64)
        factory.assert_called_once_with(S3_ENDPOINT, "", "", "cf-ref")

    def test_key_validates_digest(self, mocker: MockerFixture) -> None:
        store = self._store(mocker, mocker.MagicMock())
        with pytest.raises(ValueError):
            store.has("not-a-valid-digest")

    def test_key_is_the_bare_digest(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        store = self._store(mocker, client)
        digest = "a" * 64
        store.has(digest)
        client.head_object.assert_called_once_with(Bucket=BUCKET, Key=digest)

    def test_put_uploads_when_absent(self, mocker: MockerFixture, tmp_path: Path) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("404", 404)
        store = self._store(mocker, client)
        blob = tmp_path / "blob.nc"
        blob.write_bytes(b"hello")
        digest = hashlib.sha256(b"hello").hexdigest()

        assert store.put(blob) == digest
        client.upload_file.assert_called_once_with(str(blob), BUCKET, digest)

    def test_put_is_idempotent_when_present(self, mocker: MockerFixture, tmp_path: Path) -> None:
        client = mocker.MagicMock()  # head_object succeeds, so the blob is already present
        store = self._store(mocker, client)
        blob = tmp_path / "blob.nc"
        blob.write_bytes(b"hello")

        assert store.put(blob) == hashlib.sha256(b"hello").hexdigest()
        client.upload_file.assert_not_called()

    def test_has_returns_true_when_present(self, mocker: MockerFixture) -> None:
        store = self._store(mocker, mocker.MagicMock())
        assert store.has("b" * 64) is True

    def test_has_returns_false_on_404(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("404", 404)
        store = self._store(mocker, client)
        assert store.has("b" * 64) is False

    def test_has_reraises_non_404(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("AccessDenied", 403)
        store = self._store(mocker, client)
        with pytest.raises(ClientError):
            store.has("c" * 64)

    def test_fetch_writes_and_verifies(self, mocker: MockerFixture, tmp_path: Path) -> None:
        content = b"native-bytes"
        digest = hashlib.sha256(content).hexdigest()
        client = mocker.MagicMock()
        client.download_file.side_effect = lambda bucket, key, dest: Path(dest).write_bytes(content)
        store = self._store(mocker, client)

        dest = tmp_path / "nested" / "blob.nc"
        store.fetch(digest, dest)
        assert dest.read_bytes() == content

    def test_fetch_hash_mismatch_raises(self, mocker: MockerFixture, tmp_path: Path) -> None:
        digest = hashlib.sha256(b"expected").hexdigest()
        client = mocker.MagicMock()
        client.download_file.side_effect = lambda bucket, key, dest: Path(dest).write_bytes(b"different")
        store = self._store(mocker, client)
        with pytest.raises(ValueError):
            store.fetch(digest, tmp_path / "blob.nc")

    def test_fetch_missing_raises_filenotfound(self, mocker: MockerFixture, tmp_path: Path) -> None:
        client = mocker.MagicMock()
        client.download_file.side_effect = _client_error("404", 404, "GetObject")
        store = self._store(mocker, client)
        with pytest.raises(FileNotFoundError):
            store.fetch("d" * 64, tmp_path / "blob.nc")

    def test_preflight_ok_on_404(self, mocker: MockerFixture) -> None:
        # 404 on the sentinel HEAD means authenticated with the object simply absent, so usable.
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("404", 404)
        store = self._store(mocker, client)
        store.preflight()  # must not raise
        # probe is a HEAD on a sentinel key, never a real digest
        _, kwargs = client.head_object.call_args
        assert kwargs["Key"] == ".ref-preflight-probe"

    def test_preflight_ok_on_200(self, mocker: MockerFixture) -> None:
        store = self._store(mocker, mocker.MagicMock())  # head_object succeeds
        store.preflight()  # must not raise

    def test_preflight_401_raises_actionable(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("Unauthorized", 401)
        store = self._store(mocker, client)
        with pytest.raises(NativeStoreUnavailableError, match="REF_NATIVE_STORE_PROFILE"):
            store.preflight()

    def test_preflight_400_treated_as_bad_credentials(self, mocker: MockerFixture) -> None:
        # A malformed access key id makes R2 return 400 on the HEAD, so treat it as a creds problem.
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("BadRequest", 400)
        store = self._store(mocker, client)
        with pytest.raises(NativeStoreUnavailableError, match="credentials were rejected or malformed"):
            store.preflight()

    def test_preflight_403_raises_actionable(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("AccessDenied", 403)
        store = self._store(mocker, client)
        with pytest.raises(NativeStoreUnavailableError, match="403"):
            store.preflight()

    def test_preflight_other_error_raises(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("InternalError", 500)
        store = self._store(mocker, client)
        with pytest.raises(NativeStoreUnavailableError, match="preflight failed"):
            store.preflight()


class TestBuildNativeStore:
    def test_writable_false_never_requires_creds(self, tmp_path: Path) -> None:
        """``writable=False`` returns a credential-free store for every supported URL."""
        for url in [
            str(tmp_path / "store"),
            (tmp_path / "store").as_uri(),
            REMOTE_URL,
        ]:
            cfg = _StubConfig(url=url, cache_dir=tmp_path / "cache")
            store = build_native_store(cfg, writable=False)
            assert store.write is None

    def test_local_url_is_writable_either_way(self, tmp_path: Path) -> None:
        cfg = _StubConfig(url=str(tmp_path / "store"), cache_dir=tmp_path / "cache")
        for writable in (False, True):
            store = build_native_store(cfg, writable=writable)
            assert store.root == tmp_path / "store"
            assert store.write is None

    def test_writable_true_remote_reads_creds_from_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("REF_NATIVE_STORE_ACCESS_KEY_ID", "akid")
        monkeypatch.setenv("REF_NATIVE_STORE_SECRET_ACCESS_KEY", "secret")
        monkeypatch.delenv("REF_NATIVE_STORE_PROFILE", raising=False)
        cfg = _StubConfig(url=REMOTE_URL, cache_dir=tmp_path / "cache")

        store = build_native_store(cfg, writable=True)
        assert store.root is None
        assert store.write is not None
        assert store.write.endpoint_url == S3_ENDPOINT
        assert store.write.bucket == BUCKET
        assert store.write.access_key_id == "akid"
        assert store.write.secret_access_key == "secret"  # noqa: S105 - a fixture value, not a real secret
        assert store.write.profile == ""

    def test_writable_true_remote_falls_back_to_default_chain(self, tmp_path: Path, monkeypatch) -> None:
        # Credentials must come from the environment, never from the (serialisable) config.
        for var in (
            "REF_NATIVE_STORE_ACCESS_KEY_ID",
            "REF_NATIVE_STORE_SECRET_ACCESS_KEY",
            "REF_NATIVE_STORE_PROFILE",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = _StubConfig(url=REMOTE_URL, cache_dir=tmp_path / "cache")

        store = build_native_store(cfg, writable=True)
        assert store.write is not None
        # Empty creds and empty profile mean boto3's default chain is used at client-build time.
        assert store.write.access_key_id == ""
        assert store.write.secret_access_key == ""
        assert store.write.profile == ""

    def test_writable_true_remote_reads_profile_from_env(self, tmp_path: Path, monkeypatch) -> None:
        # A named profile authenticates without putting secrets in the config or env creds.
        monkeypatch.delenv("REF_NATIVE_STORE_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("REF_NATIVE_STORE_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.setenv("REF_NATIVE_STORE_PROFILE", "cf-ref")
        cfg = _StubConfig(url=REMOTE_URL, cache_dir=tmp_path / "cache")

        store = build_native_store(cfg, writable=True)
        assert store.write is not None
        assert store.write.profile == "cf-ref"
        assert store.write.access_key_id == ""
        assert store.write.secret_access_key == ""

    def test_writable_true_remote_without_endpoint_raises(self, tmp_path: Path) -> None:
        cfg = _StubConfig(url=REMOTE_URL, cache_dir=tmp_path / "cache", s3_endpoint_url="")
        with pytest.raises(ValueError, match="S3 endpoint URL"):
            build_native_store(cfg, writable=True)

    def test_writable_true_remote_without_bucket_raises(self, tmp_path: Path) -> None:
        cfg = _StubConfig(url=REMOTE_URL, cache_dir=tmp_path / "cache", bucket="")
        with pytest.raises(ValueError, match="bucket name"):
            build_native_store(cfg, writable=True)

    def test_unsupported_scheme_raises_value_error(self, tmp_path: Path) -> None:
        for url in ["s3://bucket/blobs", "gs://bucket/blobs", "ftp://host/blobs"]:
            cfg = _StubConfig(url=url, cache_dir=tmp_path / "cache")
            with pytest.raises(ValueError, match="not recognised"):
                build_native_store(cfg, writable=False)
