from pathlib import Path

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from pytest_mock import MockerFixture

from climate_ref_core.regression.report_store import (
    ReportStore,
    build_report_store,
)
from climate_ref_core.regression.store import NativeStoreUnavailableError

REMOTE_URL = "https://reports.example.com"
S3_ENDPOINT = "https://account.r2.cloudflarestorage.com"
BUCKET = "ref-baselines-reports"
KEY = "912/0c7e1d4abc12/index.html"
HTML = "text/html; charset=utf-8"


def _client_error(code: str, status: int, operation: str = "HeadObject") -> ClientError:
    """Build a botocore ``ClientError`` with the given S3 error code / HTTP status."""
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


@pytest.fixture()
def page(tmp_path: Path) -> Path:
    p = tmp_path / "index.html"
    p.write_text("<p>report</p>", encoding="utf-8")
    return p


@pytest.fixture()
def local_store(tmp_path: Path) -> ReportStore:
    return ReportStore(url=str(tmp_path / "reports"))


class _StubConfig:
    """Minimal config double satisfying _ReportStoreConfigProtocol."""

    def __init__(self, url: str, s3_endpoint_url: str = S3_ENDPOINT, bucket: str = BUCKET) -> None:
        self._url = url
        self._s3_endpoint_url = s3_endpoint_url
        self._bucket = bucket

    @property
    def url(self) -> str:
        return self._url

    @property
    def s3_endpoint_url(self) -> str:
        return self._s3_endpoint_url

    @property
    def bucket(self) -> str:
        return self._bucket


class TestLocalStore:
    def test_put_lands_the_file_under_the_key(self, local_store: ReportStore, page: Path) -> None:
        url = local_store.put(KEY, page, HTML)

        assert local_store.root is not None
        landed = local_store.root / KEY
        assert landed.read_text(encoding="utf-8") == "<p>report</p>"
        assert url == landed.absolute().as_uri()
        assert url.startswith("file://")
        assert url.endswith(KEY)

    def test_put_overwrites(self, local_store: ReportStore, page: Path, tmp_path: Path) -> None:
        local_store.put(KEY, page, HTML)
        newer = tmp_path / "newer.html"
        newer.write_text("<p>newer</p>", encoding="utf-8")

        local_store.put(KEY, newer, HTML)

        assert local_store.root is not None
        assert (local_store.root / KEY).read_text(encoding="utf-8") == "<p>newer</p>"

    @pytest.mark.parametrize("key", ["../x", "a/../../x", "/abs/index.html", "a b/index.html", ""])
    def test_put_rejects_an_unsafe_key(self, local_store: ReportStore, page: Path, key: str) -> None:
        with pytest.raises(ValueError, match="report store key"):
            local_store.put(key, page, HTML)

    def test_url_for_is_the_root_joined_with_the_key(self, local_store: ReportStore) -> None:
        assert local_store.root is not None
        assert local_store.url_for(KEY) == (local_store.root / KEY).absolute().as_uri()

    def test_preflight_creates_the_root(self, local_store: ReportStore) -> None:
        local_store.preflight()

        assert local_store.root is not None
        assert local_store.root.is_dir()

    def test_preflight_reports_an_unwritable_root(
        self, local_store: ReportStore, mocker: MockerFixture
    ) -> None:
        mocker.patch("climate_ref_core.regression.report_store.os.access", return_value=False)

        with pytest.raises(NativeStoreUnavailableError, match="not writable"):
            local_store.preflight()

    def test_preflight_reports_an_uncreatable_root(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        store = ReportStore(url=str(blocker / "reports"))

        with pytest.raises(NativeStoreUnavailableError, match="could not be created"):
            store.preflight()


class TestRemoteStore:
    """The credentialed R2 write path, with a mocked boto3 client."""

    def _store(self, mocker: MockerFixture, client, **kwargs) -> ReportStore:
        mocker.patch("climate_ref_core.regression.store._s3_client", return_value=client)
        return build_report_store(_StubConfig(REMOTE_URL, **kwargs), writable=True)

    def test_url_for_is_served_from_the_base_url(self) -> None:
        store = ReportStore(url="https://h")

        assert store.url_for("1/a/index.html") == "https://h/1/a/index.html"

    def test_url_for_ignores_a_trailing_slash(self) -> None:
        assert ReportStore(url="https://h/").url_for("1/a/index.html") == "https://h/1/a/index.html"

    def test_put_sets_the_content_type(self, mocker: MockerFixture, page: Path) -> None:
        client = mocker.MagicMock()
        store = self._store(mocker, client)

        url = store.put(KEY, page, HTML)

        client.upload_file.assert_called_once_with(str(page), BUCKET, KEY, ExtraArgs={"ContentType": HTML})
        assert url == f"{REMOTE_URL}/{KEY}"

    def test_put_without_credentials_is_read_only(self, page: Path) -> None:
        store = ReportStore(url=REMOTE_URL)

        with pytest.raises(NotImplementedError, match="public-read store"):
            store.put(KEY, page, HTML)

    def test_preflight_accepts_a_missing_probe(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("404", 404)
        store = self._store(mocker, client)

        store.preflight()

        client.head_object.assert_called_once()

    def test_preflight_reports_absent_credentials(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = NoCredentialsError()
        store = self._store(mocker, client)

        with pytest.raises(NativeStoreUnavailableError, match="REF_REPORT_STORE_ACCESS_KEY_ID"):
            store.preflight()

    def test_preflight_rejects_bad_credentials(self, mocker: MockerFixture) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error("InvalidAccessKeyId", 401)
        store = self._store(mocker, client)

        with pytest.raises(NativeStoreUnavailableError, match="REF_REPORT_STORE_ACCESS_KEY_ID"):
            store.preflight()

    @pytest.mark.parametrize(
        "code, status, expected",
        [
            ("AccessDenied", 403, "access denied"),
            ("InternalError", 500, "preflight failed"),
        ],
    )
    def test_preflight_reports_other_failures(
        self, mocker: MockerFixture, code: str, status: int, expected: str
    ) -> None:
        client = mocker.MagicMock()
        client.head_object.side_effect = _client_error(code, status)
        store = self._store(mocker, client)

        with pytest.raises(NativeStoreUnavailableError, match=expected):
            store.preflight()

    def test_a_read_only_remote_store_has_nothing_to_preflight(self) -> None:
        ReportStore(url=REMOTE_URL).preflight()


class TestBuildReportStore:
    def test_reads_the_report_store_env_vars(self, mocker: MockerFixture, monkeypatch) -> None:
        monkeypatch.setenv("REF_REPORT_STORE_ACCESS_KEY_ID", "report-key")
        monkeypatch.setenv("REF_REPORT_STORE_SECRET_ACCESS_KEY", "report-secret")
        monkeypatch.setenv("REF_REPORT_STORE_PROFILE", "report-profile")
        monkeypatch.setenv("REF_NATIVE_STORE_ACCESS_KEY_ID", "native-key")
        monkeypatch.setenv("REF_NATIVE_STORE_SECRET_ACCESS_KEY", "native-secret")
        monkeypatch.setenv("REF_NATIVE_STORE_PROFILE", "native-profile")
        factory = mocker.patch(
            "climate_ref_core.regression.store._s3_client", return_value=mocker.MagicMock()
        )

        store = build_report_store(_StubConfig(REMOTE_URL), writable=True)
        store.put(KEY, Path(__file__), HTML)

        factory.assert_called_once_with(S3_ENDPOINT, "report-key", "report-secret", "report-profile")

    def test_read_only_needs_no_credentials(self) -> None:
        store = build_report_store(_StubConfig(REMOTE_URL), writable=False)

        assert store.write is None

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"s3_endpoint_url": ""}, "REF_REPORT_STORE_S3_ENDPOINT_URL"),
            ({"bucket": ""}, "REF_REPORT_STORE_BUCKET"),
        ],
    )
    def test_missing_routing_names_the_report_store_env_var(self, kwargs, expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            build_report_store(_StubConfig(REMOTE_URL, **kwargs), writable=True)

    def test_a_local_store_is_writable_without_credentials(self, tmp_path: Path) -> None:
        store = build_report_store(_StubConfig(str(tmp_path / "reports")), writable=True)

        assert store.write is None
        assert store.root == tmp_path / "reports"
