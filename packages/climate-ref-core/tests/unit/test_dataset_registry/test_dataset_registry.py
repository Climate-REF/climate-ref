import importlib.resources
from pathlib import Path

import pytest

from climate_ref_core.dataset_registry import (
    DatasetRegistryManager,
    _verify_hash_matches,
    dataset_registry_manager,
    fetch_all_files,
    validate_registry_cache,
)

NUM_OBS4REF_FILES = 58


@pytest.fixture
def fake_registry_file():
    file_path = Path(importlib.resources.files("climate_ref_core") / "fake_registry.txt")

    yield file_path, "climate_ref_core", "fake_registry.txt"

    # Clean up the fake registry file after the test
    if file_path.exists():
        file_path.unlink()


class TestDatasetRegistry:
    def setup_registry_file(self, fake_registry_file):
        registry_path, package, resource = fake_registry_file
        # Create a dummy resource file
        with registry_path.open("w") as f:
            f.write("file1.txt sha256:checksum1\n")
            f.write("file2.txt sha256:checksum2\n")
        return package, resource

    def test_dataset_registry(self):
        registry = DatasetRegistryManager()
        assert isinstance(registry, DatasetRegistryManager)
        assert len(registry._registries) == 0

    def test_register(self, fake_registry_file):
        registry = DatasetRegistryManager()
        name = "test_registry"
        base_url = "http://example.com"

        package, resource = self.setup_registry_file(fake_registry_file)

        registry.register(name, base_url, package, resource)

        assert name in registry._registries
        r = registry._registries[name]
        assert r.base_url == base_url + "/"
        assert len(r.registry_files) == 2

    def test_register_invalid(self, fake_registry_file):
        registry = DatasetRegistryManager()
        name = "test_registry"
        base_url = "http://example.com"

        package, resource = self.setup_registry_file(fake_registry_file)
        with fake_registry_file[0].open("a") as f:
            f.write("invalid-line\n")

        with pytest.raises(OSError):
            registry.register(name, base_url, package, resource)

    def test_getitem_missing(self):
        registry = DatasetRegistryManager()
        with pytest.raises(KeyError):
            registry["missing_registry"]

    def test_getitem(self, mocker, fake_registry_file):
        registry = DatasetRegistryManager()
        name = "test_registry"
        base_url = "http://example.com"

        mock_pooch = mocker.patch("climate_ref_core.dataset_registry.pooch")
        package, resource = self.setup_registry_file(fake_registry_file)

        mock_pooch_instance = mock_pooch.create.return_value
        registry.register(name, base_url, package, resource)
        retrieved_registry = registry[name]

        assert retrieved_registry == mock_pooch_instance

    @pytest.mark.parametrize(
        "cache_name, expected", [(None, "climate_ref"), ("custom_cache", "custom_cache")]
    )
    def test_with_cache_name(self, mocker, fake_registry_file, cache_name, expected):
        registry = DatasetRegistryManager()
        name = "test_registry"
        base_url = "http://example.com"

        mock_pooch = mocker.patch("climate_ref_core.dataset_registry.pooch")
        package, resource = self.setup_registry_file(fake_registry_file)

        registry.register(name, base_url, package, resource, cache_name=cache_name)

        mock_pooch.os_cache.assert_called_with(expected)
        assert name in registry._registries
        mock_pooch.create.assert_called_once()


@pytest.mark.parametrize("symlink", [True, False])
@pytest.mark.parametrize("verify", [True, False])
def test_fetch_all_files(mocker, tmp_path, symlink, verify):
    mock_verify = mocker.patch("climate_ref_core.dataset_registry._verify_hash_matches")

    downloaded_file = tmp_path / "out.txt"
    downloaded_file.write_text("foo")

    registry = dataset_registry_manager["obs4ref"]
    registry.fetch = mocker.MagicMock(return_value=downloaded_file)

    fetch_all_files(registry, "obs4ref", tmp_path, symlink=symlink, verify=verify)
    assert registry.fetch.call_count == NUM_OBS4REF_FILES

    key = "obs4REF/MOHC/HadISST-1-1/mon/ts/gn/v20210727/ts_mon_HadISST-1-1_PCMDI_gn_187001-201907.nc"
    expected_file = tmp_path / key

    assert expected_file.exists()
    assert expected_file.is_symlink() == symlink
    assert expected_file.read_text() == "foo"

    if verify:
        mock_verify.assert_any_call(expected_file, registry.registry[key])
    else:
        mock_verify.assert_not_called()


def test_verify_hash_matches(mocker, tmp_path):
    expected_hash = "sha256:expectedhashvalue"

    mock_hashes = mocker.patch("climate_ref_core.dataset_registry.pooch.hashes")
    mock_hashes.hash_algorithm.return_value = "sha256"
    mock_hashes.file_hash.return_value = "expectedhashvalue"

    file_path = tmp_path / "file.txt"
    file_path.touch()

    _verify_hash_matches(file_path, expected_hash)


def test_verify_hash_missing_file(tmp_path):
    expected_hash = "sha256:expectedhashvalue"

    file_path = tmp_path / "file.txt"

    with pytest.raises(FileNotFoundError, match=r"file.txt does not exist. Cannot verify hash"):
        _verify_hash_matches(file_path, expected_hash)


def test_verify_hash_differs(mocker, tmp_path):
    expected_hash = "sha256:expectedhashvalue"

    mock_hashes = mocker.patch("climate_ref_core.dataset_registry.pooch.hashes")
    mock_hashes.hash_algorithm.return_value = "sha256"
    mock_hashes.file_hash.return_value = "opps"

    file_path = tmp_path / "file.txt"
    file_path.touch()

    with pytest.raises(
        ValueError, match=f"does not match the known hash. expected {expected_hash} but got opps."
    ):
        _verify_hash_matches(file_path, expected_hash)


def test_fetch_all_files_no_output(mocker):
    registry = dataset_registry_manager["obs4ref"]
    registry.fetch = mocker.MagicMock()

    fetch_all_files(registry, "obs4ref", None)
    assert registry.fetch.call_count == NUM_OBS4REF_FILES


class TestValidateRegistryCache:
    """Tests for validate_registry_cache function."""

    def test_all_files_valid(self, mocker, tmp_path):
        """Test validation passes when all files exist with correct checksums."""
        mock_registry = mocker.Mock()
        mock_registry.registry = {
            "file1.txt": "sha256:abc123",
            "file2.txt": "sha256:def456",
        }
        mock_registry.abspath = tmp_path

        # Create cached files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        # Mock successful hash verification
        mocker.patch(
            "climate_ref_core.dataset_registry._verify_hash_matches",
            return_value=True,
        )

        errors = validate_registry_cache(mock_registry, "test-registry")
        assert errors == []

    def test_file_not_cached(self, mocker, tmp_path):
        """Test validation fails when a file is not cached."""
        mock_registry = mocker.Mock()
        mock_registry.registry = {
            "file1.txt": "sha256:abc123",
            "missing.txt": "sha256:def456",
        }
        mock_registry.abspath = tmp_path

        # Only create one file
        (tmp_path / "file1.txt").write_text("content1")

        mocker.patch(
            "climate_ref_core.dataset_registry._verify_hash_matches",
            return_value=True,
        )

        errors = validate_registry_cache(mock_registry, "test-registry")
        assert len(errors) == 1
        assert "File not cached: missing.txt" in errors[0]

    def test_checksum_mismatch(self, mocker, tmp_path):
        """Test validation fails when checksum doesn't match."""
        mock_registry = mocker.Mock()
        mock_registry.registry = {
            "file1.txt": "sha256:abc123",
        }
        mock_registry.abspath = tmp_path

        # Create cached file
        (tmp_path / "file1.txt").write_text("corrupted content")

        # Mock hash verification failure
        mocker.patch(
            "climate_ref_core.dataset_registry._verify_hash_matches",
            side_effect=ValueError("Hash mismatch"),
        )

        errors = validate_registry_cache(mock_registry, "test-registry")
        assert len(errors) == 1
        assert "Hash mismatch" in errors[0]

    def test_multiple_errors(self, mocker, tmp_path):
        """Test validation collects multiple errors."""
        mock_registry = mocker.Mock()
        mock_registry.registry = {
            "missing1.txt": "sha256:abc123",
            "missing2.txt": "sha256:def456",
            "corrupted.txt": "sha256:ghi789",
        }
        mock_registry.abspath = tmp_path

        # Only create one file (which will have wrong checksum)
        (tmp_path / "corrupted.txt").write_text("bad content")

        mocker.patch(
            "climate_ref_core.dataset_registry._verify_hash_matches",
            side_effect=ValueError("Hash mismatch"),
        )

        errors = validate_registry_cache(mock_registry, "test-registry")
        assert len(errors) == 3
        assert any("missing1.txt" in e for e in errors)
        assert any("missing2.txt" in e for e in errors)
        assert any("Hash mismatch" in e for e in errors)

    def test_empty_registry(self, mocker, tmp_path):
        """Test validation passes for empty registry."""
        mock_registry = mocker.Mock()
        mock_registry.registry = {}
        mock_registry.abspath = tmp_path

        errors = validate_registry_cache(mock_registry, "test-registry")
        assert errors == []
