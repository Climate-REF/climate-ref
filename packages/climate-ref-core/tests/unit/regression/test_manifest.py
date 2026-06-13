"""
Tests for :mod:`climate_ref_core.regression.manifest`.
"""

import hashlib
import json
from pathlib import Path

import pooch.hashes
import pytest

from climate_ref_core.regression.manifest import (
    SCHEMA_VERSION,
    Manifest,
    NativeEntry,
    compute_committed_digests,
    sha256_bytes,
    sha256_file,
    verify_committed_integrity,
)


@pytest.fixture()
def tmp_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.bin"
    p.write_bytes(b"hello manifest test")
    return p


@pytest.fixture()
def regression_dir(tmp_path: Path) -> Path:
    d = tmp_path / "regression"
    d.mkdir()
    (d / "series.json").write_text('{"series": 1}\n', encoding="utf-8")
    (d / "diagnostic.json").write_text('{"diagnostic": 2}\n', encoding="utf-8")
    (d / "output.json").write_text('{"output": 3}\n', encoding="utf-8")
    return d


class TestSha256File:
    """``sha256_file`` must agree with pooch's own file_hash."""

    def test_agrees_with_pooch(self, tmp_file: Path) -> None:
        expected = pooch.hashes.file_hash(str(tmp_file), alg="sha256")
        assert sha256_file(tmp_file) == expected

    def test_returns_hex_string(self, tmp_file: Path) -> None:
        result = sha256_file(tmp_file)
        assert isinstance(result, str)
        # sha256 hex digest is exactly 64 characters
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestSha256Bytes:
    def test_known_value(self) -> None:
        data = b"abc"
        assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()

    def test_empty_bytes(self) -> None:
        assert sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()

    def test_agrees_with_sha256_file(self, tmp_file: Path) -> None:
        data = tmp_file.read_bytes()
        assert sha256_bytes(data) == sha256_file(tmp_file)


class TestSeedV1:
    def test_test_case_version_is_1(self) -> None:
        m = Manifest.seed_v1({"series.json": "abc123"})
        assert m.test_case_version == 1

    def test_native_is_empty(self) -> None:
        m = Manifest.seed_v1({"series.json": "abc123"})
        assert m.native == {}

    def test_schema_equals_schema_version(self) -> None:
        m = Manifest.seed_v1({})
        assert m.schema == SCHEMA_VERSION

    def test_committed_digests_stored(self) -> None:
        digests = {"series.json": "aa" * 32, "diagnostic.json": "bb" * 32}
        m = Manifest.seed_v1(digests)
        assert m.committed == digests


class TestDumpLoad:
    """dump then load must be byte-identical and stable."""

    def _make_manifest(self) -> Manifest:
        return Manifest(
            schema=SCHEMA_VERSION,
            test_case_version=2,
            committed={"output.json": "cc" * 32, "series.json": "dd" * 32},
            native={"foo/bar.nc": NativeEntry(sha256="ee" * 32, size=1024)},
        )

    def test_round_trip_values(self, tmp_path: Path) -> None:
        original = self._make_manifest()
        p = tmp_path / "manifest.json"
        original.dump(p)
        loaded = Manifest.load(p)
        assert loaded.schema == original.schema
        assert loaded.test_case_version == original.test_case_version
        assert loaded.committed == original.committed
        assert loaded.native == original.native

    def test_dump_then_load_byte_identical(self, tmp_path: Path) -> None:
        """Two successive dumps must produce the same bytes."""
        original = self._make_manifest()
        p1 = tmp_path / "m1.json"
        p2 = tmp_path / "m2.json"
        original.dump(p1)
        original.dump(p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_trailing_newline(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        self._make_manifest().dump(p)
        text = p.read_text(encoding="utf-8")
        assert text.endswith("\n")

    def test_stable_key_order(self, tmp_path: Path) -> None:
        """Top-level keys must be sorted (sort_keys=True)."""
        p = tmp_path / "manifest.json"
        self._make_manifest().dump(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        keys = list(data.keys())
        assert keys == sorted(keys)

    def test_native_entry_round_trip(self, tmp_path: Path) -> None:
        entry = NativeEntry(sha256="ff" * 32, size=512)
        m = Manifest(
            schema=SCHEMA_VERSION,
            test_case_version=1,
            committed={},
            native={"path/to/file.nc": entry},
        )
        p = tmp_path / "manifest.json"
        m.dump(p)
        loaded = Manifest.load(p)
        assert loaded.native["path/to/file.nc"] == entry

    def test_load_missing_keys_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps({"schema": SCHEMA_VERSION, "committed": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match=r"missing required keys \['test_case_version', 'native'\]"):
            Manifest.load(p)

    def test_load_malformed_native_entry_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        payload = {
            "schema": SCHEMA_VERSION,
            "test_case_version": 1,
            "committed": {},
            "native": {"file.nc": {"sha256": "aa" * 32}},  # missing "size"
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="malformed 'native' entry"):
            Manifest.load(p)


class TestComputeCommittedDigests:
    def test_all_three_present(self, regression_dir: Path) -> None:
        digests = compute_committed_digests(regression_dir)
        assert set(digests.keys()) == {"series.json", "diagnostic.json", "output.json"}

    def test_digest_values_match_sha256_file(self, regression_dir: Path) -> None:
        digests = compute_committed_digests(regression_dir)
        for relpath, digest in digests.items():
            assert digest == sha256_file(regression_dir / relpath)

    def test_missing_file_excluded(self, regression_dir: Path) -> None:
        (regression_dir / "output.json").unlink()
        digests = compute_committed_digests(regression_dir)
        assert "output.json" not in digests
        assert "series.json" in digests
        assert "diagnostic.json" in digests

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert compute_committed_digests(empty) == {}


class TestVerifyCommittedIntegrity:
    def _make_manifest_for(self, regression_dir: Path) -> Manifest:
        digests = compute_committed_digests(regression_dir)
        return Manifest.seed_v1(digests)

    def test_clean_bundle_returns_empty_list(self, regression_dir: Path) -> None:
        manifest = self._make_manifest_for(regression_dir)
        result = verify_committed_integrity(manifest, regression_dir)
        assert result == []

    def test_tampered_file_returns_named_mismatch(self, regression_dir: Path) -> None:
        manifest = self._make_manifest_for(regression_dir)
        # Tamper with series.json
        (regression_dir / "series.json").write_text('{"series": 999}\n', encoding="utf-8")
        result = verify_committed_integrity(manifest, regression_dir)
        assert len(result) == 1
        assert "series.json" in result[0]

    def test_missing_file_returns_named_mismatch(self, regression_dir: Path) -> None:
        manifest = self._make_manifest_for(regression_dir)
        (regression_dir / "diagnostic.json").unlink()
        result = verify_committed_integrity(manifest, regression_dir)
        assert len(result) == 1
        assert "diagnostic.json" in result[0]

    def test_multiple_tampers_all_reported(self, regression_dir: Path) -> None:
        manifest = self._make_manifest_for(regression_dir)
        (regression_dir / "series.json").write_text("changed\n", encoding="utf-8")
        (regression_dir / "output.json").unlink()
        result = verify_committed_integrity(manifest, regression_dir)
        assert len(result) == 2
        reported_files = {line.split(":")[0] for line in result}
        assert reported_files == {"series.json", "output.json"}
