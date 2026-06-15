"""
Manifest model and digest utilities for test case regression bundles.

The manifest (``manifest.json``) lives alongside a test case's regression data.
It records:
- ``schema``: the manifest schema version (``SCHEMA_VERSION``).
- ``test_case_version``: a monotonic, author-bumped integer coupling the committed
  bundle to its native outputs.
- ``committed``: sha256 digests of the committed regression JSON artefacts,
  computed over the exact placeholder-substituted bytes as they sit on disk,
  so that a CI recompute is deterministic.
- ``native``: digests of the curated native output files, authored ONLY by ``mint``.

Digests use sha256 throughout for hashing files.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pooch.hashes
from attrs import asdict, frozen

from climate_ref_core.paths import safe_path

SCHEMA_VERSION: int = 1
"""Current manifest schema version."""

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
"""A 64-character lowercase hex sha256 digest."""


def _validate_digest(digest: str) -> str:
    """
    Validate that ``digest`` is a 64-character lowercase hex sha256 string.

    Used wherever a digest is taken from untrusted input (a hand-edited manifest,
    a store key) before it is used to build a filesystem path.

    Parameters
    ----------
    digest
        The candidate sha256 hex digest.

    Returns
    -------
    :
        The validated digest, unchanged.

    Raises
    ------
    ValueError
        If ``digest`` is not a 64-character lowercase hex string.
    """
    if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
        raise ValueError(
            f"Invalid sha256 digest {digest!r}: expected 64 lowercase hex characters."
        )
    return digest


COMMITTED_BUNDLE_FILES: tuple[str, ...] = (
    "series.json",
    "diagnostic.json",
    "output.json",
)
"""The committed CMEC artefacts tracked in git.

Their digests are tracked in :attr:`Manifest.committed`.
"""


def sha256_file(path: Path) -> str:
    """
    Compute the sha256 digest of a file.

    Reuses :func:`pooch.hashes.file_hash` so the digest agrees with pooch elsewhere.

    Parameters
    ----------
    path
        Path to the file to hash.

    Returns
    -------
    :
        The hex-encoded sha256 digest.
    """
    return pooch.hashes.file_hash(str(path), alg="sha256")


def sha256_bytes(data: bytes) -> str:
    """
    Compute the sha256 digest of an in-memory byte string.

    Parameters
    ----------
    data
        The bytes to hash.

    Returns
    -------
    :
        The hex-encoded sha256 digest.
    """
    return hashlib.sha256(data).hexdigest()


@frozen
class NativeEntry:
    """A single curated native output file recorded in the manifest."""

    sha256: str
    """Hex-encoded sha256 digest of the curated file."""

    size: int
    """Size of the curated file in bytes."""


@frozen
class Manifest:
    """
    The on-disk manifest for a test case regression bundle.

    Serialised as ``manifest.json`` with stable key ordering and a trailing newline,
    so repeated dumps are byte-identical.
    """

    schema: int
    """Manifest schema version; equals :data:`SCHEMA_VERSION` for current manifests."""

    test_case_version: int
    """Monotonic, author-bumped version coupling the bundle to its native outputs."""

    committed: dict[str, str]
    """Digests of committed regression JSON artefacts: ``{relpath: sha256}``."""

    native: dict[str, NativeEntry]
    """Digests of curated native output files: ``{relpath: NativeEntry}``."""

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """
        Load a manifest from ``manifest.json``.

        Parameters
        ----------
        path
            Path to the manifest file.

        Returns
        -------
        :
            The parsed manifest.

        Raises
        ------
        ValueError
            If the manifest is missing required keys or has malformed native entries
            (e.g. hand-edited or written by an incompatible version).
        """
        return cls.loads(path.read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def loads(cls, text: str, *, source: str = "<string>") -> Manifest:
        """
        Parse a manifest from its JSON text.

        Used when the manifest does not live on disk at parse time,
        e.g. when reading the base-branch copy via ``git show`` for the CI coupling gate.

        Parameters
        ----------
        text
            The manifest JSON.
        source
            A label for the text's origin, used in error messages.

        Returns
        -------
        :
            The parsed manifest.

        Raises
        ------
        ValueError
            If the manifest is missing required keys or has malformed native entries
            (e.g. hand-edited or written by an incompatible version).
        """
        data = json.loads(text)
        missing = [
            key
            for key in ("schema", "test_case_version", "committed", "native")
            if key not in data
        ]
        if missing:
            raise ValueError(
                f"Invalid manifest {source}: missing required keys {missing}. "
                "The manifest may be corrupted or written by an incompatible version; "
                "regenerate it with `ref test-cases run --force-regen`."
            )
        try:
            native = {
                relpath: NativeEntry(sha256=entry["sha256"], size=entry["size"])
                for relpath, entry in data["native"].items()
            }
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"Invalid manifest {source}: malformed 'native' entry ({exc!r}). "
                "Each entry must be a mapping with 'sha256' and 'size' keys."
            ) from exc
        # Reject hand-edited or hostile manifests that could escape the
        # destination directory or carry a malformed digest when materialised.
        for relpath, entry in native.items():
            try:
                safe_path(relpath, label="native path")
            except ValueError as exc:
                raise ValueError(f"Invalid manifest {source}: {exc}") from exc
            _validate_digest(entry.sha256)
        return cls(
            schema=data["schema"],
            test_case_version=data["test_case_version"],
            committed=dict(data["committed"]),
            native=native,
        )

    def dump(self, path: Path) -> None:
        """
        Write the manifest to ``manifest.json``.

        Keys are stably ordered (``sort_keys=True``) and a trailing newline is added,
        so ``dump`` followed by ``load`` round-trips byte-identically.

        Parameters
        ----------
        path
            Path to write the manifest to.
        """
        payload = {
            "schema": self.schema,
            "test_case_version": self.test_case_version,
            "committed": self.committed,
            "native": {
                relpath: asdict(entry) for relpath, entry in self.native.items()
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")

    @classmethod
    def seed_v1(cls, committed_digests: dict[str, str]) -> Manifest:
        """
        Create an initial manifest at ``test_case_version == 1`` with no native outputs.

        Parameters
        ----------
        committed_digests
            Digests of the committed regression JSON artefacts.

        Returns
        -------
        :
            A fresh manifest with ``test_case_version=1`` and ``native={}``.
        """
        return cls(
            schema=SCHEMA_VERSION,
            test_case_version=1,
            committed=dict(committed_digests),
            native={},
        )


def compute_committed_digests(regression_dir: Path) -> dict[str, str]:
    """
    Compute sha256 digests of the committed regression JSON artefacts.

    The digests are taken over the bytes exactly as they sit on disk (placeholder text included),
    so a CI recompute is deterministic. Only files that exist are included.

    Parameters
    ----------
    regression_dir
        The test case ``regression/`` directory.

    Returns
    -------
    :
        Mapping of ``{relpath: sha256}`` for each present committed artefact.
    """
    digests: dict[str, str] = {}
    for relpath in COMMITTED_BUNDLE_FILES:
        candidate = regression_dir / relpath
        if candidate.exists():
            digests[relpath] = sha256_file(candidate)
    return digests


def verify_committed_integrity(manifest: Manifest, regression_dir: Path) -> list[str]:
    """
    Check that the committed regression artefacts match the manifest digests.

    Used by the CI integrity check. An empty return value means the bundle is intact.

    Parameters
    ----------
    manifest
        The manifest holding the expected committed digests.
    regression_dir
        The test case ``regression/`` directory to verify against.

    Returns
    -------
    :
        A list of human-readable mismatch descriptions; empty when everything matches.
    """
    mismatches: list[str] = []
    for relpath, expected in manifest.committed.items():
        candidate = regression_dir / relpath
        if not candidate.exists():
            mismatches.append(
                f"{relpath}: missing on disk — expected at {candidate} (manifest sha256 {expected})"
            )
            continue
        actual = sha256_file(candidate)
        if actual != expected:
            mismatches.append(
                f"{relpath}: content differs from manifest — {candidate} "
                f"(manifest sha256 {expected}, on-disk sha256 {actual})"
            )
    return mismatches
