"""
Regression-baseline primitives for diagnostic test cases (RFC 0005).

This package holds the building blocks for the two-bundle regression model:

- :mod:`~climate_ref_core.regression.manifest` — the ``manifest.json`` model and
  sha256 digest helpers coupling a committed bundle to its native outputs.
- :mod:`~climate_ref_core.regression.store` — the content-addressed ``NativeStore``
  Protocol and its local / public-read implementations.
- :mod:`~climate_ref_core.regression.compare` — the tolerant JSON comparator used
  to gate committed bundles.
- :mod:`~climate_ref_core.regression.capture` — capture of a committed bundle and a
  native snapshot, built on :func:`climate_ref_core.output_files.copy_execution_outputs`
  so the captured set is exactly what production persists.

These primitives are pure: they take plain paths and data, with no dependency on the
application ``Config`` or database. The ``ref test-cases`` CLI wires them together.
"""

from climate_ref_core.regression.capture import (
    build_native_snapshot,
    capture_execution,
    materialise_native,
    write_committed_bundle,
)
from climate_ref_core.regression.compare import (
    DEFAULT_TOLERANCE,
    Tolerance,
    assert_bundle_regression,
    compare_json_content,
    resolve_tolerance,
)
from climate_ref_core.regression.manifest import (
    SCHEMA_VERSION,
    Manifest,
    NativeEntry,
    compute_committed_digests,
    sha256_bytes,
    sha256_file,
    verify_committed_integrity,
)
from climate_ref_core.regression.store import (
    LocalFilesystemStore,
    NativeStore,
    PoochReadStore,
    R2WriteStore,
    build_native_store,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "SCHEMA_VERSION",
    "LocalFilesystemStore",
    "Manifest",
    "NativeEntry",
    "NativeStore",
    "PoochReadStore",
    "R2WriteStore",
    "Tolerance",
    "assert_bundle_regression",
    "build_native_snapshot",
    "build_native_store",
    "capture_execution",
    "compare_json_content",
    "compute_committed_digests",
    "materialise_native",
    "resolve_tolerance",
    "sha256_bytes",
    "sha256_file",
    "verify_committed_integrity",
    "write_committed_bundle",
]
