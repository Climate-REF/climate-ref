"""
Regression-baseline primitives for diagnostic test cases.

This package holds the building blocks for the two-bundle regression model,
where we capture and commit a small, sanitised **committed bundle** of artefacts for a test case.
These committed bundles are tracked in git and form the regression baseline for the test case.

We also snapshot the native files persisted by the execution.
These files are typically large and not always portable,
so we keep them out of git and refer to them by their sha256 digest in the committed bundle.
These data will be able to be fetched from an object store in CI and replayed locally for debugging.
"""

from climate_ref_core.regression.capture import (
    build_native_snapshot,
    capture_execution,
    materialise_native,
    write_committed_bundle,
)
from climate_ref_core.regression.compare import (
    Tolerance,
    assert_bundle_regression,
    compare_json_content,
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
    "sha256_bytes",
    "sha256_file",
    "verify_committed_integrity",
    "write_committed_bundle",
]
