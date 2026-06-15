"""
CI coupling gate for test case regression bundles.

For each test case, CI must decide *how* to verify the regression baseline in a pull request,
based on what changed relative to the base branch:

- **EXECUTE** — re-run the diagnostic end-to-end and compare the regenerated committed bundle to the in-repo copy (tolerant compare).
    Required when the author has bumped ``test_case_version`` to authorise a new baseline.
- **REPLAY** — materialise the cached native baseline and re-run only ``build_execution_result`` against it,
    comparing to the committed bundle.
    Cheap and anonymous; used when extraction code changed but the committed bundle did not, and to seed a newly added manifest.
- **FAIL** — the committed bundle changed without a ``test_case_version`` bump (an unauthorised baseline change), or the version moved backwards.
- **SKIP** — nothing relevant to this test case changed, or the case is not yet under regression-baseline management.

The decision logic in :func:`decide_coupling` is pure.
It takes the current and base :class:`~climate_ref_core.regression.manifest.Manifest` plus a single ``extraction_changed`` signal.
Mapping a pull request's changed-file list onto the per-case= ``extraction_changed`` boolean is the caller's responsibility;
:func:`paths_under` is provided as a small helper for that mapping.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable

from attrs import frozen

from climate_ref_core.regression.manifest import Manifest


class Action(enum.Enum):
    """The verification action the CI gate selects for a single test case."""

    SKIP = "skip"
    """Nothing relevant changed, or the case is not under regression management."""

    REPLAY = "replay"
    """Replay the cached native baseline and compare to the committed bundle."""

    EXECUTE = "execute"
    """Re-run the diagnostic end-to-end and compare to the committed bundle."""

    FAIL = "fail"
    """The change is not permissible (unauthorised baseline change or bad version)."""


@frozen
class GateDecision:
    """The gate's decision for one test case: an :class:`Action` and why."""

    action: Action
    """The selected verification action."""

    reason: str
    """A human-readable explanation, surfaced in CI logs."""


def decide_coupling(  # noqa: PLR0911
    manifest: Manifest | None,
    base_manifest: Manifest | None,
    *,
    extraction_changed: bool = False,
    committed_integrity_ok: bool = True,
) -> GateDecision:
    """
    Decide how CI should verify a single test case's regression baseline.

    The decision is pure: it performs no I/O.
    All on-disk reality is summarised by its arguments
    — the two manifests, whether the diagnostic's extraction code
    changed, and whether the committed bundle on disk still matches the current
    manifest's digests.

    The gate **fails closed**. Any state it cannot positively verify
    (a deleted managed manifest, a committed bundle that drifted from its manifest)
    is treated as a failure rather than silently skipped.

    See the module docstring for the meaning of each :class:`Action`.

    Parameters
    ----------
    manifest
        The current ``manifest.json`` for the test case,
        or ``None`` if the case has no manifest on this branch (never managed, or deleted in this change).
    base_manifest
        The ``manifest.json`` as it exists on the base branch,
        or ``None`` if the manifest is newly added in this change (seeding).
    extraction_changed
        Whether code that influences ``build_execution_result`` for this test case changed in this pull request.
        Only consulted when the committed bundle is unchanged and the version was not bumped.
    committed_integrity_ok
        Whether the committed bundle on disk matches the current manifest's ``committed`` digests exactly
        (no edited, added, or removed committed file).
        The caller computes this against the working tree.
        ``False`` means the manifest no longer describes the bundle it is supposed to gate, which is a hard failure.

        Ignored when ``manifest`` is ``None`` (nothing to verify).

    Returns
    -------
    :
        The gate's decision, pairing an :class:`Action` with a reason.
    """
    if manifest is None:
        # Distinguish a never-managed case from the deletion of a managed baseline.
        # Deleting manifest.json must not be a silent way to disable the gate.
        if base_manifest is not None:
            return GateDecision(
                Action.FAIL,
                "manifest.json is absent but exists on the base branch; a managed "
                "regression baseline cannot be removed without review",
            )
        return GateDecision(
            Action.SKIP,
            "no manifest.json; test case not under regression-baseline management",
        )

    # The manifest must faithfully describe the committed bundle it gates.
    # If the bundle on disk drifted from the manifest digests (an edit/add/remove without  regenerating the manifest),
    # the metadata comparisons below are meaningless.
    if not committed_integrity_ok:
        return GateDecision(
            Action.FAIL,
            "committed bundle on disk does not match manifest.json digests; "
            "regenerate the manifest with `ref test-cases run` after changing the bundle",
        )

    if base_manifest is None:
        return GateDecision(
            Action.REPLAY,
            "manifest newly added (seeding); replaying native baseline against committed bundle",
        )

    if manifest.test_case_version < base_manifest.test_case_version:
        return GateDecision(
            Action.FAIL,
            f"test_case_version decreased ({base_manifest.test_case_version} -> "
            f"{manifest.test_case_version}); version must be monotonic",
        )

    version_bumped = manifest.test_case_version > base_manifest.test_case_version
    committed_changed = manifest.committed != base_manifest.committed
    native_changed = manifest.native != base_manifest.native

    if version_bumped:
        return GateDecision(
            Action.EXECUTE,
            f"test_case_version bumped ({base_manifest.test_case_version} -> "
            f"{manifest.test_case_version}); executing full run with tolerant compare",
        )

    if committed_changed:
        return GateDecision(
            Action.FAIL,
            "committed bundle changed without a test_case_version bump; "
            "bump test_case_version to authorise the new baseline",
        )

    if native_changed:
        # Native blobs were re-authored (re-minted) without a version bump.
        # The committed bundle is unchanged,
        # so verify the new native snapshot still reproduces it rather than skipping unverified.
        return GateDecision(
            Action.REPLAY,
            "native baseline changed with committed bundle unchanged; "
            "replaying to confirm the new native snapshot reproduces the committed bundle",
        )

    if extraction_changed:
        return GateDecision(
            Action.REPLAY,
            "extraction code changed with committed bundle unchanged; "
            "replaying cached native baseline to verify",
        )

    return GateDecision(
        Action.SKIP,
        "no committed-bundle, native, version, or extraction-code change; nothing to verify",
    )


def paths_under(changed_files: Iterable[str], roots: Iterable[str]) -> bool:
    """
    Return whether any changed file lies within one of the given directory roots.

    A small helper for deriving the ``extraction_changed`` signal from a pull request's changed-file list.
    Paths are compared textually as POSIX-style, repo-relative strings,
    so callers must normalise both ``changed_files`` and ``roots`` to the same convention
    (e.g. ``git diff --name-only`` output and a package source directory relative to the repo root).

    Parameters
    ----------
    changed_files
        Repo-relative paths changed in the pull request.
    roots
        Repo-relative directory prefixes to test against. A trailing slash is
        optional; an empty root never matches.

    Returns
    -------
    :
        ``True`` if any changed file equals or sits beneath any root.
    """
    normalised_roots = [root.rstrip("/") for root in roots if root.rstrip("/")]
    for changed in changed_files:
        for root in normalised_roots:
            if changed == root or changed.startswith(root + "/"):
                return True
    return False
