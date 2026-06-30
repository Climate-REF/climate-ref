"""
CI coupling gate for test case regression bundles.

For each test case, CI must decide *how* to verify the regression baseline in a pull request,
based on what changed relative to the base branch:

- **EXECUTE** — re-run the diagnostic end-to-end and compare the regenerated committed
    bundle to the in-repo copy (tolerant compare).
    Required when the author has bumped ``test_case_version`` to authorise a new baseline
    but no native baseline exists to replay against.
- **REPLAY** — materialise the cached native baseline and re-run only
    ``build_execution_result`` against it, comparing to the committed bundle.
    Cheap and anonymous; used when extraction code changed but the committed bundle did
    not, to seed a newly added manifest, and to verify a ``test_case_version`` bump that ships native blobs
    — but only when native blobs exist to replay.
- **FAIL** — the committed bundle (or its input catalog) changed without a
    ``test_case_version`` bump (an unauthorised baseline change), or the version moved backwards.
- **SKIP** — nothing relevant to this test case changed, or the case is not yet under
    regression-baseline management.

See [Regression Baselines](https://climate-ref.readthedocs.io/en/latest/background/regression-baselines/)
for more information about the motivation and workflow for regression baselines.
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


def decide_coupling(  # noqa: PLR0911, PLR0912
    manifest: Manifest | None,
    base_manifest: Manifest | None,
    *,
    extraction_changed: bool = False,
    committed_integrity_ok: bool = True,
    catalog_integrity_ok: bool = True,
) -> GateDecision:
    """
    Decide how CI should verify a single test case's regression baseline.

    The decision is pure: it performs no I/O.
    All on-disk reality is summarised by its arguments
    — the two manifests, whether the diagnostic's extraction code
    changed, and whether the committed bundle on disk still matches the current
    manifest's digests.

    The gate fails if any state cannot be positively verified.
    A deleted managed manifest, a committed bundle that drifted from its manifest,
    an input catalog that drifted from its manifest
    is a failure rather than a silent skip.

    Changes to the native baseline are not failures.
    This is due to the workflow for minting requires credentials.
    This means fork contributors cannot author or edit native blobs.
    ``replay`` is therefore only selected when native blobs actually exist to replay;
    an absent or removed native baseline downgrades to ``skip`` (with a warning in the reason),
    never ``fail``.

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
        Whether code that influences ``build_execution_result`` for this test case
        changed in this pull request.
        Only consulted when the committed bundle is unchanged and the version was not bumped.
    committed_integrity_ok
        Whether the committed bundle on disk matches the current manifest's ``committed`` digests exactly
        (no edited, added, or removed committed file).
        The caller computes this against the working tree.
        ``False`` means the manifest no longer describes the bundle it is supposed to gate,
        which is a hard failure.

        Ignored when ``manifest`` is ``None`` (nothing to verify).
    catalog_integrity_ok
        Whether the test case's input ``catalog.yaml`` still matches the current
        manifest's ``catalog_hash``.
        The caller computes this against the working tree.
        ``False`` means the inputs changed without the baseline being regenerated,
        so the committed bundle no longer reflects its inputs — a hard failure.

        Always ``True`` when the manifest carries no ``catalog_hash``

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
    # If the bundle on disk drifted from the manifest digests
    # (an edit/add/remove without regenerating the manifest), the metadata comparisons below are meaningless.
    if not committed_integrity_ok:
        return GateDecision(
            Action.FAIL,
            "committed bundle on disk does not match manifest.json digests; "
            "regenerate the manifest with `ref test-cases run` after changing the bundle",
        )

    # The input catalog must still describe the baseline it produced.
    if not catalog_integrity_ok:
        return GateDecision(
            Action.FAIL,
            "input catalog.yaml does not match manifest.json catalog_hash; "
            "regenerate the baseline with `ref test-cases run` after changing the inputs",
        )

    if base_manifest is None:
        # Seeding a newly added manifest.
        # Replay only verifies something when native blobs exist
        if manifest.native:
            return GateDecision(
                Action.REPLAY,
                "manifest newly added (seeding); replaying native baseline against committed bundle",
            )
        return GateDecision(
            Action.SKIP,
            "manifest newly added (seeding) with no native baseline; "
            "the committed bundle is the only signal and is reviewed in the diff",
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
        version_change = f"{base_manifest.test_case_version} -> {manifest.test_case_version}"
        if manifest.native:
            # A native baseline ships with the bump, so the replay can prove the new committed bundle
            # actually reproduces from those blobs.
            return GateDecision(
                Action.REPLAY,
                f"test_case_version bumped ({version_change}) with a native baseline present; "
                "replaying to confirm the native baseline reproduces the new committed bundle",
            )
        return GateDecision(
            Action.EXECUTE,
            f"test_case_version bumped ({version_change}) with no native baseline to replay; "
            "full end-to-end re-run required to verify the new committed bundle",
        )

    if committed_changed:
        return GateDecision(
            Action.FAIL,
            "committed bundle changed without a test_case_version bump; "
            "bump test_case_version to authorise the new baseline",
        )

    if native_changed:
        if manifest.native:
            # Native blobs were re-authored (re-minted) without a version bump.
            # The committed bundle is unchanged,
            # so verify the new native snapshot still reproduces it rather than skipping unverified.
            return GateDecision(
                Action.REPLAY,
                "native baseline changed with committed bundle unchanged; "
                "replaying to confirm the new native snapshot reproduces the committed bundle",
            )
        # De-mint: the native baseline was removed while the committed bundle stayed.
        return GateDecision(
            Action.SKIP,
            "WARNING: native baseline removed (de-mint) with committed bundle unchanged; "
            "the committed bundle still gates this case but native replay is no longer possible",
        )

    if extraction_changed:
        if manifest.native:
            return GateDecision(
                Action.REPLAY,
                "extraction code changed with committed bundle unchanged; "
                "replaying cached native baseline to verify",
            )
        return GateDecision(
            Action.SKIP,
            "extraction code changed but no native baseline exists to replay; "
            "the committed bundle is unchanged and remains the only signal",
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
