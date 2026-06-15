"""
Tests for :mod:`climate_ref_core.regression.gate`.

The full decision matrix of :func:`decide_coupling` is exercised offline; no I/O
or git interaction is involved.
"""

from __future__ import annotations

import pytest

from climate_ref_core.regression.gate import (
    Action,
    decide_coupling,
    paths_under,
)
from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest, NativeEntry


def make_manifest(
    version: int,
    committed: dict[str, str] | None = None,
    native: dict[str, NativeEntry] | None = None,
) -> Manifest:
    """Build a manifest with a given version and committed/native digests."""
    return Manifest(
        schema=SCHEMA_VERSION,
        test_case_version=version,
        committed=committed if committed is not None else {"output.json": "a" * 64},
        native=native if native is not None else {},
    )


class TestDecideCoupling:
    def test_no_manifest_no_base_skips(self) -> None:
        # Never managed: no manifest on this branch and none on the base.
        decision = decide_coupling(None, None)
        assert decision.action is Action.SKIP
        assert "not under regression-baseline management" in decision.reason

    def test_manifest_deletion_fails(self) -> None:
        # The manifest existed on the base branch and was removed: fail closed.
        decision = decide_coupling(None, make_manifest(1))
        assert decision.action is Action.FAIL
        assert "removed without review" in decision.reason

    def test_committed_integrity_failure_fails(self) -> None:
        # The committed bundle on disk drifted from the manifest digests.
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(1, {"output.json": "a" * 64})
        decision = decide_coupling(current, base, committed_integrity_ok=False)
        assert decision.action is Action.FAIL
        assert "does not match manifest.json digests" in decision.reason

    def test_integrity_failure_outranks_seeding(self) -> None:
        # Even a newly added manifest must describe its on-disk bundle.
        decision = decide_coupling(make_manifest(1), None, committed_integrity_ok=False)
        assert decision.action is Action.FAIL

    def test_native_change_replays(self) -> None:
        # Native re-minted (different blob) with committed + version unchanged.
        base = make_manifest(1, {"output.json": "a" * 64}, {"data.nc": NativeEntry("a" * 64, 10)})
        current = make_manifest(1, {"output.json": "a" * 64}, {"data.nc": NativeEntry("b" * 64, 12)})
        decision = decide_coupling(current, base)
        assert decision.action is Action.REPLAY
        assert "native baseline changed" in decision.reason

    def test_newly_added_manifest_seeds_with_replay(self) -> None:
        decision = decide_coupling(make_manifest(1), None)
        assert decision.action is Action.REPLAY
        assert "seeding" in decision.reason

    def test_version_bump_executes(self) -> None:
        base = make_manifest(1)
        current = make_manifest(2)
        decision = decide_coupling(current, base)
        assert decision.action is Action.EXECUTE
        assert "1 -> 2" in decision.reason

    def test_version_bump_executes_even_with_committed_change(self) -> None:
        # A bump authorises a new baseline, so a committed change is expected.
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(2, {"output.json": "b" * 64})
        decision = decide_coupling(current, base)
        assert decision.action is Action.EXECUTE

    def test_committed_change_without_bump_fails(self) -> None:
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(1, {"output.json": "b" * 64})
        decision = decide_coupling(current, base)
        assert decision.action is Action.FAIL
        assert "without a test_case_version bump" in decision.reason

    def test_version_decrease_fails(self) -> None:
        base = make_manifest(3)
        current = make_manifest(2)
        decision = decide_coupling(current, base)
        assert decision.action is Action.FAIL
        assert "monotonic" in decision.reason

    def test_extraction_change_replays_when_bundle_unchanged(self) -> None:
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(1, {"output.json": "a" * 64})
        decision = decide_coupling(current, base, extraction_changed=True)
        assert decision.action is Action.REPLAY
        assert "extraction code changed" in decision.reason

    def test_no_change_skips(self) -> None:
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(1, {"output.json": "a" * 64})
        decision = decide_coupling(current, base, extraction_changed=False)
        assert decision.action is Action.SKIP
        assert "nothing to verify" in decision.reason


class TestPathsUnder:
    def test_file_directly_under_root_matches(self) -> None:
        assert paths_under(["pkg/src/mod.py"], ["pkg/src"])

    def test_file_equal_to_root_matches(self) -> None:
        assert paths_under(["pkg/src"], ["pkg/src"])

    def test_root_trailing_slash_normalised(self) -> None:
        assert paths_under(["pkg/src/mod.py"], ["pkg/src/"])

    def test_no_match_returns_false(self) -> None:
        assert not paths_under(["other/mod.py"], ["pkg/src"])

    def test_prefix_collision_does_not_match(self) -> None:
        # "pkg/src2" must not match root "pkg/src".
        assert not paths_under(["pkg/src2/mod.py"], ["pkg/src"])

    def test_empty_root_never_matches(self) -> None:
        assert not paths_under(["anything.py"], [""])

    def test_multiple_roots(self) -> None:
        assert paths_under(["b/x.py"], ["a", "b"])

    @pytest.mark.parametrize("changed", [[], ["unrelated.txt"]])
    def test_empty_or_unrelated(self, changed: list[str]) -> None:
        assert not paths_under(changed, ["pkg/src"])
