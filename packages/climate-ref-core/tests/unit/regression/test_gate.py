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
    diagnostic_version: int = 1,
) -> Manifest:
    """Build a manifest with a given version and committed/native digests."""
    return Manifest(
        schema=SCHEMA_VERSION,
        test_case_version=version,
        diagnostic_version=diagnostic_version,
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
        # Seeding with a native baseline present: replay verifies it reproduces the bundle.
        manifest = make_manifest(1, native={"data.nc": NativeEntry("a" * 64, 10)})
        decision = decide_coupling(manifest, None)
        assert decision.action is Action.REPLAY
        assert "seeding" in decision.reason

    def test_newly_added_manifest_without_native_skips(self) -> None:
        # Seeding with no native baseline: nothing to replay; the committed bundle is the signal.
        decision = decide_coupling(make_manifest(1), None)
        assert decision.action is Action.SKIP
        assert "seeding" in decision.reason

    def test_catalog_integrity_failure_fails(self) -> None:
        # The input catalog drifted from the manifest's recorded hash.
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(1, {"output.json": "a" * 64})
        decision = decide_coupling(current, base, catalog_integrity_ok=False)
        assert decision.action is Action.FAIL
        assert "catalog.yaml does not match" in decision.reason

    def test_native_demint_skips_with_warning(self) -> None:
        # Native baseline removed (de-mint) with committed bundle unchanged: warn, do not fail.
        base = make_manifest(1, {"output.json": "a" * 64}, {"data.nc": NativeEntry("a" * 64, 10)})
        current = make_manifest(1, {"output.json": "a" * 64}, {})
        decision = decide_coupling(current, base)
        assert decision.action is Action.SKIP
        assert "WARNING" in decision.reason
        assert "de-mint" in decision.reason

    def test_version_bump_without_native_executes(self) -> None:
        # No native baseline to replay against, so the bump needs a full re-run.
        base = make_manifest(1)
        current = make_manifest(2)
        decision = decide_coupling(current, base)
        assert decision.action is Action.EXECUTE
        assert "1 -> 2" in decision.reason
        assert "no native baseline" in decision.reason

    def test_version_bump_without_native_executes_even_with_committed_change(self) -> None:
        # A bump authorises a new baseline, so a committed change is expected.
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(2, {"output.json": "b" * 64})
        decision = decide_coupling(current, base)
        assert decision.action is Action.EXECUTE

    def test_version_bump_with_native_replays(self) -> None:
        native = {"data.nc": NativeEntry("a" * 64, 10)}
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(2, {"output.json": "b" * 64}, native)
        decision = decide_coupling(current, base)
        assert decision.action is Action.REPLAY
        assert "1 -> 2" in decision.reason
        assert "native baseline present" in decision.reason

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
        native = {"data.nc": NativeEntry("a" * 64, 10)}
        base = make_manifest(1, {"output.json": "a" * 64}, native)
        current = make_manifest(1, {"output.json": "a" * 64}, native)
        decision = decide_coupling(current, base, extraction_changed=True)
        assert decision.action is Action.REPLAY
        assert "extraction code changed" in decision.reason

    def test_extraction_change_without_native_skips(self) -> None:
        # Extraction code changed but no native baseline exists to replay.
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(1, {"output.json": "a" * 64})
        decision = decide_coupling(current, base, extraction_changed=True)
        assert decision.action is Action.SKIP
        assert "no native baseline exists to replay" in decision.reason

    def test_no_change_skips(self) -> None:
        base = make_manifest(1, {"output.json": "a" * 64})
        current = make_manifest(1, {"output.json": "a" * 64})
        decision = decide_coupling(current, base, extraction_changed=False)
        assert decision.action is Action.SKIP
        assert "nothing to verify" in decision.reason


class TestDiagnosticVersionGate:
    """The staleness (2a) and conditioned-monotonic (2b) diagnostic_version branches."""

    def test_code_ahead_of_manifest_fails(self) -> None:
        # The diagnostic was bumped but the baseline was not re-minted: stale -> FAIL.
        manifest = make_manifest(1, diagnostic_version=1)
        decision = decide_coupling(manifest, None, code_diagnostic_version=2)
        assert decision.action is Action.FAIL
        assert "re-mint" in decision.reason

    def test_code_below_manifest_fails(self) -> None:
        # The code version dropped below the recorded baseline with no re-mint: FAIL.
        manifest = make_manifest(1, diagnostic_version=2)
        decision = decide_coupling(manifest, None, code_diagnostic_version=1)
        assert decision.action is Action.FAIL
        assert "append-only" in decision.reason

    def test_code_equals_manifest_falls_through(self) -> None:
        # Equality must NOT early-return: a matched baseline proceeds to the normal logic.
        manifest = make_manifest(1, native={"data.nc": NativeEntry("a" * 64, 10)}, diagnostic_version=2)
        decision = decide_coupling(manifest, None, code_diagnostic_version=2)
        assert decision.action is Action.REPLAY
        assert "seeding" in decision.reason

    def test_staleness_outranks_seeding(self) -> None:
        # A newly added manifest (no base) whose code is ahead is stale -> FAIL, not REPLAY/SKIP.
        manifest = make_manifest(1, native={"data.nc": NativeEntry("a" * 64, 10)}, diagnostic_version=1)
        decision = decide_coupling(manifest, None, code_diagnostic_version=2)
        assert decision.action is Action.FAIL

    def test_staleness_below_integrity(self) -> None:
        # An integrity failure outranks the staleness check.
        manifest = make_manifest(1, diagnostic_version=1)
        decision = decide_coupling(manifest, None, committed_integrity_ok=False, code_diagnostic_version=2)
        assert decision.action is Action.FAIL
        assert "does not match manifest.json digests" in decision.reason

    def test_diagnostic_version_decrease_bare_downgrade_fails(self) -> None:
        # A diagnostic_version decrease with the committed bundle unchanged is a bare downgrade.
        base = make_manifest(1, {"output.json": "a" * 64}, diagnostic_version=2)
        current = make_manifest(1, {"output.json": "a" * 64}, diagnostic_version=1)
        decision = decide_coupling(current, base, code_diagnostic_version=1)
        assert decision.action is Action.FAIL
        assert "bare downgrade" in decision.reason

    def test_diagnostic_version_decrease_with_reminted_bundle_falls_through(self) -> None:
        # A decrease that ships a re-minted bundle (committed changed) + a tcv bump is authorised.
        native = {"data.nc": NativeEntry("a" * 64, 10)}
        base = make_manifest(1, {"output.json": "a" * 64}, native, diagnostic_version=2)
        current = make_manifest(2, {"output.json": "b" * 64}, native, diagnostic_version=1)
        decision = decide_coupling(current, base, code_diagnostic_version=1)
        # 2b falls through; the tcv bump with native present routes to REPLAY.
        assert decision.action is Action.REPLAY
        assert "1 -> 2" in decision.reason

    def test_diagnostic_version_increase_proceeds(self) -> None:
        # An increase (with matching code) is not a 2b failure; normal logic applies.
        base = make_manifest(1, {"output.json": "a" * 64}, diagnostic_version=1)
        current = make_manifest(1, {"output.json": "a" * 64}, diagnostic_version=2)
        decision = decide_coupling(current, base, code_diagnostic_version=2)
        assert decision.action is Action.SKIP
        assert "nothing to verify" in decision.reason

    def test_legacy_none_skips_all_new_branches(self) -> None:
        # code_diagnostic_version=None preserves legacy behaviour: no staleness check at all.
        manifest = make_manifest(1, diagnostic_version=1)
        decision = decide_coupling(manifest, None, code_diagnostic_version=None)
        assert decision.action is Action.SKIP

    def test_tcv_bump_with_matched_diagnostic_version_still_executes(self) -> None:
        # A test_case_version bump with code == manifest.diagnostic_version behaves as before.
        base = make_manifest(1, diagnostic_version=2)
        current = make_manifest(2, diagnostic_version=2)
        decision = decide_coupling(current, base, code_diagnostic_version=2)
        assert decision.action is Action.EXECUTE
        assert "1 -> 2" in decision.reason


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
