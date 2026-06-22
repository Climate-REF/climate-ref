"""
Integration tests for the RFC-0005 native-baseline round-trip.

These tests prove the native-baseline lifecycle ``run -> mint -> sync -> replay``
end-to-end against a *local* content-addressed store
(:class:`~climate_ref_core.regression.LocalFilesystemStore`) created inside ``tmp_path``.

This runs offline, without any credentials.


Notes on the shipped capture mechanics (verified against the source)
--------------------------------------------------------------------
For a ``result`` from a run, ``defn = result.definition``;
``output_dir = defn.output_directory``; ``fragment = defn.output_fragment()`` (a
method); ``scratch_root = output_dir.parent``.
A *separate* temporary directory is used as the ``results_base`` for
``capture_execution`` (it must differ from the scratch root), and the captured
native blob bytes live at ``results_base / fragment / <relpath>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from climate_ref.config import Config
from climate_ref.testing import TEST_DATA_DIR, TestCaseRunner
from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
)
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.output_files import from_placeholders
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputCV
from climate_ref_core.regression import (
    LocalFilesystemStore,
    Manifest,
    NativeEntry,
    Tolerance,
    assert_bundle_regression,
    capture_execution,
    materialise_native,
)
from climate_ref_core.testing import (
    TestCase,
    TestCasePaths,
    TestDataSpecification,
)

# A stable recipe-run directory name (no timestamp), so no ``<RECIPE_RUN>``
# sanitisation is required for the synthetic case.
_RECIPE_SUBDIR = "executions/recipe"

# The nested native data file the synthetic diagnostic writes and reads back.
# Relative to the execution output directory.
_NATIVE_DATA_RELPATH = f"{_RECIPE_SUBDIR}/data/synth_data.bin"


class _SyntheticNestedDiagnostic(Diagnostic):
    """
    A synthetic diagnostic that writes nested native outputs.

    Output layout under ``output_directory``::

        executions/recipe/
            data/
                synth_data.bin
            index.html
            run/
                main_log_debug.txt

    The CMEC ``output.json`` registers the nested data file in the ``data``
    section (so ``copy_execution_outputs`` curates it) and uses the *absolute
    paths* of ``index.html`` and ``main_log_debug.txt`` as ``html`` /
    ``provenance.log`` values.
    The ``html`` entry is additionally keyed by its absolute path, exercising the
    M2 key-sanitisation path in
    :func:`~climate_ref_core.regression.assert_bundle_regression`.

    The ``diagnostic.json`` carries a float metric so the tolerance comparison is
    exercised by the negative drift tests.
    """

    name = "Synthetic Nested"
    slug = "synthetic-nested"
    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=tuple(),
            group_by=None,
        ),
    )
    facets = ("source_id",)

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Synthetic round-trip test case (no external data required)",
                requests=None,
            ),
        ),
    )

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Write a stable, deterministic nested output tree under ``executions/recipe/``.

        Parameters
        ----------
        definition
            The execution definition carrying the output directory.
        """
        recipe_dir = definition.output_directory / _RECIPE_SUBDIR

        data_path = definition.output_directory / _NATIVE_DATA_RELPATH
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(b"SYNTH-NATIVE-DATA\x00\x01\x02")

        index_path = recipe_dir / "index.html"
        index_path.write_text("<html><body>synthetic</body></html>", encoding="utf-8")

        log_path = recipe_dir / "run" / "main_log_debug.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("INFO Synthetic run completed.\n", encoding="utf-8")

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Build the CMEC output bundle, reading back a nested native file.

        Reading back the nested data file makes the curated native set
        load-bearing: if the file is missing after materialisation, this raises
        and the replay genuinely fails.

        Parameters
        ----------
        definition
            The execution definition carrying the output directory.

        Returns
        -------
        :
            A successful :class:`~climate_ref_core.diagnostics.ExecutionResult`.
        """
        recipe_dir = definition.output_directory / _RECIPE_SUBDIR

        # HONESTY: read back the nested native data file.  ``copy_execution_outputs``
        # only captures files registered in the bundle, so on replay this file must
        # have been materialised from the store; otherwise this raises.
        data_path = definition.output_directory / _NATIVE_DATA_RELPATH
        data_size = data_path.stat().st_size
        if data_size == 0:  # pragma: no cover - defensive
            raise ValueError(f"Nested native data file {data_path} is empty")

        index_path = recipe_dir / "index.html"
        log_path = recipe_dir / "run" / "main_log_debug.txt"

        output_args = CMECOutput.create_template()

        # The nested data file: ``filename`` is the relative path under the output
        # directory so ``copy_execution_outputs`` (which resolves filenames against
        # the output directory) curates it into the native set.
        output_args[OutputCV.DATA.value]["synth_data"] = {
            OutputCV.FILENAME.value: _NATIVE_DATA_RELPATH,
            OutputCV.LONG_NAME.value: "Synthetic nested data",
            OutputCV.DESCRIPTION.value: f"Nested native data file ({data_size} bytes).",
        }

        # The html entry is keyed by an ABSOLUTE path (M2 key-sanitisation path).
        output_args[OutputCV.HTML.value][str(index_path)] = {
            OutputCV.FILENAME.value: str(index_path),
            OutputCV.LONG_NAME.value: "Results page",
            OutputCV.DESCRIPTION.value: "Synthetic results page.",
        }
        output_args[OutputCV.INDEX.value] = str(index_path)
        output_args[OutputCV.PROVENANCE.value][OutputCV.LOG.value] = str(log_path)

        cmec_output = CMECOutput.model_validate(output_args)

        metric_args: dict[str, Any] = {
            "DIMENSIONS": {
                "json_structure": ["region", "metric", "statistic"],
                "region": {"global": {}},
                "metric": {"synth": {}},
                "statistic": {"score": {}},
            },
            "RESULTS": {
                "global": {"synth": {"score": 1.23456789}},
            },
        }
        cmec_metric = CMECMetric.model_validate(metric_args)

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output,
            cmec_metric_bundle=cmec_metric,
        )


def _build_synthetic_provider() -> DiagnosticProvider:
    """Return a :class:`DiagnosticProvider` carrying the synthetic diagnostic."""
    provider = DiagnosticProvider("SyntheticTest", "0.0.1", slug="synthetic-test")
    provider.register(_SyntheticNestedDiagnostic())
    return provider


def _build_synthetic_datasets() -> ExecutionDatasetCollection:
    """
    Build a minimal :class:`ExecutionDatasetCollection` for the synthetic diagnostic.

    The synthetic ``execute`` reads no input files, so the dataset DataFrame only
    needs to satisfy the runner's ``path`` column check.
    """
    df = pd.DataFrame(
        {
            "instance_id": ["CMIP6.synthetic.tas"],
            "source_id": ["synthetic-model"],
            "variable_id": ["tas"],
            "path": ["/nonexistent/synthetic.nc"],
        }
    )
    collection = DatasetCollection(
        datasets=df,
        slug_column="instance_id",
        selector=(("source_id", "synthetic-model"),),
    )
    return ExecutionDatasetCollection({SourceDatasetType.CMIP6: collection})


def _run_synthetic(tmp_path: Path) -> ExecutionResult:
    """
    Run the synthetic diagnostic and return the result.

    Parameters
    ----------
    tmp_path
        A per-test temporary directory.

    Returns
    -------
    :
        The successful execution result.
    """
    config = Config.default()
    config.paths.results = tmp_path / "results"

    datasets = _build_synthetic_datasets()
    runner = TestCaseRunner(config=config, datasets=datasets)

    provider = _build_synthetic_provider()
    diag = provider.diagnostics()[0]

    return runner.run(diag, "default", tmp_path / "output", clean=True)


def _capture_synthetic(
    result: ExecutionResult,
    *,
    results_base: Path,
    regression_dir: Path,
    test_data_dir: Path,
) -> tuple[dict[str, str], dict[str, NativeEntry]]:
    """
    Run the shipped capture entrypoint for a synthetic result.

    Mirrors the capture-path mechanics used by ``ref test-cases mint``.

    Parameters
    ----------
    result
        The successful execution result.
    results_base
        A fresh temporary directory used as the persistence base (must differ from
        the scratch root).
    regression_dir
        The destination ``regression/`` directory for the committed bundle.
    test_data_dir
        The provider test-data directory for path substitution.

    Returns
    -------
    :
        ``(committed_digests, native_snapshot)``.
    """
    defn = result.definition
    output_dir = defn.output_directory
    fragment = defn.output_fragment()
    scratch_root = output_dir.parent

    return capture_execution(
        scratch_root,
        results_base,
        fragment,
        result,
        regression_dir=regression_dir,
        output_dir=output_dir,
        test_data_dir=test_data_dir,
    )


class TestSyntheticNestedRoundTrip:
    """
    Full ``run -> mint -> sync -> replay`` on the synthetic nested-output diagnostic.

    This exercises every layer of the native-baseline system on nested native outputs
    with absolute-path dict keys.
    """

    def test_run_mint_sync_replay(self, tmp_path: Path) -> None:
        """
        Replay of minted blobs matches the committed bundle within tolerance.

        Steps
        -----
        1. **Run** the synthetic diagnostic.
        2. **Capture + Mint**: ``capture_execution`` writes the committed bundle and
           snapshots the curated native files; ``store.put`` PUTs each native blob;
           a ``Manifest`` is authored carrying the native block.
        3. **Sync**: every blob in the manifest is reachable from the store.
        4. **Replay**: ``materialise_native`` into a fresh directory, expand
           placeholders, rebuild the result, and tolerantly compare each committed
           artefact.
        """
        store = LocalFilesystemStore(root=tmp_path / "store")
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()
        test_data_dir = tmp_path / "test-data"
        test_data_dir.mkdir()
        results_base = tmp_path / "results-base"

        # Step 1: run
        result = _run_synthetic(tmp_path)
        assert result.successful, "Synthetic diagnostic must succeed"
        definition = result.definition
        output_dir = definition.output_directory

        # Step 2: capture + mint
        committed_digests, native = _capture_synthetic(
            result,
            results_base=results_base,
            regression_dir=regression_dir,
            test_data_dir=test_data_dir,
        )
        assert committed_digests, "capture must produce committed digests"
        assert native, "capture must snapshot at least one native file"
        # The nested data file must be part of the curated native set.
        assert _NATIVE_DATA_RELPATH in native, (
            f"Curated native set must include the nested data file; got {sorted(native)}"
        )

        # PUT each native blob from the persisted results base.
        fragment = definition.output_fragment()
        base_dir = results_base / fragment
        for relpath, entry in native.items():
            digest = store.put(base_dir / relpath)
            assert digest == entry.sha256, "store digest must equal the captured digest"

        manifest = Manifest(
            schema=1,
            test_case_version=1,
            committed=committed_digests,
            native=native,
        )
        manifest_path = tmp_path / "manifest.json"
        manifest.dump(manifest_path)

        # Manifest round-trips byte-stably.
        reloaded = Manifest.load(manifest_path)
        assert reloaded.committed == manifest.committed
        assert set(reloaded.native) == set(manifest.native)

        # Step 3: sync — every referenced blob must be in the store.
        for relpath, entry in reloaded.native.items():
            assert store.has(entry.sha256), (
                f"Sync failed: blob {entry.sha256} not in store (relpath={relpath!r})"
            )

        # Step 4: replay
        replay_dir = tmp_path / "replay_output"
        replay_dir.mkdir()
        materialise_native(reloaded.native, store, replay_dir)
        from_placeholders(replay_dir, output_dir=replay_dir, test_data_dir=test_data_dir)

        replay_definition = ExecutionDefinition(
            diagnostic=definition.diagnostic,
            key="test-default",
            datasets=definition.datasets,
            output_directory=replay_dir,
            root_directory=tmp_path,
        )
        replay_result = definition.diagnostic.build_execution_result(replay_definition)

        replacements = {
            str(replay_dir): "<OUTPUT_DIR>",
            str(test_data_dir): "<TEST_DATA_DIR>",
        }
        for filename in ("series.json", "diagnostic.json", "output.json"):
            assert_bundle_regression(
                regression_dir / filename,
                replay_result.to_output_path(filename),
                slug=definition.diagnostic.slug,
                tol=Tolerance(),
                replacements=replacements,
            )

        # Sanity: the committed output.json must contain the placeholder where the
        # absolute output_dir was used (including as a dict key — the M2 path).
        committed_output = (regression_dir / "output.json").read_text(encoding="utf-8")
        assert str(output_dir) not in committed_output, (
            "Absolute output_dir must be replaced by <OUTPUT_DIR> in committed output.json"
        )
        assert "<OUTPUT_DIR>" in committed_output

    def test_absolute_path_keys_sanitised(self, tmp_path: Path) -> None:
        """
        The committed ``output.json`` carries ``<OUTPUT_DIR>`` in a dict key.

        Validates the M2 key-rewriting path: the synthetic bundle uses an absolute
        path as an ``html`` object key, which must be sanitised to a placeholder in
        the committed bundle.
        """
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()
        test_data_dir = tmp_path / "test-data"
        test_data_dir.mkdir()
        results_base = tmp_path / "results-base"

        result = _run_synthetic(tmp_path)
        _capture_synthetic(
            result,
            results_base=results_base,
            regression_dir=regression_dir,
            test_data_dir=test_data_dir,
        )

        committed_obj = json.loads((regression_dir / "output.json").read_text(encoding="utf-8"))

        def _has_placeholder_key(obj: Any) -> bool:
            if isinstance(obj, dict):
                return any("<OUTPUT_DIR>" in k or _has_placeholder_key(v) for k, v in obj.items())
            if isinstance(obj, list):
                return any(_has_placeholder_key(item) for item in obj)
            if isinstance(obj, str):
                return "<OUTPUT_DIR>" in obj
            return False

        assert _has_placeholder_key(committed_obj), (
            "Committed output.json must contain at least one <OUTPUT_DIR> placeholder "
            "(in a dict key or value, exercising M2 key-sanitisation)"
        )

    def test_replay_fails_without_nested_native_file(self, tmp_path: Path) -> None:
        """
        Replay genuinely fails if the nested native file is missing.

        Proves the curated native set is load-bearing: ``build_execution_result``
        reads back the nested data file, so a replay directory missing that file
        cannot rebuild the result.
        """
        replay_dir = tmp_path / "replay_output"
        replay_dir.mkdir()

        result = _run_synthetic(tmp_path)
        definition = result.definition

        # Build a replay definition over an EMPTY directory (no native file).
        replay_definition = ExecutionDefinition(
            diagnostic=definition.diagnostic,
            key="test-default",
            datasets=definition.datasets,
            output_directory=replay_dir,
            root_directory=tmp_path,
        )

        with pytest.raises((FileNotFoundError, OSError)):
            definition.diagnostic.build_execution_result(replay_definition)


@pytest.mark.slow
class TestExampleSmokeRoundTrip:
    """
    Smoke round-trip on the real ``example`` provider's ``GlobalMeanTimeseries``.

    Requires offline sample data; skipped clearly when unavailable.
    """

    def test_example_roundtrip(self, tmp_path: Path) -> None:  # noqa: PLR0915
        """
        Run -> mint -> sync -> replay the example diagnostic, or skip clearly.

        The skip message names the reason so CI / developers know what data is
        needed.
        """
        try:
            from climate_ref_example import provider as example_provider  # noqa: PLC0415
            from climate_ref_example.example import GlobalMeanTimeseries  # noqa: PLC0415
        except ImportError:
            pytest.skip("climate-ref-example is not installed — example smoke round-trip skipped")

        if TEST_DATA_DIR is None:
            pytest.skip(
                "TEST_DATA_DIR is not configured — example smoke round-trip requires sample data; skipped"
            )

        sample_data_dir = TEST_DATA_DIR / "sample-data"
        if not sample_data_dir.exists():
            pytest.skip(
                f"Sample data directory does not exist at {sample_data_dir} — "
                "run `make fetch-test-data`; example smoke round-trip skipped"
            )

        diag = GlobalMeanTimeseries()
        diag.provider = example_provider

        paths = TestCasePaths.from_diagnostic(diag, "default")
        if paths is None or not paths.catalog.exists():
            pytest.skip(
                "Example diagnostic catalog not found — "
                "run `ref test-cases fetch --provider example` first; skipped"
            )

        from climate_ref_core.testing import load_datasets_from_yaml  # noqa: PLC0415

        datasets = load_datasets_from_yaml(paths.catalog)
        for src_type, collection in datasets.items():
            if len(collection.datasets) > 0 and "path" not in collection.datasets.columns:
                pytest.skip(
                    f"Example catalog for {src_type!r} has no 'path' column — "
                    "run `ref test-cases fetch --provider example`; skipped"
                )

        config = Config.default()
        config.paths.results = tmp_path / "results"

        runner = TestCaseRunner(config=config, datasets=datasets)
        try:
            result = runner.run(diag, "default", tmp_path / "example_output", clean=True)
        except FileNotFoundError as exc:
            pytest.skip(f"Example sample data files unavailable ({exc}); skipped")

        assert result.successful, "Example GlobalMeanTimeseries must succeed"

        definition = result.definition
        test_data_dir = paths.test_data_dir
        results_base = tmp_path / "results-base"
        regression_dir = tmp_path / "example_regression"
        regression_dir.mkdir()
        store = LocalFilesystemStore(root=tmp_path / "example_store")

        # Mint
        committed_digests, native = _capture_synthetic(
            result,
            results_base=results_base,
            regression_dir=regression_dir,
            test_data_dir=test_data_dir,
        )
        assert native, "example capture must include the registered NetCDF native output"

        fragment = definition.output_fragment()
        base_dir = results_base / fragment
        for relpath, entry in native.items():
            digest = store.put(base_dir / relpath)
            assert digest == entry.sha256

        manifest = Manifest(
            schema=1,
            test_case_version=1,
            committed=committed_digests,
            native=native,
        )

        # Sync
        for relpath, entry in manifest.native.items():
            assert store.has(entry.sha256), (
                f"Example sync failed: blob {entry.sha256} not in store (relpath={relpath!r})"
            )

        # Replay
        replay_dir = tmp_path / "example_replay"
        replay_dir.mkdir()
        materialise_native(manifest.native, store, replay_dir)
        from_placeholders(replay_dir, output_dir=replay_dir, test_data_dir=test_data_dir)

        replay_definition = ExecutionDefinition(
            diagnostic=diag,
            key="test-default",
            datasets=datasets,
            output_directory=replay_dir,
            root_directory=tmp_path,
        )
        replay_result = diag.build_execution_result(replay_definition)

        replacements = {
            str(replay_dir): "<OUTPUT_DIR>",
            str(test_data_dir): "<TEST_DATA_DIR>",
        }
        for filename in ("series.json", "diagnostic.json", "output.json"):
            assert_bundle_regression(
                regression_dir / filename,
                replay_result.to_output_path(filename),
                slug=diag.slug,
                tol=Tolerance(),
                replacements=replacements,
            )
        # The example diagnostic registers a NetCDF native output.
        assert any(relpath.endswith(".nc") for relpath in manifest.native), (
            f"Example native set should include the NetCDF output; got {sorted(manifest.native)}"
        )


class TestNegativeFloatDrift:
    """Perturbing a committed float beyond tolerance produces a path-precise failure."""

    def test_float_drift_raises_assertion_error(self, tmp_path: Path) -> None:
        """
        A committed float perturbed by 10 percent makes replay raise ``AssertionError``.

        The message must name the diagnostic slug, the JSON location, and the
        remediation hint.
        """
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()
        test_data_dir = tmp_path / "test-data"
        test_data_dir.mkdir()
        results_base = tmp_path / "results-base"

        result = _run_synthetic(tmp_path)
        _capture_synthetic(
            result,
            results_base=results_base,
            regression_dir=regression_dir,
            test_data_dir=test_data_dir,
        )

        committed_path = regression_dir / "diagnostic.json"
        committed_obj = json.loads(committed_path.read_text(encoding="utf-8"))
        committed_obj["RESULTS"]["global"]["synth"]["score"] *= 1.1
        committed_path.write_text(json.dumps(committed_obj, indent=2) + "\n", encoding="utf-8")

        actual_path = result.to_output_path("diagnostic.json")

        with pytest.raises(AssertionError) as exc_info:
            assert_bundle_regression(
                committed_path,
                actual_path,
                slug="synthetic-nested",
                tol=Tolerance(),
                replacements={},
            )

        error_msg = str(exc_info.value)
        assert "synthetic-nested" in error_msg
        assert "score" in error_msg or "RESULTS" in error_msg or "global" in error_msg
        assert "force-regen" in error_msg or "regen" in error_msg.lower()

    def test_float_within_tolerance_passes(self, tmp_path: Path) -> None:
        """A perturbation within ``Tolerance()`` rtol must not raise."""
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()
        test_data_dir = tmp_path / "test-data"
        test_data_dir.mkdir()
        results_base = tmp_path / "results-base"

        result = _run_synthetic(tmp_path)
        _capture_synthetic(
            result,
            results_base=results_base,
            regression_dir=regression_dir,
            test_data_dir=test_data_dir,
        )

        committed_path = regression_dir / "diagnostic.json"
        committed_obj = json.loads(committed_path.read_text(encoding="utf-8"))
        original = committed_obj["RESULTS"]["global"]["synth"]["score"]
        committed_obj["RESULTS"]["global"]["synth"]["score"] = original * (1 + 1e-9)
        committed_path.write_text(json.dumps(committed_obj, indent=2) + "\n", encoding="utf-8")

        actual_path = result.to_output_path("diagnostic.json")

        assert_bundle_regression(
            committed_path,
            actual_path,
            slug="synthetic-nested",
            tol=Tolerance(),
            replacements={},
        )


class TestNegativeMissingBlob:
    """Deleting a referenced blob causes a loud, actionable failure."""

    def _mint(self, tmp_path: Path) -> tuple[LocalFilesystemStore, Manifest, dict[str, NativeEntry]]:
        """Run + capture + PUT, returning the store, manifest, and native snapshot."""
        store = LocalFilesystemStore(root=tmp_path / "store")
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()
        test_data_dir = tmp_path / "test-data"
        test_data_dir.mkdir()
        results_base = tmp_path / "results-base"

        result = _run_synthetic(tmp_path)
        committed_digests, native = _capture_synthetic(
            result,
            results_base=results_base,
            regression_dir=regression_dir,
            test_data_dir=test_data_dir,
        )
        fragment = result.definition.output_fragment()
        base_dir = results_base / fragment
        for relpath, entry in native.items():
            store.put(base_dir / relpath)

        manifest = Manifest(
            schema=1,
            test_case_version=1,
            committed=committed_digests,
            native=native,
        )
        return store, manifest, native

    def test_missing_blob_detected_on_sync(self, tmp_path: Path) -> None:
        """
        A blob deleted after minting is caught when verifying ``store.has``.

        The failure report must name the digest and relpath.
        """
        store, manifest, native = self._mint(tmp_path)

        some_relpath, some_entry = next(iter(native.items()))
        store._blob_path(some_entry.sha256).unlink()

        missing: list[tuple[str, str]] = [
            (relpath, entry.sha256)
            for relpath, entry in manifest.native.items()
            if not store.has(entry.sha256)
        ]

        assert missing, "Sync must detect the missing blob"
        assert some_relpath in [m[0] for m in missing]
        assert some_entry.sha256 in [m[1] for m in missing]

    def test_missing_blob_detected_on_replay(self, tmp_path: Path) -> None:
        """
        Materialising with a missing blob raises ``FileNotFoundError`` naming the digest.
        """
        store, manifest, native = self._mint(tmp_path)

        _some_relpath, some_entry = next(iter(native.items()))
        store._blob_path(some_entry.sha256).unlink()

        replay_dir = tmp_path / "replay_output"
        replay_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            materialise_native(manifest.native, store, replay_dir)

        assert some_entry.sha256 in str(exc_info.value), (
            f"FileNotFoundError must name the missing digest {some_entry.sha256!r}"
        )


def _synthetic_cli_case(tmp_path: Path) -> tuple[MagicMock, Diagnostic, MagicMock]:
    """
    Build a ``(registry, diagnostic, test_case)`` trio backed by the synthetic diagnostic.

    The registry is a mock whose single provider exposes the real synthetic
    diagnostic, so the CLI verbs run the genuine capture path.
    """
    provider = _build_synthetic_provider()
    diag = provider.diagnostics()[0]

    tc = MagicMock()
    tc.name = "default"
    tc.description = "synthetic"

    mock_provider = MagicMock(slug="synthetic-test")
    mock_provider.diagnostics.return_value = [diag]
    registry = MagicMock(providers=[mock_provider])
    return registry, diag, tc


def _local_store_config(tmp_path: Path) -> Config:
    """
    Build a :class:`Config` whose ``native_store`` points at a local writable path.

    Uses a ``file://`` URL so :func:`~climate_ref_core.regression.build_native_store`
    returns a :class:`LocalFilesystemStore` for both the writable mint path and the
    read-only sync/replay path.
    """
    config = Config.default()
    config.paths.results = tmp_path / "results"
    store_root = tmp_path / "native-store"
    store_root.mkdir(parents=True, exist_ok=True)
    config.native_store.url = store_root.as_uri()
    config.native_store.cache_dir = tmp_path / "native-cache"
    return config


class TestNativeStoreConfig:
    """The local-store config maps to a LocalFilesystemStore for both directions."""

    def test_file_url_builds_local_store(self, tmp_path: Path) -> None:
        """A ``file://`` native-store URL yields a writable and readable local store."""
        from climate_ref_core.regression import build_native_store  # noqa: PLC0415

        config = _local_store_config(tmp_path)
        writable = build_native_store(config.native_store, writable=True)
        readable = build_native_store(config.native_store, writable=False)
        assert isinstance(writable, LocalFilesystemStore)
        assert isinstance(readable, LocalFilesystemStore)

        blob = tmp_path / "blob.bin"
        blob.write_bytes(b"abc")
        digest = writable.put(blob)
        assert readable.has(digest)


class TestCliMintSyncReplay:
    """Drive ``ref test-cases mint`` / ``sync`` / ``replay`` against a local store."""

    def test_mint_sync_replay_succeeds(self, invoke_cli: Any, mocker: Any, tmp_path: Path) -> None:
        """
        The full CLI lifecycle exits 0 on the synthetic case against a local store.

        ``mint`` PUTs blobs and authors the manifest; ``sync`` confirms every blob
        is reachable; ``replay`` materialises and tolerantly compares.
        """
        registry, diag, _tc = _synthetic_cli_case(tmp_path)
        config = _local_store_config(tmp_path)

        case_dir = tmp_path / "td" / diag.slug / "default"
        case_dir.mkdir(parents=True)
        (case_dir / "catalog.yaml").touch()
        paths = TestCasePaths(root=case_dir)

        datasets = _build_synthetic_datasets()

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=datasets)
        mocker.patch("climate_ref_core.testing.get_catalog_hash", return_value=None)

        # The CLI builds its store from the app-context config; rather than rely on
        # env plumbing, patch build_native_store to use our local-store config.
        from climate_ref_core.regression.store import build_native_store as real_build  # noqa: PLC0415

        def _build(_cfg: Any, *, writable: bool) -> Any:
            return real_build(config.native_store, writable=writable)

        mocker.patch("climate_ref_core.regression.store.build_native_store", side_effect=_build)

        # Mint
        invoke_cli(["test-cases", "mint", "--provider", "synthetic-test"], expected_exit_code=0)
        manifest = Manifest.load(paths.manifest)
        assert manifest.native, "mint must author a non-empty native block"
        assert _NATIVE_DATA_RELPATH in manifest.native

        # Sync
        invoke_cli(["test-cases", "sync", "--provider", "synthetic-test"], expected_exit_code=0)

        # Replay
        invoke_cli(
            ["test-cases", "replay", "--provider", "synthetic-test", "--diagnostic", diag.slug],
            expected_exit_code=0,
        )

    def test_replay_fails_on_corrupted_committed_bundle(
        self, invoke_cli: Any, mocker: Any, tmp_path: Path
    ) -> None:
        """
        Corrupting the committed bundle after minting makes ``replay`` exit non-zero.

        The tolerant comparison cannot reconcile the corrupted committed baseline against the
        regenerated bundle (here it no longer parses as JSON), so replay reports drift.
        """
        registry, diag, _tc = _synthetic_cli_case(tmp_path)
        config = _local_store_config(tmp_path)

        case_dir = tmp_path / "td" / diag.slug / "default"
        case_dir.mkdir(parents=True)
        (case_dir / "catalog.yaml").touch()
        paths = TestCasePaths(root=case_dir)

        datasets = _build_synthetic_datasets()

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=datasets)
        mocker.patch("climate_ref_core.testing.get_catalog_hash", return_value=None)

        from climate_ref_core.regression.store import build_native_store as real_build  # noqa: PLC0415

        def _build(_cfg: Any, *, writable: bool) -> Any:
            return real_build(config.native_store, writable=writable)

        mocker.patch("climate_ref_core.regression.store.build_native_store", side_effect=_build)

        invoke_cli(["test-cases", "mint", "--provider", "synthetic-test"], expected_exit_code=0)

        # Corrupt the committed bundle so the integrity check fails on replay.
        committed = paths.regression / "diagnostic.json"
        committed.write_text(committed.read_text(encoding="utf-8") + "\n// drift\n", encoding="utf-8")

        invoke_cli(
            ["test-cases", "replay", "--provider", "synthetic-test", "--diagnostic", diag.slug],
            expected_exit_code=1,
        )

    def test_mint_refuses_without_writable_store(self, invoke_cli: Any, mocker: Any, tmp_path: Path) -> None:
        """``mint`` exits non-zero when no writable store can be built."""
        registry, _diag, _tc = _synthetic_cli_case(tmp_path)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch(
            "climate_ref_core.regression.store.build_native_store",
            side_effect=NotImplementedError("R2 backend deferred"),
        )

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "synthetic-test"],
            expected_exit_code=1,
        )
        assert "Cannot mint" in result.stderr
