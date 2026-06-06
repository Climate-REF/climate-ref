"""
Unit tests for :mod:`climate_ref_core.regression.compare`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from climate_ref_core.regression.compare import (
    DEFAULT_TOLERANCE,
    Tolerance,
    assert_bundle_regression,
    compare_json_content,
    resolve_tolerance,
)


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


TOL = DEFAULT_TOLERANCE


class TestTolerance:
    def test_defaults(self) -> None:
        t = Tolerance()
        assert t.rtol == 1e-6
        assert t.atol == 0.0

    def test_custom_values(self) -> None:
        t = Tolerance(rtol=0.01, atol=1e-9)
        assert t.rtol == 0.01
        assert t.atol == 1e-9

    def test_default_tolerance_singleton(self) -> None:
        assert DEFAULT_TOLERANCE.rtol == 1e-6
        assert DEFAULT_TOLERANCE.atol == 0.0


class TestResolveTolerance:
    def test_returns_default(self) -> None:
        t = resolve_tolerance("some-diagnostic")
        assert t == DEFAULT_TOLERANCE

    def test_returns_default_with_test_case(self) -> None:
        t = resolve_tolerance("some-diagnostic", test_case="default")
        assert t == DEFAULT_TOLERANCE


# ---------------------------------------------------------------------------
# compare_json_content — scalars
# ---------------------------------------------------------------------------


class TestCompareJsonScalars:
    def test_equal_floats_pass(self) -> None:
        assert compare_json_content(1.0, 1.0, tol=TOL) == []

    def test_float_within_rtol_pass(self) -> None:
        tol = Tolerance(rtol=0.01)
        assert compare_json_content(1.0, 1.005, tol=tol) == []

    def test_float_beyond_rtol_fail(self) -> None:
        tol = Tolerance(rtol=0.01)
        result = compare_json_content(1.0, 1.02, tol=tol)
        assert len(result) == 1
        assert "float mismatch" in result[0]
        assert "rtol" in result[0]

    def test_float_within_atol_pass(self) -> None:
        tol = Tolerance(rtol=0.0, atol=0.1)
        assert compare_json_content(0.0, 0.05, tol=tol) == []

    def test_float_beyond_atol_fail(self) -> None:
        tol = Tolerance(rtol=0.0, atol=0.01)
        result = compare_json_content(0.0, 0.1, tol=tol)
        assert len(result) == 1
        assert "float mismatch" in result[0]

    def test_nan_vs_nan_pass(self) -> None:
        assert compare_json_content(math.nan, math.nan, tol=TOL) == []

    def test_equal_ints_pass(self) -> None:
        assert compare_json_content(42, 42, tol=TOL) == []

    def test_different_ints_fail(self) -> None:
        result = compare_json_content(1, 2, tol=TOL)
        assert len(result) == 1
        assert "1" in result[0]
        assert "2" in result[0]

    def test_equal_strings_pass(self) -> None:
        assert compare_json_content("hello", "hello", tol=TOL) == []

    def test_different_strings_fail(self) -> None:
        result = compare_json_content("hello", "world", tol=TOL)
        assert len(result) == 1
        assert "hello" in result[0]

    def test_equal_bools_pass(self) -> None:
        assert compare_json_content(True, True, tol=TOL) == []

    def test_different_bools_fail(self) -> None:
        result = compare_json_content(True, False, tol=TOL)
        assert len(result) == 1

    def test_none_equal_pass(self) -> None:
        assert compare_json_content(None, None, tol=TOL) == []

    def test_type_mismatch_fail(self) -> None:
        result = compare_json_content("1", 1, tol=TOL)
        assert len(result) == 1
        assert "type mismatch" in result[0]

    def test_int_float_numeric_comparison(self) -> None:
        """int vs float should be treated as numeric, not type-mismatch."""
        assert compare_json_content(1, 1.0, tol=TOL) == []
        result = compare_json_content(1, 2.0, tol=TOL)
        assert len(result) == 1
        assert "float mismatch" in result[0]


class TestCompareJsonLists:
    def test_equal_lists_pass(self) -> None:
        assert compare_json_content([1, 2, 3], [1, 2, 3], tol=TOL) == []

    def test_length_mismatch_fail(self) -> None:
        result = compare_json_content([1, 2], [1], tol=TOL)
        assert any("length mismatch" in m for m in result)

    def test_element_mismatch_path_precise(self) -> None:
        result = compare_json_content([1.0, 99.0], [1.0, 2.0], tol=TOL)
        assert len(result) == 1
        assert "[1]" in result[0]

    def test_nested_list_path(self) -> None:
        result = compare_json_content([[1.0]], [[2.0]], tol=TOL)
        assert "[0][0]" in result[0]


class TestCompareJsonDicts:
    def test_equal_dicts_pass(self) -> None:
        assert compare_json_content({"a": 1}, {"a": 1}, tol=TOL) == []

    def test_missing_key_reported(self) -> None:
        result = compare_json_content({"a": 1, "b": 2}, {"a": 1}, tol=TOL)
        assert any("missing keys" in m for m in result)
        assert any("b" in m for m in result)

    def test_extra_key_reported(self) -> None:
        result = compare_json_content({"a": 1}, {"a": 1, "extra": 2}, tol=TOL)
        assert any("extra keys" in m for m in result)
        assert any("extra" in m for m in result)

    def test_nested_value_mismatch_path_precise(self) -> None:
        result = compare_json_content(
            {"outer": {"inner": 1.0}},
            {"outer": {"inner": 99.0}},
            tol=TOL,
        )
        assert len(result) == 1
        assert "outer" in result[0]
        assert "inner" in result[0]

    def test_path_prefix_propagated(self) -> None:
        result = compare_json_content({"x": 1}, {"x": 2}, tol=TOL, path="parent")
        assert "parent" in result[0]


class TestAssertBundleRegressionBasic:
    def test_byte_equal_fast_path_passes(self, tmp_path: Path) -> None:
        """When both files are byte-identical, no comparison overhead occurs."""
        content = {"key": 1.0, "nested": {"val": "hello"}}
        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, content)
        _write_json(actual, content)
        # Must not raise
        assert_bundle_regression(expected, actual, slug="test-diag", tol=TOL, replacements={})

    def test_missing_expected_silently_skipped(self, tmp_path: Path) -> None:
        """If expected_path does not exist, the check is silently skipped."""
        actual = tmp_path / "actual.json"
        _write_json(actual, {"val": 1})
        # Must not raise even though expected does not exist
        assert_bundle_regression(
            tmp_path / "nonexistent.json",
            actual,
            slug="test-diag",
            tol=TOL,
            replacements={},
        )

    def test_value_drift_beyond_tolerance_raises(self, tmp_path: Path) -> None:
        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {"score": 1.0})
        _write_json(actual, {"score": 999.0})
        with pytest.raises(AssertionError, match="score"):
            assert_bundle_regression(expected, actual, slug="test-diag", tol=TOL, replacements={})

    def test_value_within_tolerance_passes(self, tmp_path: Path) -> None:
        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {"score": 1.0})
        # Within rtol=1e-6
        _write_json(actual, {"score": 1.0000005})
        assert_bundle_regression(expected, actual, slug="test-diag", tol=TOL, replacements={})

    def test_assertion_error_contains_slug(self, tmp_path: Path) -> None:
        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {"v": 1.0})
        _write_json(actual, {"v": 999.0})
        with pytest.raises(AssertionError, match="my-diag"):
            assert_bundle_regression(expected, actual, slug="my-diag", tol=TOL, replacements={})

    def test_assertion_error_contains_remediation_hint(self, tmp_path: Path) -> None:
        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {"v": 1.0})
        _write_json(actual, {"v": 999.0})
        with pytest.raises(AssertionError, match="force-regen"):
            assert_bundle_regression(expected, actual, slug="diag", tol=TOL, replacements={})


class TestAssertBundleRegressionKeySanitisation:
    """
    ESMValTool's output.json uses absolute filesystem paths as dict keys.
    TODO: Should we sanitise this globally?
    """

    def test_absolute_path_key_sanitised_passes(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path / "output")
        expected_key = "<OUTPUT_DIR>/result.nc"
        actual_key = f"{output_dir}/result.nc"

        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {expected_key: {"size": 100}})
        _write_json(actual, {actual_key: {"size": 100}})

        assert_bundle_regression(
            expected,
            actual,
            slug="diag",
            tol=TOL,
            replacements={output_dir: "<OUTPUT_DIR>"},
        )

    def test_value_drift_under_sanitised_key_fails_path_precisely(self, tmp_path: Path) -> None:
        """
        Edge case for drift which has been sanitised.
        """
        output_dir = str(tmp_path / "output")
        expected_key = "<OUTPUT_DIR>/result.nc"
        actual_key = f"{output_dir}/result.nc"

        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {expected_key: {"size": 100}})
        _write_json(actual, {actual_key: {"size": 999}})

        with pytest.raises(AssertionError) as exc_info:
            assert_bundle_regression(
                expected,
                actual,
                slug="diag",
                tol=TOL,
                replacements={output_dir: "<OUTPUT_DIR>"},
            )
        msg = str(exc_info.value)
        assert "size" in msg

    def test_test_data_dir_placeholder_in_keys(self, tmp_path: Path) -> None:
        """
        <TEST_DATA_DIR> must also be sanitised in dict keys.
        """
        test_data_dir = str(tmp_path / "test-data")
        expected_key = "<TEST_DATA_DIR>/catalog.yaml"
        actual_key = f"{test_data_dir}/catalog.yaml"

        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {expected_key: "present"})
        _write_json(actual, {actual_key: "present"})

        assert_bundle_regression(
            expected,
            actual,
            slug="diag",
            tol=TOL,
            replacements={test_data_dir: "<TEST_DATA_DIR>"},
        )

    def test_longest_key_first_prevents_short_placeholder_preempt(self, tmp_path: Path) -> None:
        # Simulate: test_data_dir is a prefix of output_dir
        test_data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "data" / "output")

        expected_key = "<OUTPUT_DIR>/file.nc"
        actual_key = f"{output_dir}/file.nc"

        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(expected, {expected_key: "ok"})
        _write_json(actual, {actual_key: "ok"})

        assert_bundle_regression(
            expected,
            actual,
            slug="diag",
            tol=TOL,
            replacements={
                output_dir: "<OUTPUT_DIR>",
                test_data_dir: "<TEST_DATA_DIR>",
            },
        )

    def test_both_key_and_value_rewritten(self, tmp_path: Path) -> None:
        """
        When the same real path appears in both a key AND a leaf value, both must be rewritten.
        """
        output_dir = str(tmp_path / "output")
        expected = tmp_path / "expected.json"
        actual = tmp_path / "actual.json"
        _write_json(
            expected,
            {"<OUTPUT_DIR>/f.nc": {"path": "<OUTPUT_DIR>/f.nc"}},
        )
        _write_json(
            actual,
            {f"{output_dir}/f.nc": {"path": f"{output_dir}/f.nc"}},
        )
        # Must pass — both key and value are rewritten
        assert_bundle_regression(
            expected,
            actual,
            slug="diag",
            tol=TOL,
            replacements={output_dir: "<OUTPUT_DIR>"},
        )
