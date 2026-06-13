"""
Content comparison utilities for regression testing.
"""

import json
import math
from pathlib import Path
from typing import Any

from attrs import frozen

from climate_ref_core.output_files import ordered_replacements


@frozen
class Tolerance:
    """
    Float comparison tolerance for bundle regression checks.

    Parameters
    ----------
    rtol
        Relative tolerance — the allowed proportional difference between expected
        and actual float values (default ``1e-6``).
    atol
        Absolute tolerance — the minimum absolute difference that can ever be
        flagged, regardless of magnitude (default ``0.0``).
    """

    rtol: float = 1e-6
    atol: float = 0.0


def _rewrite_keys_and_values(obj: Any, replacements: list[tuple[str, str]]) -> Any:
    """
    Recursively rewrite both dict *keys* and leaf string *values* in ``obj``.

    Replacements are applied in the order supplied
    (caller is responsible for longest-key-first ordering).

    Parameters
    ----------
    obj
        The parsed JSON object to rewrite (dict, list, or scalar).
    replacements
        Ordered list of ``(real, placeholder)`` pairs to apply.

    Returns
    -------
    :
        A new object with all matching substrings replaced.
    """
    if isinstance(obj, dict):
        return {
            _apply_replacements(k, replacements): _rewrite_keys_and_values(v, replacements)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_rewrite_keys_and_values(item, replacements) for item in obj]
    if isinstance(obj, str):
        return _apply_replacements(obj, replacements)
    return obj


def _apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    """
    Apply a sequence of ``(real, placeholder)`` string substitutions to ``text``.

    Parameters
    ----------
    text
        The input string.
    replacements
        Ordered list of ``(real, placeholder)`` pairs.

    Returns
    -------
    :
        The rewritten string.
    """
    for real, placeholder in replacements:
        text = text.replace(real, placeholder)
    return text


def compare_json_content(
    expected: Any,
    actual: Any,
    *,
    tol: Tolerance,
    path: str = "",
) -> list[str]:
    """
    Recursively compare two parsed JSON values with float tolerance.

    Rules:
    - **Floats**: compared with relative tolerance ``tol.rtol``
      and absolute tolerance ``tol.atol``.
    - **Ints, strings, bools, ``None``**: exact equality.
    - **Lists**: element-by-element, same length required.
    - **Dicts**: key sets must match; values compared recursively.

    Parameters
    ----------
    expected
        The reference (committed) parsed JSON value.
    actual
        The regenerated parsed JSON value.
    tol
        Float comparison tolerance.
    path
        Dot-/bracket-notation path prefix for error messages (empty at top level).

    Returns
    -------
    :
        A list of human-readable mismatch descriptions.
        An empty list means the values are equivalent within tolerance.
    """
    mismatches: list[str] = []
    _compare_recursive(expected, actual, tol=tol, path=path, out=mismatches)
    return mismatches


def _compare_recursive(
    expected: Any,
    actual: Any,
    *,
    tol: Tolerance,
    path: str,
    out: list[str],
) -> None:
    """Recursive helper; appends to ``out``.

    ``path`` is empty at the top level; ``label`` substitutes ``<root>`` in messages
    so top-level keys render bare (``key``) while a root-level scalar/type mismatch
    is still identifiable.
    """
    label = path or "<root>"
    # Type mismatch (treat int/float as numeric together)
    if type(expected) is not type(actual):
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            _compare_numeric(float(expected), float(actual), tol=tol, path=label, out=out)
            return
        out.append(
            f"{label}: type mismatch — expected {type(expected).__name__} ({expected!r}), "
            f"got {type(actual).__name__} ({actual!r})"
        )
        return

    if isinstance(expected, float):
        _compare_numeric(expected, actual, tol=tol, path=label, out=out)

    elif isinstance(expected, dict):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing:
            out.append(f"{label}: missing keys {sorted(missing)!r}")
        if extra:
            out.append(f"{label}: unexpected extra keys {sorted(extra)!r}")
        # Sort shared keys for deterministic mismatch ordering across runs.
        for key in sorted(expected_keys & actual_keys):
            child_path = f"{path}.{key}" if path else str(key)
            _compare_recursive(expected[key], actual[key], tol=tol, path=child_path, out=out)

    elif isinstance(expected, list):
        if len(expected) != len(actual):
            out.append(f"{label}: length mismatch — expected {len(expected)}, got {len(actual)}")
        # Compare the common prefix so partial mismatches are visible.
        for i, (e, a) in enumerate(zip(expected, actual)):
            _compare_recursive(e, a, tol=tol, path=f"{path}[{i}]", out=out)

    # int, bool, str, None — exact equality
    elif expected != actual:
        out.append(f"{label}: expected {expected!r}, got {actual!r}")


def _compare_numeric(
    expected: float,
    actual: float,
    *,
    tol: Tolerance,
    path: str,
    out: list[str],
) -> None:
    """Compare two numeric values with tolerance; append to ``out`` on failure."""
    # Both NaN is considered equal
    if math.isnan(expected) and math.isnan(actual):
        return
    if not math.isclose(expected, actual, rel_tol=tol.rtol, abs_tol=tol.atol):
        out.append(
            f"{path}: float mismatch — expected {expected!r}, got {actual!r} "
            f"(rtol={tol.rtol}, atol={tol.atol})"
        )


def assert_bundle_regression(
    expected_path: Path,
    actual_path: Path,
    *,
    slug: str,
    tol: Tolerance = Tolerance(),
    replacements: dict[str, str],
) -> None:
    """
    Assert that a regenerated committed-bundle JSON file matches the committed copy.

    Algorithm:

    1. **Byte-equal fast path** — if the raw bytes of both files are identical, return immediately.
    2. Parse both files as JSON.
    3. Rewrite both dict *keys* and leaf string *values* in the regenerated
       (``actual``) document using ``replacements``, applied **longest-key-first**
       so that a shorter placeholder cannot pre-empt a longer overlapping one.
    4. Call :func:`compare_json_content` with the given tolerance.
    5. Raise ``AssertionError`` with the full mismatch list and a remediation hint
       if any mismatches are found.

    The replacements map follows the convention used throughout the testing
    infrastructure: keys are real runtime values (absolute paths, recipe-dir
    timestamps), values are the committed-bundle placeholders (e.g.
    ``"<OUTPUT_DIR>"``, ``"<TEST_DATA_DIR>"``, ``"<RECIPE_RUN>"``).
    Only the *actual* document is rewritten: the committed *expected* file is
    assumed to already contain placeholders, which
    :func:`~climate_ref_core.regression.capture.write_committed_bundle` guarantees
    at capture time. A hand-edited baseline with raw paths will surface as ordinary
    value mismatches.

    Both ``<OUTPUT_DIR>`` and ``<RECIPE_RUN>`` participate in dict-KEY rewriting
    because ESMValTool's ``output.json`` embeds absolute paths as object keys.

    Parameters
    ----------
    expected_path
        Path to the committed (on-disk) bundle file containing placeholders.
    actual_path
        Path to the regenerated bundle file containing real runtime paths.
    slug
        Diagnostic slug used in error messages.
    tol
        Float comparison tolerance.
    replacements
        Mapping of ``{real_value: placeholder}`` applied to the actual document.

    Raises
    ------
    AssertionError
        If the bundles differ beyond tolerance after sanitisation.

    Notes
    -----
    If ``expected_path`` does not exist (a legacy regression without a committed bundle),
    the comparison is skipped silently and the function returns.
    """
    if not expected_path.exists():
        # Legacy regression data without this bundle file — skip silently
        return

    expected_bytes = expected_path.read_bytes()
    actual_bytes = actual_path.read_bytes()

    # Fast path: byte-identical means no difference.
    if expected_bytes == actual_bytes:
        return

    expected_obj = json.loads(expected_bytes.decode("utf-8"))
    actual_obj = json.loads(actual_bytes.decode("utf-8"))

    # Rewrite actual — both keys and leaf values — before comparison.
    actual_sanitised = _rewrite_keys_and_values(actual_obj, ordered_replacements(replacements))

    mismatches = compare_json_content(expected_obj, actual_sanitised, tol=tol)
    if mismatches:
        mismatch_detail = "\n  ".join(mismatches)
        msg = (
            f"Diagnostic {slug!r}: committed bundle {expected_path.name!r} "
            f"differs from regenerated output after sanitisation.\n"
            f"Mismatches ({len(mismatches)}):\n  {mismatch_detail}\n\n"
            f"Remediation: if the change is intentional, bump the diagnostic's "
            f"test_case_version and regenerate with `ref test-cases run --force-regen`."
        )
        raise AssertionError(msg)
