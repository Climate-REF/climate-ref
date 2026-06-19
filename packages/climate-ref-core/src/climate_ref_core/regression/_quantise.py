"""
Float quantisation for committed regression bundles.

Committed regression JSON (``series.json`` / ``diagnostic.json`` / ``output.json``)
records full-precision floats whose least-significant digits are platform-dependent
(CPU, BLAS, library versions).
Those last digits churn byte-for-byte between CI and local runs even when the result
is numerically identical, producing noisy, unreviewable diffs in the committed bundle.

Rounding every float to a fixed number of significant figures at write time gives
stable, reviewable committed bytes.
We round to seven significant figures: one digit finer than the regression compare
tolerance (``rtol=1e-6`` in :mod:`climate_ref_core.regression.compare`),
so the rounding error stays an order of magnitude under tolerance
and can never flip a boundary gate verdict.

This affects only the committed bundle.
The native blobs (``.nc`` / ``.png``) and their content-addressed digests are never touched.
"""

from __future__ import annotations

from typing import Any

DEFAULT_SIG_FIGS: int = 7
"""Default significant figures for committed-bundle floats.

Deliberately one digit finer than the ``rtol=1e-6`` regression compare tolerance,
so rounding never flips a boundary gate verdict.
"""


def round_floats(obj: Any, sig_figs: int = DEFAULT_SIG_FIGS) -> Any:
    """
    Recursively round every ``float`` in a JSON-like structure to ``sig_figs`` figures.

    Walks dicts, lists and tuples, rounding each ``float`` via the ``g`` format
    (``float(f"{x:.{sig_figs}g}")``) and leaving ``int``, ``bool``, ``str`` and ``None``
    untouched.
    ``bool`` is a subclass of ``int`` (and not of ``float``),
    so booleans are never rounded.
    The operation is idempotent: rounding an already-rounded value is a no-op.

    Tuples are returned as lists, matching JSON serialisation semantics
    (JSON has no tuple type; the standard library serialises tuples as arrays).

    Parameters
    ----------
    obj
        A JSON-like object: a scalar, or an arbitrarily nested dict / list / tuple.
    sig_figs
        The number of significant figures to round each float to.

    Returns
    -------
    :
        A copy of ``obj`` with every float rounded to ``sig_figs`` significant figures.
    """
    # ``bool`` is a subclass of ``int``; check it explicitly so booleans are
    # preserved rather than being coerced through the float branch.
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return float(f"{obj:.{sig_figs}g}")
    if isinstance(obj, dict):
        return {key: round_floats(value, sig_figs) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(item, sig_figs) for item in obj]
    return obj
