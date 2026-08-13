"""
What a check reports, and how serious it is.
"""

from collections.abc import Sequence
from enum import StrEnum

from attrs import frozen


class Severity(StrEnum):
    """How much a finding matters. Declared worst-first for reporting."""

    ERROR = "error"
    """Results computed in this state are wrong."""

    WARNING = "warning"
    """Something the deployment probably did not intend, but results remain valid."""

    INFO = "info"
    """Worth knowing, no action required."""


SEVERITY_ORDER = tuple(Severity)


@frozen
class Finding:
    """
    One problem found by a check.
    """

    severity: Severity
    """How much it matters."""

    summary: str
    """One line stating what is wrong."""

    detail: str = ""
    """Optional further explanation of this finding alone."""

    remedy: str = ""
    """
    Optional instruction for fixing it.

    Findings that share a remedy are reported under it once rather than repeating it,
    so keep the wording free of anything specific to one finding.
    """

    command: str = ""
    """
    Optional command that carries out the remedy.

    Held apart from ``remedy`` so it can be printed unwrapped and stay pasteable.
    """

    check: str = ""
    """
    Slug of the check that produced it, e.g. ``duplicate-coverage``.

    A check does not set this itself.
    The runner stamps it from the check's registration, so the slug has one definition.
    """


def worst_severity(findings: Sequence[Finding]) -> Severity | None:
    """
    Return the most serious severity present, or ``None`` when there are no findings.

    Parameters
    ----------
    findings
        The findings to inspect.

    Returns
    -------
    :
        The worst severity present, or ``None`` when ``findings`` is empty.
    """
    for severity in SEVERITY_ORDER:
        if any(finding.severity == severity for finding in findings):
            return severity
    return None
