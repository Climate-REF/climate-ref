"""
What a check reports, and how serious it is.
"""

from collections.abc import Sequence

from attrs import frozen


class Severity:
    """How much a finding matters. Ordered worst-first for reporting."""

    ERROR = "error"
    """Results computed in this state are wrong."""

    WARNING = "warning"
    """Something the deployment probably did not intend, but results remain valid."""

    INFO = "info"
    """Worth knowing; no action required."""


SEVERITY_ORDER = (Severity.ERROR, Severity.WARNING, Severity.INFO)


@frozen
class Finding:
    """
    One problem found by a check.
    """

    severity: str
    """One of `Severity`."""

    summary: str
    """One line stating what is wrong."""

    detail: str = ""
    """Optional further explanation, including what to do about it."""

    check: str = ""
    """
    Slug of the check that produced it, e.g. ``duplicate-coverage``.

    A check does not set this itself: the runner stamps it from the check's registration, so
    the slug has one definition.
    """


def worst_severity(findings: Sequence[Finding]) -> str | None:
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
