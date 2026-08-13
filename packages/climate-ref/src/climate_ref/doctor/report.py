"""
What examining a deployment produced.

Running the checks, ordering what they found, counting them, and describing the deployment they
ran against are one job, so `diagnose` does all of it and hands back one value. A caller that
wants to display a report does not have to know how any of that is assembled.
"""

from attrs import frozen

from climate_ref.doctor.context import DoctorContext
from climate_ref.doctor.environment import collect_environment
from climate_ref.doctor.findings import Finding, Severity, worst_severity
from climate_ref.doctor.registry import iter_checks, run_checks


@frozen
class DoctorReport:
    """
    What the checks found, and the deployment they ran against.
    """

    findings: tuple[Finding, ...]
    """Everything the checks found, worst first, then by the check that produced them."""

    check_count: int
    """How many checks ran, including those that found nothing."""

    environment: dict[str, dict[str, str]] | None = None
    """
    A description of the deployment as sections of ``name: value`` pairs, or ``None``.

    This is deliberately an untyped blob of data.
    It is used as context for a bug report and the shape may change at any time.

    Sensitive values have been redacted.
    """

    @property
    def worst_severity(self) -> Severity | None:
        """The most serious severity found, or ``None`` when nothing was found."""
        return worst_severity(self.findings)


def diagnose(context: DoctorContext, *, environment: bool = False) -> DoctorReport:
    """
    Examine a REF deployment and generate a report.

    Every check runs.
    One that raises becomes a finding rather than stopping the rest,
    so a broken check cannot make the deployment look healthy.

    Parameters
    ----------
    context
        The deployment to examine.
    environment
        Whether to describe the deployment as well as check it.

    Returns
    -------
    :
        What the checks found, and the deployment they ran against.
    """
    return DoctorReport(
        findings=tuple(run_checks(context)),
        check_count=len(iter_checks()),
        environment=collect_environment(context) if environment else None,
    )
