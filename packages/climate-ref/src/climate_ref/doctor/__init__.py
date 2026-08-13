"""
Health checks for a Climate-REF deployment.

These look for the conditions that make a solve quietly do the wrong thing rather than fail.
These include reference data that no diagnostic can reach,
reference data that is missing so its diagnostics never run,
and datasets whose files cover the same period twice.

`diagnose` examines a deployment and returns a `DoctorReport`,
which is everything a caller needs to display one.

`check` declares a new check, a function taking a `DoctorContext` and returning `Finding`s.
"""

from climate_ref.doctor import checks  # noqa: F401  Importing the built-in checks registers them
from climate_ref.doctor.context import DoctorContext
from climate_ref.doctor.findings import Finding, Severity, worst_severity
from climate_ref.doctor.registry import check, iter_checks
from climate_ref.doctor.report import DoctorReport, diagnose

__all__ = [
    "DoctorContext",
    "DoctorReport",
    "Finding",
    "Severity",
    "check",
    "diagnose",
    "iter_checks",
    "worst_severity",
]
