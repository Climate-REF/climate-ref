"""
Health checks for a Climate-REF deployment.

These look for the conditions that make a solve quietly do the wrong thing rather than fail:
reference data that no diagnostic can reach, reference data that is missing so its diagnostics
never run, and datasets whose files cover the same period twice.

Each check is a function taking a `DoctorContext` and returning `Finding`s, declared with
`check`. Checks read the database and the configuration; none of them write.
"""

# Importing the built-in checks registers them, so this must happen when the package is
# imported rather than when a check is first needed.
from climate_ref.doctor import checks  # noqa: F401
from climate_ref.doctor.context import EMPTY_CATALOG, DoctorContext
from climate_ref.doctor.environment import EnvironmentReport, collect_environment
from climate_ref.doctor.findings import SEVERITY_ORDER, Finding, Severity, worst_severity
from climate_ref.doctor.registry import (
    CHECK_ENTRY_POINT_GROUP,
    RegisteredCheck,
    check,
    iter_checks,
    register_check,
    run_checks,
)

__all__ = [
    "CHECK_ENTRY_POINT_GROUP",
    "EMPTY_CATALOG",
    "SEVERITY_ORDER",
    "DoctorContext",
    "EnvironmentReport",
    "Finding",
    "RegisteredCheck",
    "Severity",
    "check",
    "collect_environment",
    "iter_checks",
    "register_check",
    "run_checks",
    "worst_severity",
]
