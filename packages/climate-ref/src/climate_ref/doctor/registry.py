"""
The set of checks that `ref doctor` runs.

A check is a plain function over a `DoctorContext`, declared with `@check`. The built-in
checks live under `climate_ref.doctor.checks`; a package outside this one contributes its
own by advertising a module in the ``climate-ref.doctor-checks`` entry point group, which
is imported for its ``@check`` declarations:

    [project.entry-points."climate-ref.doctor-checks"]
    my_provider = "my_package.doctor_checks"
"""

import importlib.metadata
from collections.abc import Callable, Iterable

from attrs import evolve, frozen
from loguru import logger

from climate_ref.doctor.context import DoctorContext
from climate_ref.doctor.findings import SEVERITY_ORDER, Finding, Severity

CHECK_ENTRY_POINT_GROUP = "climate-ref.doctor-checks"
"""Entry point group through which another package contributes checks."""

BUILT_IN = "built-in"
"""`RegisteredCheck.source` for a check that ships with `climate_ref`."""

CheckFunction = Callable[[DoctorContext], list[Finding]]


@frozen
class RegisteredCheck:
    """
    A check and what it is for.
    """

    slug: str
    """Stable identifier, e.g. ``duplicate-coverage``. Stamped onto the check's findings."""

    description: str
    """One line describing what the check looks for, shown by ``ref doctor --list``."""

    func: CheckFunction
    """The check itself."""

    source: str = BUILT_IN
    """Where the check came from: `BUILT_IN`, or the name of the entry point that supplied it."""

    def __call__(self, context: DoctorContext) -> list[Finding]:
        """
        Run the check, stamping this check's slug onto every finding.

        Parameters
        ----------
        context
            The deployment to check.

        Returns
        -------
        :
            The findings, each carrying `slug`.
        """
        return [evolve(finding, check=self.slug) for finding in self.func(context)]


_REGISTRY: dict[str, RegisteredCheck] = {}
_LOAD_ERRORS: dict[str, str] = {}
_LOADED_PLUGINS: set[str] = set()


def register_check(registered: RegisteredCheck) -> None:
    """
    Add a check to the registry.

    Parameters
    ----------
    registered
        The check to add.

    Raises
    ------
    ValueError
        If another check already claims the same slug. Two checks sharing a slug would be
        indistinguishable in the output and in ``--only``.
    """
    existing = _REGISTRY.get(registered.slug)
    if existing is not None:
        raise ValueError(
            f"A doctor check with the slug '{registered.slug}' is already registered (from {existing.source})"
        )
    _REGISTRY[registered.slug] = registered


def check(slug: str, description: str) -> Callable[[CheckFunction], CheckFunction]:
    """
    Declare a function as a doctor check.

    The decorated function is returned unchanged, so it stays directly callable in tests.

    Parameters
    ----------
    slug
        Stable identifier for the check, in kebab-case.
    description
        One line describing what the check looks for.

    Returns
    -------
    :
        A decorator that registers the function and returns it.
    """

    def decorate(func: CheckFunction) -> CheckFunction:
        register_check(RegisteredCheck(slug=slug, description=description, func=func))
        return func

    return decorate


def load_plugin_checks() -> dict[str, str]:
    """
    Import the check modules advertised by other packages.

    Importing a module runs its ``@check`` declarations. A module that cannot be imported is
    recorded rather than raised, so one broken plugin does not take the command down. The
    error is reported as a finding by `run_checks`.

    Returns
    -------
    :
        The load errors, keyed by entry point name.
    """
    for entry_point in importlib.metadata.entry_points(group=CHECK_ENTRY_POINT_GROUP):
        # Importing twice would re-run the `@check` declarations, which refuse a duplicate slug.
        if entry_point.name in _LOADED_PLUGINS:
            continue
        _LOADED_PLUGINS.add(entry_point.name)

        known = set(_REGISTRY)
        try:
            entry_point.load()
        except Exception as exc:
            logger.exception(f"Could not load doctor checks from '{entry_point.value}'")
            _LOAD_ERRORS[entry_point.name] = f"{type(exc).__name__}: {exc}"
            continue
        for slug in set(_REGISTRY) - known:
            _REGISTRY[slug] = evolve(_REGISTRY[slug], source=entry_point.name)

    return dict(_LOAD_ERRORS)


def iter_checks() -> tuple[RegisteredCheck, ...]:
    """
    Every check available to this deployment, built-in first, then each plugin's.

    Returns
    -------
    :
        The registered checks, ordered by source and then by registration order.
    """
    load_plugin_checks()
    built_in = [c for c in _REGISTRY.values() if c.source == BUILT_IN]
    plugin = [c for c in _REGISTRY.values() if c.source != BUILT_IN]
    return tuple(built_in + plugin)


def run_checks(
    context: DoctorContext,
    checks: Iterable[RegisteredCheck] | None = None,
) -> list[Finding]:
    """
    Run the checks and collect their findings, worst first.

    A check that raises is reported as a finding rather than stopping the run, so one broken
    check cannot hide the others. So is a plugin that could not be imported, because a check
    that never ran must not look like a check that passed.

    Parameters
    ----------
    context
        The deployment to check.
    checks
        The checks to run. Defaults to every registered check.

    Returns
    -------
    :
        Findings ordered by severity, then by the check that produced them.
    """
    findings: list[Finding] = []

    if checks is None:
        checks = iter_checks()
        findings.extend(
            Finding(
                check="plugin-load",
                severity=Severity.ERROR,
                summary=f"Doctor checks from '{name}' could not be loaded, so they did not run",
                detail=error,
            )
            for name, error in sorted(load_plugin_checks().items())
        )

    for registered in checks:
        try:
            findings.extend(registered(context))
        except Exception as exc:
            logger.exception(f"Check '{registered.slug}' failed")
            findings.append(
                Finding(
                    check=registered.slug,
                    severity=Severity.ERROR,
                    summary=f"Check '{registered.slug}' could not run",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

    return sorted(findings, key=lambda f: (SEVERITY_ORDER.index(f.severity), f.check, f.summary))
