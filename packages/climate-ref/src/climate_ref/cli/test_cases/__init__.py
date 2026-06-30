"""
Test management commands for diagnostic development.

These commands are intended for diagnostic and require
a source checkout of the project with test data directories available.

The command group is split across modules by concern:

- :mod:`~climate_ref.cli.test_cases.discovery` -- ``fetch`` / ``list``
- :mod:`~climate_ref.cli.test_cases.run` -- ``run``
- :mod:`~climate_ref.cli.test_cases.baselines` -- ``replay`` / ``mint`` / ``build``
- :mod:`~climate_ref.cli.test_cases.store` -- ``sync`` / ``check-store``
- :mod:`~climate_ref.cli.test_cases.ci_gate` -- ``ci-gate``
- :mod:`~climate_ref.cli.test_cases.migrate` -- ``migrate-manifests``
"""

from importlib import import_module

from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._catalog import (
    _build_catalog,
    _fetch_and_build_catalog,
    _solve_test_case,
)
from climate_ref.cli.test_cases._common import _iter_test_cases

# Import each command module for its registration side effect on ``app``.
# The sequence below is the order the verbs appear in ``ref test-cases --help``
# a dynamic import keeps it explicit, where a plain ``import`` block would be alphabetised by ruff.
for _command_module in ("discovery", "run", "baselines", "store", "ci_gate", "migrate"):
    import_module(f"{__name__}.{_command_module}")

__all__ = [
    "_build_catalog",
    "_fetch_and_build_catalog",
    "_iter_test_cases",
    "_solve_test_case",
    "app",
]
