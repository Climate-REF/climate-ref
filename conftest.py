"""
Re-useable fixtures etc. for tests that are shared across the whole project

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files

All shared fixtures are defined in ``climate_ref.conftest_plugin`` and are
auto-discovered via the ``pytest11`` entry point (registered as ``climate_ref``).
"""

pytest_plugins = ("celery.contrib.pytest",)
