"""
Test that all of our modules can be imported

Also test that associated constants are set correctly

Thanks https://stackoverflow.com/a/25562415/10473080
"""

import importlib
import pkgutil

import typer
from loguru import logger

ignored_modules = [
    "climate_ref.executor.hpc",  # Optional dependencies not installed on Windows
    "climate_ref.conftest_plugin",  # Requires pytest (optional via climate-ref[test])
]


def import_submodules(package_name):
    """
    Test import of submodules
    """
    package = importlib.import_module(package_name)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        logger.info(f"Importing {full_name}")

        # Ignore certain modules that have optional dependencies
        if full_name in ignored_modules:
            logger.warning(f"Skipping {full_name}")
            continue

        importlib.import_module(full_name)
        if is_pkg:
            import_submodules(full_name)


def main(package: str = typer.Argument(..., help="List of package names to test import for")):
    try:
        import_submodules(package)
        pkg = importlib.import_module(package)
        version = getattr(pkg, "__version__", "<no __version__>")
        print(f"{package} version: {version}")
    except Exception as e:
        logger.exception("Failed to import package")
        print(f"Failed to import {package}: {e}")
        raise typer.Abort()


if __name__ == "__main__":
    typer.run(main)
