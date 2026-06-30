"""
The Typer application backing the ``ref test-cases`` command group.
"""

import typer

app = typer.Typer(
    help=(
        "Test data management commands for diagnostic development.\n\n"
        "These commands are intended for diagnostic developers and require "
        "a source checkout of the project with test data directories available."
    )
)
