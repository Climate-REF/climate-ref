"""
Check a deployment for problems that a solve would otherwise hide.
"""

from typing import Annotated

import typer
from rich.console import Console

from climate_ref.doctor import (
    CHECKS,
    DoctorContext,
    Finding,
    Severity,
    run_checks,
    worst_severity,
)

_SEVERITY_STYLE = {
    Severity.ERROR: "red",
    Severity.WARNING: "yellow",
    Severity.INFO: "cyan",
}


def _print_findings(console: Console, findings: list[Finding], verbose: bool) -> None:
    for finding in findings:
        style = _SEVERITY_STYLE.get(finding.severity, "white")
        console.print(f"[{style}]{finding.severity.upper():<7}[/{style}] {finding.summary}")
        if finding.detail and verbose:
            console.print(f"          {finding.detail}", style="dim")
        console.print(f"          [dim]check: {finding.check}[/dim]")


def doctor(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--quiet", help="Include the explanation and remedy for each finding."),
    ] = True,
    strict: Annotated[
        bool,
        typer.Option(help="Exit non-zero for warnings as well as errors."),
    ] = False,
) -> None:
    """
    Check this deployment for data and configuration problems.

    Reports reference data that is missing (so its diagnostics will never run), data ingested
    under a source type nothing asks for, datasets whose files cover the same period twice,
    and collections that overlap.

    Exits non-zero if any error is found, or any warning when --strict is used.
    """
    console = ctx.obj.console
    context = DoctorContext(config=ctx.obj.config, database=ctx.obj.database)

    findings = run_checks(context)

    if not findings:
        console.print(f"[green]No problems found[/green] ({len(CHECKS)} checks)")
        return

    _print_findings(console, findings, verbose)

    counts = {severity: sum(f.severity == severity for f in findings) for severity in _SEVERITY_STYLE}
    summary = ", ".join(f"{count} {severity}" for severity, count in counts.items() if count)
    console.print(f"\n{len(findings)} finding(s): {summary}")

    worst = worst_severity(findings)
    if worst == Severity.ERROR or (strict and worst == Severity.WARNING):
        raise typer.Exit(1)
