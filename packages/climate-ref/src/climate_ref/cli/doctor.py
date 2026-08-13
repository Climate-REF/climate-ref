"""
Check a deployment for problems that a solve would otherwise hide.
"""

import json
from enum import StrEnum
from typing import Annotated

import typer
from attrs import asdict
from rich.console import Console

from climate_ref.doctor import (
    SEVERITY_ORDER,
    DoctorContext,
    EnvironmentReport,
    Finding,
    Severity,
    collect_environment,
    iter_checks,
    run_checks,
    worst_severity,
)

_SEVERITY_STYLE = {
    Severity.ERROR: "red",
    Severity.WARNING: "yellow",
    Severity.INFO: "cyan",
}


class DoctorFormat(StrEnum):
    """
    Output format for ``ref doctor``.
    """

    text = "text"
    markdown = "markdown"
    json = "json"


def _print_findings(console: Console, findings: list[Finding], verbose: bool) -> None:
    for finding in findings:
        style = _SEVERITY_STYLE.get(finding.severity, "white")
        console.print(f"[{style}]{finding.severity.upper():<7}[/{style}] {finding.summary}")
        if finding.detail and verbose:
            console.print(f"          {finding.detail}", style="dim")
        console.print(f"          [dim]check: {finding.check}[/dim]")


def _print_environment(console: Console, environment: EnvironmentReport) -> None:
    for name, values in environment.sections.items():
        if not values:
            continue
        console.print(f"\n[bold]{name}[/bold]")
        for key, value in values.items():
            console.print(f"  {key}: {value}", style="dim")


def _summary_line(findings: list[Finding]) -> str:
    counts = {severity: sum(f.severity == severity for f in findings) for severity in SEVERITY_ORDER}
    summary = ", ".join(f"{count} {severity}" for severity, count in counts.items() if count)
    return f"{len(findings)} finding(s): {summary}"


def _table_cell(finding: Finding) -> str:
    """Render a finding as one Markdown cell, neutralising what would break the table."""
    detail = f" {finding.detail}" if finding.detail else ""
    return f"{finding.summary}.{detail}".replace("|", "\\|").replace("\n", " ")


def _render_markdown(findings: list[Finding], environment: EnvironmentReport | None, check_count: int) -> str:
    """Render a report that can be pasted into an issue as-is."""
    lines = ["## `ref doctor`", ""]
    if findings:
        lines.append(_summary_line(findings))
        lines.append("")
        lines.append("| Severity | Check | Finding |")
        lines.append("| --- | --- | --- |")
        lines.extend(
            f"| {finding.severity} | `{finding.check}` | {_table_cell(finding)} |" for finding in findings
        )
    else:
        lines.append(f"No problems found ({check_count} checks).")

    if environment is not None:
        lines.extend(["", "<details><summary>Environment</summary>", ""])
        for name, values in environment.sections.items():
            if not values:
                continue
            lines.append(f"**{name}**")
            lines.append("")
            lines.extend(f"- `{key}`: {value}" for key, value in values.items())
            lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def _render_json(findings: list[Finding], environment: EnvironmentReport | None) -> str:
    report: dict[str, object] = {
        # `asdict` rather than named fields, so a field added to `Finding` reaches the JSON
        # output as well as the text one.
        "findings": [asdict(finding) for finding in findings],
        "worst_severity": worst_severity(findings),
    }
    if environment is not None:
        report["environment"] = environment.sections
    return json.dumps(report, indent=2)


def _list_checks(console: Console) -> None:
    for registered in iter_checks():
        console.print(f"{registered.slug} [dim]({registered.source})[/dim]")
        console.print(f"  {registered.description}", style="dim")


def doctor(  # noqa: PLR0913
    ctx: typer.Context,
    output_format: Annotated[
        DoctorFormat,
        typer.Option(
            "--format",
            help="Output format: 'text' (default), 'markdown' to paste into an issue, or 'json'.",
        ),
    ] = DoctorFormat.text,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--quiet", help="Include the explanation and remedy for each finding."),
    ] = True,
    strict: Annotated[
        bool,
        typer.Option(help="Exit non-zero for warnings as well as errors."),
    ] = False,
    environment: Annotated[
        bool | None,
        typer.Option(
            "--environment/--no-environment",
            help="Include a description of this deployment. On by default for the markdown and json formats.",
        ),
    ] = None,
    list_checks: Annotated[
        bool,
        typer.Option("--list", help="List the available checks and where they came from, then exit."),
    ] = False,
) -> None:
    """
    Check this deployment for data and configuration problems.

    Reports reference data that is missing (so its diagnostics will never run), data ingested
    under a source type nothing asks for, datasets whose files cover the same period twice,
    and collections that overlap.

    Use `--format markdown` to produce a report, including a description of this deployment,
    that can be pasted into a bug report.

    Exits non-zero if any error is found, or any warning when --strict is used.
    """
    console = ctx.obj.console

    if list_checks:
        _list_checks(console)
        return

    context = DoctorContext(config=ctx.obj.config, database=ctx.obj.database)
    findings = run_checks(context)
    check_count = len(iter_checks())

    # The environment is the point of the machine-readable formats, and noise in the default one.
    if environment is None:
        environment = output_format != DoctorFormat.text
    report = collect_environment(context) if environment else None

    if output_format == DoctorFormat.json:
        print(_render_json(findings, report))
    elif output_format == DoctorFormat.markdown:
        print(_render_markdown(findings, report, check_count))
    else:
        if findings:
            _print_findings(console, findings, verbose)
            console.print(f"\n{_summary_line(findings)}")
        else:
            console.print(f"[green]No problems found[/green] ({check_count} checks)")
        if report is not None:
            _print_environment(console, report)

    worst = worst_severity(findings)
    if worst == Severity.ERROR or (strict and worst == Severity.WARNING):
        raise typer.Exit(1)
