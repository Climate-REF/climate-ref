"""
Check a deployment for problems that a solve would otherwise hide.
"""

import json
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Annotated

import typer
from attrs import asdict
from rich.console import Console
from rich.padding import Padding

from climate_ref.doctor import (
    DoctorContext,
    DoctorReport,
    Finding,
    Severity,
    diagnose,
    iter_checks,
    worst_severity,
)
from climate_ref.text import pluralise

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


def _group[K](findings: Sequence[Finding], key: Callable[[Finding], K]) -> dict[K, list[Finding]]:
    """Group findings, keeping both the groups and their contents in the order given."""
    grouped: dict[K, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(key(finding), []).append(finding)
    return grouped


def _indented(console: Console, text: str, indent: int, style: str = "") -> None:
    """Print text at an indent that its wrapped continuation lines keep."""
    console.print(Padding(text, (0, 0, 0, indent), expand=False), style=style)


def _severity_counts(findings: Sequence[Finding]) -> str:
    # Severity is declared worst-first
    counts = {severity: sum(f.severity == severity for f in findings) for severity in Severity}
    return ", ".join(
        pluralise(count, severity, severity if severity == Severity.INFO else None)
        for severity, count in counts.items()
        if count
    )


def _print_findings(console: Console, findings: Sequence[Finding], verbose: bool) -> None:
    """
    Print the findings grouped by the check that produced them.

    Everything shared within a group is stated once: the check that found them, and the remedy
    they have in common. A deployment missing twenty reference datasets is one instruction and
    twenty names, not twenty copies of the instruction.
    """
    for check, found in _group(findings, lambda f: f.check).items():
        worst = worst_severity(found)
        style = _SEVERITY_STYLE.get(worst, "white") if worst else "white"
        console.print(f"[bold {style}]{check}[/bold {style}] [dim]{_severity_counts(found)}[/dim]")

        # Only worth naming a finding's severity where the group holds more than one.
        mixed = len({f.severity for f in found}) > 1

        for (remedy, command), sharing in _group(found, lambda f: (f.remedy, f.command)).items():
            if verbose and remedy:
                # Ahead of the findings, so a long list is read already knowing what to do about it.
                _indented(console, remedy, indent=2, style="dim")
            if verbose and command:
                # Printed alone and unwrapped so it stays pasteable in a narrow terminal.
                console.print(Padding(command, (0, 0, 0, 2), expand=False), style="cyan", soft_wrap=True)
            if verbose and (remedy or command):
                console.print()
            for finding in sharing:
                prefix = f"[{_SEVERITY_STYLE[finding.severity]}]{finding.severity}[/] " if mixed else ""
                _indented(console, f"{prefix}{finding.summary}", indent=2)
                if finding.detail and verbose:
                    _indented(console, finding.detail, indent=4, style="dim")
            console.print()


def _print_environment(console: Console, sections: dict[str, dict[str, str]]) -> None:
    for name, values in sections.items():
        if not values:
            continue
        console.print(f"\n[bold]{name}[/bold]")
        for key, value in values.items():
            console.print(f"  {key}: {value}", style="dim")


def _summary_line(findings: Sequence[Finding], check_count: int) -> str:
    return (
        f"{pluralise(len(findings), 'finding')} from {pluralise(check_count, 'check')}: "
        f"{_severity_counts(findings)}"
    )


def _table_cell(finding: Finding) -> str:
    """Render a finding as one Markdown cell, neutralising what would break the table."""
    parts = [f"{finding.summary}.", finding.detail, finding.remedy]
    if finding.command:
        parts.append(f"`{finding.command}`")
    return " ".join(part for part in parts if part).replace("|", "\\|").replace("\n", " ")


def _render_markdown(report: DoctorReport) -> str:
    """Render a report that can be pasted into an issue as-is."""
    findings = report.findings
    lines = ["## `ref doctor`", ""]
    if findings:
        lines.append(_summary_line(findings, report.check_count))
        lines.append("")
        lines.append("| Severity | Check | Finding |")
        lines.append("| --- | --- | --- |")
        lines.extend(
            f"| {finding.severity} | `{finding.check}` | {_table_cell(finding)} |" for finding in findings
        )
    else:
        lines.append(f"No problems found ({pluralise(report.check_count, 'check')}).")

    if report.environment is not None:
        lines.extend(["", "<details><summary>Environment</summary>", ""])
        for name, values in report.environment.items():
            if not values:
                continue
            lines.append(f"**{name}**")
            lines.append("")
            lines.extend(f"- `{key}`: {value}" for key, value in values.items())
            lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def _render_json(report: DoctorReport) -> str:
    rendered: dict[str, object] = {
        "findings": [asdict(finding) for finding in report.findings],
        "worst_severity": report.worst_severity,
    }
    if report.environment is not None:
        rendered["environment"] = report.environment
    return json.dumps(rendered, indent=2)


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

    Use `--format markdown` to produce a report,
    including a description of this deployment that can be pasted into a bug report.

    Exits non-zero if any error is found, or any warning when --strict is used.
    """
    console = ctx.obj.console

    if list_checks:
        _list_checks(console)
        return

    # The environment is the point of the machine-readable formats, and noise in the default one.
    if environment is None:
        environment = output_format != DoctorFormat.text

    context = DoctorContext(config=ctx.obj.config, database=ctx.obj.database)
    report = diagnose(context, environment=environment)

    if output_format == DoctorFormat.json:
        print(_render_json(report))
    elif output_format == DoctorFormat.markdown:
        print(_render_markdown(report))
    else:
        if report.findings:
            # Stated before the findings so the size of the problem does not need scrolling to.
            console.print(f"[bold]{_summary_line(report.findings, report.check_count)}[/bold]")
            _print_findings(console, report.findings, verbose)
        else:
            console.print(f"[green]No problems found[/green] ({pluralise(report.check_count, 'check')})")
        if report.environment is not None:
            _print_environment(console, report.environment)

    if report.worst_severity == Severity.ERROR or (strict and report.worst_severity == Severity.WARNING):
        raise typer.Exit(1)
