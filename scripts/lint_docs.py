#!/usr/bin/env python
"""
Lint documentation for two recurring classes of error:

1. CLI option drift -- ``ref ...`` commands shown in docs/READMEs that use a
   malformed flag (``---foo``) or a flag that does not exist on the resolved
   ``ref`` command (e.g. ``--output directory`` instead of ``--output-directory``).
2. Broken internal links -- GitHub ``blob``/``tree`` links into this repository
   that point at a path which no longer exists.

The linter covers the top-level README, the ``docs/`` tree, and every package
README so that provider docs are checked alongside the central docs.

Run directly (``python scripts/lint_docs.py``) or via pre-commit. Exits non-zero
if any problem is found.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Markdown files to lint: top-level README, all docs, and every package README.
DOC_GLOBS = (
    "README.md",
    "docs/**/*.md",
    "packages/*/README.md",
)

# GitHub blob/tree links that point back into this repository.
_REPO_BLOB_RE = re.compile(
    r"https://github\.com/Climate-REF/climate-ref/(?:blob|tree)/[^/]+/(?P<path>[^)\s#]+)"
)
# Markdown fenced code blocks (``` ... ```), capturing the inner body.
_FENCE_RE = re.compile(r"```[^\n]*\n(?P<body>.*?)```", re.DOTALL)
# Shell tokens that end a command (redirects, pipes, separators).
_SHELL_STOP = {">", ">>", "<", "|", "||", "&&", ";", "&"}


@dataclass
class Problem:
    """A single linting problem."""

    path: Path
    line: int
    message: str

    def render(self) -> str:
        """Render the problem as a ``path:line: message`` string."""
        rel = self.path.relative_to(REPO_ROOT)
        return f"{rel}:{self.line}: {self.message}"


def _iter_doc_files() -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in DOC_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path.is_file():
                seen.setdefault(path.resolve(), None)
    return list(seen)


def _build_cli_options() -> dict[tuple[str, ...], set[str]] | None:
    """
    Introspect the ``ref`` CLI and return the valid long options per command path.

    The empty tuple key holds the global options available on every command.
    Returns ``None`` if the CLI cannot be imported (the flag check is then skipped).
    """
    # Imported lazily and guarded: the CLI may not be installed in every env that
    # runs this linter, in which case flag validation is skipped.
    try:
        import click  # noqa: PLC0415
        import typer  # noqa: PLC0415

        from climate_ref.cli import app  # noqa: PLC0415
    except Exception:  # pragma: no cover - environment without the package installed
        return None

    command = typer.main.get_command(app)
    options: dict[tuple[str, ...], set[str]] = {}

    def long_opts(cmd: click.Command) -> set[str]:
        """Collect the ``--long`` option names defined on a click command."""
        names: set[str] = set()
        for param in cmd.params:
            for opt in getattr(param, "opts", []) + getattr(param, "secondary_opts", []):
                if opt.startswith("--"):
                    names.add(opt)
        return names

    # Global options live on the root group.
    options[()] = long_opts(command)

    def walk(cmd: click.Command, path: tuple[str, ...]) -> None:
        """Recurse into command groups, recording each command's long options."""
        if isinstance(cmd, click.Group):
            for name, sub in cmd.commands.items():
                sub_path = (*path, name)
                options[sub_path] = long_opts(sub)
                walk(sub, sub_path)

    walk(command, ())
    return options


def _resolve_command(tokens: list[str], options: dict[tuple[str, ...], set[str]]) -> tuple[str, ...] | None:
    """
    Resolve the deepest known command path from the token stream.

    Flags (and any interspersed global-option values that appear before the
    subcommand) are skipped, so ``ref --verbose test-cases run`` still resolves
    to ``test-cases run``. Once a command path has started, the first token that
    does not extend it is treated as a positional argument and ends resolution.
    """
    path: tuple[str, ...] = ()
    for token in tokens:
        if token.startswith("-"):
            continue
        candidate = (*path, token)
        if candidate in options:
            path = candidate
        elif path == ():
            # Likely a value for a preceding global option (e.g. a path passed to
            # --configuration-directory); keep scanning for the real subcommand.
            continue
        else:
            # First positional argument after a resolved command: stop.
            break
    return path or None


def _check_ref_command(line: str, options: dict[tuple[str, ...], set[str]] | None) -> list[str]:
    """Return error messages for a single ``ref ...`` command line."""
    errors: list[str] = []

    # Tokenise, stopping at the first shell operator.
    raw_tokens: list[str] = []
    for token in line.split():
        if token in _SHELL_STOP or any(token.startswith(op) for op in (">", "<", "|")):
            break
        raw_tokens.append(token)

    if not raw_tokens or raw_tokens[0] != "ref":
        return errors

    # Malformed multi-dash flags (e.g. ---foo) regardless of CLI availability.
    for token in raw_tokens:
        if re.match(r"^-{3,}", token):
            errors.append(f"malformed flag '{token}' (too many leading dashes)")

    if options is None:
        return errors

    tokens = raw_tokens[1:]  # drop the leading 'ref'
    command_path = _resolve_command(tokens, options)
    if command_path is None:
        return errors

    valid = options.get(command_path, set()) | options[()]
    saw_double_dash = False
    for token in tokens:
        if token == "--":  # noqa: S105 - shell argument separator, not a secret
            saw_double_dash = True
            continue
        if saw_double_dash:
            continue
        # Only validate well-formed long options ('--' followed by a letter).
        if not re.match(r"^--[a-zA-Z]", token):
            continue
        name = token.split("=", 1)[0]
        if name not in valid:
            errors.append(f"unknown flag '{name}' for command 'ref {' '.join(command_path)}'")

    return errors


def _lint_file(path: Path, options: dict[tuple[str, ...], set[str]] | None) -> list[Problem]:
    problems: list[Problem] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1. CLI flag checks: only inside fenced code blocks, where commands live.
    for match in _FENCE_RE.finditer(text):
        start_line = text[: match.start()].count("\n") + 1
        body = match.group("body")
        for offset, raw_line in enumerate(body.splitlines(), start=1):
            stripped = raw_line.strip().lstrip("$ ").strip()
            if not stripped.startswith("ref "):
                continue
            for message in _check_ref_command(stripped, options):
                problems.append(Problem(path, start_line + offset, message))

    # 2. Internal GitHub blob/tree link checks (anywhere in the file).
    for lineno, raw_line in enumerate(lines, start=1):
        for link in _REPO_BLOB_RE.finditer(raw_line):
            target = REPO_ROOT / link.group("path")
            if not target.exists():
                problems.append(Problem(path, lineno, f"broken internal link to '{link.group('path')}'"))

    return problems


def main() -> int:
    """Lint all documentation files and return a process exit code."""
    options = _build_cli_options()
    if options is None:
        print("warning: 'ref' CLI not importable; skipping flag validation", file=sys.stderr)

    problems: list[Problem] = []
    for path in _iter_doc_files():
        problems.extend(_lint_file(path, options))

    if problems:
        print("Documentation lint failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem.render()}", file=sys.stderr)
        return 1

    print("Documentation lint passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
