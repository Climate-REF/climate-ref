#!/usr/bin/env python
"""
Render a human-readable diff of the regression baselines changed on a branch.

A mint rewrites each test case's ``manifest.json`` and uploads the curated native outputs to the
content-addressed object store.
The manifest diff therefore names every native file that changed, and both the old and the new blob
remain fetchable by digest, so the change can be reviewed without checking anything out locally.

Text outputs (JSON, CSV, YAML, HTML) are fetched from the store and rendered as a unified diff.
Binary outputs (NetCDF, PNG) are reported by size delta and linked, because a byte diff would be noise.

Usage:
  uv run python scripts/ci/mint_diff.py [--base origin/main] [--output summary.md]

  --base            git ref to compare against. Defaults to origin/${GITHUB_BASE_REF:-main}.
  --output          write the full, uncapped report here as well as to stdout.
  --comment-output  write a copy capped to GitHub's comment size limit here.
  --store-url       base URL of the native store. Defaults to $REF_NATIVE_STORE_URL.
  --no-fetch        skip all network access and report size deltas only.

Exits 0 whether or not anything changed. This reports, it does not gate.
"""

import argparse
import difflib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from git import GitCommandError, Repo

from climate_ref_core.regression.manifest import Manifest, NativeEntry

DEFAULT_STORE_URL = "https://baselines.climate-ref.org"

# Cloudflare serves a 403 to the stdlib default agent, so identify the script instead.
USER_AGENT = "climate-ref-mint-diff"

# A manifest path needs a diagnostic and test-case directory before a label can be built from it.
_MIN_LABEL_PARTS = 3

# Extensions whose blobs are worth fetching and diffing line by line.
TEXT_SUFFIXES = frozenset({".json", ".csv", ".yml", ".yaml", ".html", ".txt", ".md", ".log"})

# A blob larger than this is summarised rather than diffed. Keeps a runaway HTML report
# from stalling the job on the download alone.
MAX_FETCH_BYTES = 2_000_000

# Unified-diff lines kept per file before the rest is elided.
MAX_DIFF_LINES = 200

# Binary rows listed per case before the rest is elided. A large ILAMB or PMP case emits
# hundreds of plots, and the per-file size delta stops being informative long before then.
MAX_BINARY_ROWS = 30

# GitHub rejects a comment body over 65536 bytes, so leave headroom for the truncation notice.
MAX_COMMENT_BYTES = 60_000


def open_repo() -> Repo:
    """Return the repository containing the current directory."""
    return Repo(Path.cwd(), search_parent_directories=True)


def changed_manifests(repo: Repo, base: str) -> list[str]:
    """
    Return the repo-relative paths of every test-case manifest that differs from ``base``.

    Uses the merge-base (``base...HEAD``) so commits landing on the base branch after the
    feature branch forked are not misreported as baseline changes.
    """
    pathspec = ":(glob)packages/**/test-data/**/manifest.json"
    try:
        out = repo.git.diff("--name-only", f"{base}...HEAD", "--", pathspec)
    except GitCommandError:
        # A shallow clone may have no merge-base with the base ref. A two-dot diff over-reports
        # (it also shows base-branch commits), which is the safe direction for a report.
        out = repo.git.diff("--name-only", base, "HEAD", "--", pathspec)
    return sorted(line for line in out.splitlines() if line.strip())


def load_at_ref(repo: Repo, ref: str, rel_path: str) -> Manifest | None:
    """Load a manifest as it exists at ``ref``, or ``None`` when absent there."""
    try:
        text = repo.git.show(f"{ref}:{rel_path}")
    except GitCommandError:
        return None
    return Manifest.loads(text, source=f"{ref}:{rel_path}")


def case_label(rel_path: str) -> str:
    """
    Derive a ``provider/diagnostic/test-case`` label from a manifest path.

    ``packages/climate-ref-pmp/tests/test-data/annual-cycle/cmip6-ts/manifest.json``
    becomes ``pmp/annual-cycle/cmip6-ts``.
    """
    parts = Path(rel_path).parts
    provider = parts[1].removeprefix("climate-ref-") if len(parts) > 1 else "?"
    tail = parts[-3:-1] if len(parts) >= _MIN_LABEL_PARTS else ()
    return "/".join((provider, *tail))


@dataclass
class FileChange:
    """One native output file that was added, removed, or changed by the mint."""

    name: str
    old: NativeEntry | None
    new: NativeEntry | None
    diff: str | None = None
    note: str | None = None

    @property
    def status(self) -> str:
        """``added``, ``removed`` or ``changed``."""
        if self.old is None:
            return "added"
        if self.new is None:
            return "removed"
        return "changed"

    @property
    def is_text(self) -> bool:
        """Whether this file's contents are worth diffing line by line."""
        return Path(self.name).suffix.lower() in TEXT_SUFFIXES


@dataclass
class CaseDiff:
    """Everything that changed for a single test case."""

    label: str
    rel_path: str
    base: Manifest | None
    head: Manifest | None
    native: list[FileChange] = field(default_factory=list)
    committed: list[str] = field(default_factory=list)
    metadata: list[str] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        """Whether the whole test case is new on this branch."""
        return self.base is None

    @property
    def is_removed(self) -> bool:
        """Whether the whole test case was deleted on this branch."""
        return self.head is None


def _metadata_changes(base: Manifest | None, head: Manifest | None) -> list[str]:
    """Describe the scalar manifest fields that moved."""
    if head is None:
        return ["test case removed"]
    if base is None:
        return [f"new test case at `test_case_version` {head.test_case_version}"]
    changes = []
    for name in ("test_case_version", "diagnostic_version", "catalog_hash", "schema"):
        old, new = getattr(base, name), getattr(head, name)
        if old != new:
            changes.append(f"`{name}`: `{old}` -> `{new}`")
    return changes


def _committed_changes(base: Manifest | None, head: Manifest | None) -> list[str]:
    """Name the committed regression artefacts whose digest moved."""
    old = base.committed if base else {}
    new = head.committed if head else {}
    names = sorted(set(old) | set(new))
    out = []
    for name in names:
        if old.get(name) != new.get(name):
            if name not in old:
                out.append(f"`{name}` (added)")
            elif name not in new:
                out.append(f"`{name}` (removed)")
            else:
                out.append(f"`{name}`")
    return out


def _fetch(store_url: str, digest: str) -> tuple[bytes | None, str]:
    """
    Fetch a blob by digest.

    Returns a ``(payload, reason)`` pair, where ``payload`` is ``None`` on failure and
    ``reason`` describes it. Cloudflare rejects the stdlib default agent, so one is set.
    """
    url = f"{store_url.rstrip('/')}/{digest}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.read(MAX_FETCH_BYTES + 1), ""
    except urllib.error.HTTPError as exc:
        return None, f"store returned HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, f"store unreachable ({exc})"


def _as_lines(raw: bytes, name: str) -> list[str]:
    """
    Decode a blob into diffable lines.

    JSON is re-serialised with indentation first, because a minified bundle would otherwise
    diff as a single unreadable line.
    """
    text = raw.decode("utf-8", errors="replace")
    if Path(name).suffix.lower() == ".json":
        try:
            text = json.dumps(json.loads(text), indent=2, sort_keys=True)
        except json.JSONDecodeError:
            pass
    return text.splitlines()


def _text_diff(store_url: str, change: FileChange) -> tuple[str | None, str | None]:
    """
    Build the unified diff for one text file.

    Returns a ``(diff, note)`` pair. Exactly one is set: ``note`` explains why no diff
    could be produced (too large, or the store did not serve a blob).
    """
    for entry in (change.old, change.new):
        if entry is not None and entry.size > MAX_FETCH_BYTES:
            return None, f"too large to diff ({entry.size:,} B)"

    old_raw, old_reason = _fetch(store_url, change.old.sha256) if change.old else (b"", "")
    new_raw, new_reason = _fetch(store_url, change.new.sha256) if change.new else (b"", "")
    if old_raw is None or new_raw is None:
        return None, old_reason or new_reason

    lines = list(
        difflib.unified_diff(
            _as_lines(old_raw, change.name),
            _as_lines(new_raw, change.name),
            fromfile=f"old {change.old.sha256[:12] if change.old else '(absent)'}",
            tofile=f"new {change.new.sha256[:12] if change.new else '(absent)'}",
            lineterm="",
            n=3,
        )
    )
    if not lines:
        return None, "identical after decoding"
    if len(lines) > MAX_DIFF_LINES:
        elided = len(lines) - MAX_DIFF_LINES
        lines = [*lines[:MAX_DIFF_LINES], f"... {elided:,} more diff line(s) elided"]
    return "\n".join(lines), None


def build_case_diff(repo: Repo, base: str, rel_path: str, store_url: str, fetch: bool) -> CaseDiff | None:
    """
    Collect every change to one test case.

    A case deleted on this branch has no head manifest. It is still reported, as a removal,
    because dropping a test case is a baseline change a reviewer needs to see.
    Returns ``None`` only when the manifest is absent from both sides, which leaves nothing to say.
    """
    head_path = Path(repo.working_tree_dir or ".") / rel_path
    head = Manifest.load(head_path) if head_path.exists() else None
    base_manifest = load_at_ref(repo, base, rel_path)
    if head is None and base_manifest is None:
        return None

    diff = CaseDiff(
        label=case_label(rel_path),
        rel_path=rel_path,
        base=base_manifest,
        head=head,
        committed=_committed_changes(base_manifest, head),
        metadata=_metadata_changes(base_manifest, head),
    )

    old_native = base_manifest.native if base_manifest else {}
    new_native = head.native if head else {}
    for name in sorted(set(old_native) | set(new_native)):
        old, new = old_native.get(name), new_native.get(name)
        if old is not None and new is not None and old.sha256 == new.sha256:
            continue
        change = FileChange(name=name, old=old, new=new)
        if change.is_text:
            if fetch:
                change.diff, change.note = _text_diff(store_url, change)
            else:
                change.note = "fetching disabled"
        diff.native.append(change)

    return diff


def _size_delta(change: FileChange) -> str:
    """Render the size column for a native file."""
    if change.old is None:
        return f"{change.new.size:,} B"  # type: ignore[union-attr]
    if change.new is None:
        return f"was {change.old.size:,} B"
    delta = change.new.size - change.old.size
    return f"{change.old.size:,} -> {change.new.size:,} B ({delta:+,})"


def _blob_links(store_url: str, change: FileChange) -> str:
    """Render download links for whichever blobs exist."""
    root = store_url.rstrip("/")
    links = []
    if change.old:
        links.append(f"[old]({root}/{change.old.sha256})")
    if change.new:
        links.append(f"[new]({root}/{change.new.sha256})")
    return " ".join(links)


def _utf8_len(text: str) -> int:
    """Return the UTF-8 byte length of ``text``, which is what GitHub's comment limit measures."""
    return len(text.encode("utf-8"))


def _counts(diff: CaseDiff) -> str:
    """Summarise a case's native changes as a short ``+a ~c -r`` string."""
    tally = {"added": 0, "changed": 0, "removed": 0}
    for change in diff.native:
        tally[change.status] += 1
    parts = [
        f"+{tally['added']}" if tally["added"] else "",
        f"~{tally['changed']}" if tally["changed"] else "",
        f"-{tally['removed']}" if tally["removed"] else "",
    ]
    return " ".join(p for p in parts if p) or "none"


def render_summary(diffs: list[CaseDiff]) -> str:
    """
    Render the one-row-per-case overview table.

    Every changed case appears here, whether or not its detail section survives the size cap,
    so nothing is silently invisible.
    """
    rows = ["| case | versions | native files |\n| --- | --- | --- |\n"]
    for diff in diffs:
        base_manifest, head_manifest = diff.base, diff.head
        if head_manifest is None:
            versions = "removed"
        elif base_manifest is None:
            versions = "new"
        else:
            versions = f"v{base_manifest.test_case_version} -> v{head_manifest.test_case_version}"
        rows.append(f"| `{diff.label}` | {versions} | {_counts(diff)} |\n")
    return "".join(rows)


def render_case(diff: CaseDiff, store_url: str) -> str:
    """Render one test case as a collapsible markdown section."""
    text_changes = [c for c in diff.native if c.is_text]
    binary_changes = [c for c in diff.native if not c.is_text]
    headline = f"{len(diff.native)} native file(s)"
    if diff.is_removed:
        headline = f"removed case, {headline}"
    elif diff.is_new:
        headline = f"new case, {headline}"

    # The blank line after </summary> is load-bearing: GitHub parses markdown inside an HTML
    # block only after one, so without it the first list renders as raw text with its backticks.
    out = [f"<details>\n<summary><b>{diff.label}</b> -- {headline}</summary>\n\n"]

    if diff.metadata:
        out.append("".join(f"- {line}\n" for line in diff.metadata))
    if diff.committed:
        out.append("\nCommitted artefacts changed:\n\n")
        out.append("".join(f"- {name}\n" for name in diff.committed))

    if binary_changes:
        out.append("\n| file | status | size | blobs |\n| --- | --- | --- | --- |\n")
        for change in binary_changes[:MAX_BINARY_ROWS]:
            out.append(
                f"| `{change.name}` | {change.status} | {_size_delta(change)} "
                f"| {_blob_links(store_url, change)} |\n"
            )
        if len(binary_changes) > MAX_BINARY_ROWS:
            elided = len(binary_changes) - MAX_BINARY_ROWS
            out.append(f"\n_{elided} further binary file(s) not listed._\n")

    for change in text_changes:
        out.append(f"\n**`{change.name}`** ({change.status}, {_size_delta(change)})")
        if change.diff:
            out.append(f"\n\n```diff\n{change.diff}\n```\n")
        else:
            out.append(f" -- {change.note}, {_blob_links(store_url, change)}\n")

    out.append("\n</details>\n")
    return "".join(out)


def render(diffs: list[CaseDiff], base: str, store_url: str, max_bytes: int | None = None) -> str:
    """
    Render the whole report.

    The summary table always covers every case. When ``max_bytes`` is set, detail sections are
    appended only while the encoded report stays under it and the remainder is named rather than
    expanded. Pass ``None`` for the full report, which is what the workflow artefact carries.

    The budget counts UTF-8 bytes rather than characters, because that is what GitHub's comment
    limit measures and a diff may carry non-ASCII content.
    """
    if not diffs:
        return f"### Regression baseline diff\n\nNo baseline manifests changed against `{base}`.\n"

    diffs = sorted(diffs, key=lambda d: d.label)
    total = sum(len(d.native) for d in diffs)
    header = (
        "### Regression baseline diff\n\n"
        f"{len(diffs)} test case(s) and {total} native file(s) changed against `{base}`. "
        "Text outputs are diffed inline. NetCDF and PNG outputs are listed with a size delta "
        "and a link to each blob.\n\n"
        f"{render_summary(diffs)}\n"
    )

    used = _utf8_len(header)
    body = ""
    for index, diff in enumerate(diffs):
        section = render_case(diff, store_url)
        if max_bytes is not None and used + _utf8_len(section) > max_bytes:
            remaining = ", ".join(f"`{d.label}`" for d in diffs[index:])
            body += (
                f"\n_Detail omitted to fit the comment size limit: {remaining}._\n"
                "\n_The full report is attached to the workflow run as the `mint-diff` artefact._\n"
            )
            break
        body += section
        used += _utf8_len(section)
    return header + body


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=f"origin/{os.environ.get('GITHUB_BASE_REF', 'main')}")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the full, uncapped report here",
    )
    parser.add_argument(
        "--comment-output",
        type=Path,
        default=None,
        help="write a copy capped to GitHub's comment size limit here",
    )
    parser.add_argument("--store-url", default=os.environ.get("REF_NATIVE_STORE_URL", DEFAULT_STORE_URL))
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args(argv)

    repo = open_repo()
    diffs = []
    for rel_path in changed_manifests(repo, args.base):
        diff = build_case_diff(repo, args.base, rel_path, args.store_url, fetch=not args.no_fetch)
        if diff is not None:
            diffs.append(diff)

    full = render(diffs, args.base, args.store_url)
    sys.stdout.write(full)
    if args.output:
        args.output.write_text(full, encoding="utf-8")
    if args.comment_output:
        capped = render(diffs, args.base, args.store_url, max_bytes=MAX_COMMENT_BYTES)
        args.comment_output.write_text(capped, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
