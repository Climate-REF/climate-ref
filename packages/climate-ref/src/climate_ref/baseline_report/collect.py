"""
Read the manifests either side of a base ref and pair up every native file that moved.

Collection performs no network access. Fetching blobs and building diffs is
:mod:`~climate_ref.baseline_report.analyse`'s job.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen
from loguru import logger

from climate_ref_core.regression.manifest import Manifest, NativeEntry

if TYPE_CHECKING:
    from git import Repo

# A manifest path needs a diagnostic and test-case directory before a label can be built from it.
_MIN_LABEL_PARTS = 3

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg"})
"""Extensions rendered as a two-up image comparison."""

TEXT_SUFFIXES = frozenset({".json", ".csv", ".yml", ".yaml", ".html", ".txt", ".md", ".log"})
"""Extensions whose blobs are worth fetching and diffing line by line."""

NETCDF_SUFFIXES = frozenset({".nc"})
"""Extensions reported as a size delta until the NetCDF analysis slice lands."""


class FileKind(enum.Enum):
    """How a native output file should be presented in the report."""

    IMAGE = "image"
    TEXT = "text"
    NETCDF = "netcdf"
    OTHER = "other"


def classify(name: str) -> FileKind:
    """
    Classify a native output file by its extension.

    Parameters
    ----------
    name
        The file's path relative to the test case's native output directory.

    Returns
    -------
    :
        The kind that decides how the file is rendered.
    """
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return FileKind.IMAGE
    if suffix in TEXT_SUFFIXES:
        return FileKind.TEXT
    if suffix in NETCDF_SUFFIXES:
        return FileKind.NETCDF
    return FileKind.OTHER


@frozen
class FileChange:
    """One native output file that was added, removed, or changed by the mint."""

    name: str
    """Path of the file relative to the test case's native output directory."""

    old: NativeEntry | None
    """The manifest entry on the base ref, or ``None`` when the file is new."""

    new: NativeEntry | None
    """The manifest entry on HEAD, or ``None`` when the file was removed."""

    kind: FileKind
    """How the file should be rendered."""

    @property
    def status(self) -> str:
        """``added``, ``removed`` or ``changed``."""
        if self.old is None:
            return "added"
        if self.new is None:
            return "removed"
        return "changed"


@frozen
class CaseChange:
    """Everything that changed for a single test case."""

    label: str
    """``provider/diagnostic/test-case``."""

    slug: str
    """The output subdirectory for this case's page. Equal to :attr:`label`."""

    rel_path: str
    """Repo-relative path of the case's ``manifest.json``."""

    base: Manifest | None
    """The manifest on the base ref, or ``None`` when the case is new."""

    head: Manifest | None
    """The manifest on HEAD, or ``None`` when the case was removed."""

    files: tuple[FileChange, ...]
    """Every native file that moved, in name order."""

    committed: tuple[str, ...]
    """Committed artefacts whose digest moved, described one per entry."""

    metadata: tuple[str, ...]
    """Scalar manifest fields that moved, described one per entry."""

    @property
    def is_new(self) -> bool:
        """Whether the whole test case is new on this branch."""
        return self.base is None

    @property
    def is_removed(self) -> bool:
        """Whether the whole test case was deleted on this branch."""
        return self.head is None


@frozen
class Report:
    """Every test case whose baseline moved on this branch."""

    base_ref: str
    """The git ref HEAD was compared against."""

    head_sha: str
    """The full sha of the commit that was compared."""

    cases: tuple[CaseChange, ...]
    """The changed cases, in label order."""


def changed_manifests(repo: Repo, base: str) -> list[str]:
    """
    Return the repo-relative paths of every test-case manifest that differs from ``base``.

    Uses the merge-base (``base...HEAD``) so commits landing on the base branch after the
    feature branch forked are not misreported as baseline changes.

    Parameters
    ----------
    repo
        The repository to diff in.
    base
        The git ref to compare against.

    Returns
    -------
    :
        The manifest paths, sorted.
    """
    from git import GitCommandError  # noqa: PLC0415 - keeps the import cost off the CLI startup path

    pathspec = ":(glob)packages/**/test-data/**/manifest.json"
    try:
        out = repo.git.diff("--name-only", f"{base}...HEAD", "--", pathspec)
    except GitCommandError:
        # A shallow clone may have no merge-base with the base ref. A two-dot diff over-reports
        # (it also shows base-branch commits), which is the safe direction for a report.
        out = repo.git.diff("--name-only", base, "HEAD", "--", pathspec)
    return sorted(line for line in out.splitlines() if line.strip())


def load_at_ref(repo: Repo, ref: str, rel_path: str) -> Manifest | None:
    """
    Load a manifest as it exists at ``ref``, or ``None`` when absent there.

    Parameters
    ----------
    repo
        The repository to read from.
    ref
        The git ref to read the manifest at.
    rel_path
        Repo-relative path of the manifest.

    Returns
    -------
    :
        The parsed manifest, or ``None`` when the path does not exist at ``ref``.
    """
    from git import GitCommandError  # noqa: PLC0415 - keeps the import cost off the CLI startup path

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

    Parameters
    ----------
    rel_path
        Repo-relative path of the manifest.

    Returns
    -------
    :
        The label.
    """
    parts = Path(rel_path).parts
    provider = parts[1].removeprefix("climate-ref-") if len(parts) > 1 else "?"
    tail = parts[-3:-1] if len(parts) >= _MIN_LABEL_PARTS else ()
    return "/".join((provider, *tail))


def _metadata_changes(base: Manifest | None, head: Manifest | None) -> tuple[str, ...]:
    """
    Describe the scalar manifest fields that moved.

    Parameters
    ----------
    base
        The manifest on the base ref, or ``None``.
    head
        The manifest on HEAD, or ``None``.

    Returns
    -------
    :
        One description per changed field.
    """
    if head is None:
        return ("test case removed",)
    if base is None:
        return (f"new test case at test_case_version {head.test_case_version}",)
    changes = []
    for name in ("test_case_version", "diagnostic_version", "catalog_hash", "schema"):
        old, new = getattr(base, name), getattr(head, name)
        if old != new:
            changes.append(f"{name}: {old} -> {new}")
    return tuple(changes)


def _committed_changes(base: Manifest | None, head: Manifest | None) -> tuple[str, ...]:
    """
    Name the committed regression artefacts whose digest moved.

    Parameters
    ----------
    base
        The manifest on the base ref, or ``None``.
    head
        The manifest on HEAD, or ``None``.

    Returns
    -------
    :
        One description per changed artefact, in name order.
    """
    old = base.committed if base else {}
    new = head.committed if head else {}
    out = []
    for name in sorted(set(old) | set(new)):
        if old.get(name) == new.get(name):
            continue
        if name not in old:
            out.append(f"{name} (added)")
        elif name not in new:
            out.append(f"{name} (removed)")
        else:
            out.append(name)
    return tuple(out)


def build_case_change(repo: Repo, base: str, rel_path: str) -> CaseChange | None:
    """
    Collect every change to one test case.

    A case deleted on this branch has no head manifest. It is still reported, as a removal,
    because dropping a test case is a baseline change a reviewer needs to see.

    Parameters
    ----------
    repo
        The repository to read from.
    base
        The git ref to compare against.
    rel_path
        Repo-relative path of the case's manifest.

    Returns
    -------
    :
        The case's changes, or ``None`` when the manifest is absent from both sides
        or cannot be parsed, both of which leave nothing to say.
    """
    head_path = Path(repo.working_tree_dir or ".") / rel_path
    # One unreadable manifest should cost its own case, not the whole report.
    try:
        head = Manifest.load(head_path) if head_path.exists() else None
        base_manifest = load_at_ref(repo, base, rel_path)
    except ValueError as exc:
        logger.warning(f"Skipping {rel_path}: {exc}")
        return None
    if head is None and base_manifest is None:
        return None

    old_native = base_manifest.native if base_manifest else {}
    new_native = head.native if head else {}
    files = []
    for name in sorted(set(old_native) | set(new_native)):
        old, new = old_native.get(name), new_native.get(name)
        if old is not None and new is not None and old.sha256 == new.sha256:
            continue
        files.append(FileChange(name=name, old=old, new=new, kind=classify(name)))

    label = case_label(rel_path)
    return CaseChange(
        label=label,
        slug=label,
        rel_path=rel_path,
        base=base_manifest,
        head=head,
        files=tuple(files),
        committed=_committed_changes(base_manifest, head),
        metadata=_metadata_changes(base_manifest, head),
    )


def collect(repo: Repo, base: str) -> Report:
    """
    Collect every baseline change on this branch.

    Parameters
    ----------
    repo
        The repository to read from.
    base
        The git ref to compare against.

    Returns
    -------
    :
        The report, with cases in label order.
    """
    cases = [
        case
        for rel_path in changed_manifests(repo, base)
        if (case := build_case_change(repo, base, rel_path)) is not None
    ]
    return Report(
        base_ref=base,
        head_sha=repo.head.commit.hexsha,
        cases=tuple(sorted(cases, key=lambda case: case.label)),
    )
