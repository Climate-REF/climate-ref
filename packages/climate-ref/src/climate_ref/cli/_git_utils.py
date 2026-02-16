"""Git utilities for CLI commands."""

from pathlib import Path
from typing import TypedDict

from git import InvalidGitRepositoryError, Repo


def get_repo_for_path(path: Path) -> Repo | None:
    """
    Get the git repository containing the given path.

    Parameters
    ----------
    path
        Path to a file or directory

    Returns
    -------
    :
        The Repo object if path is within a git repository, None otherwise
    """
    try:
        return Repo(path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        return None


class FileInfo(TypedDict):
    """Information about a file in the regression directory."""

    rel_path: str
    size: int
    is_large: bool
    git_status: str


def collect_regression_file_info(
    regression_dir: Path,
    repo: Repo | None,
    size_threshold_bytes: int,
) -> list[FileInfo]:
    """
    Collect file information from a regression directory.

    Parameters
    ----------
    regression_dir
        Path to the regression data directory
    repo
        Git repository object, or None if not in a repo
    size_threshold_bytes
        Files larger than this will be flagged as large

    Returns
    -------
    :
        List of dicts with keys: rel_path, size, is_large, git_status
    """
    files = sorted(regression_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    file_info: list[FileInfo] = []
    for file_path in files:
        size = file_path.stat().st_size
        rel_path = str(file_path.relative_to(regression_dir))
        file_info.append(
            {
                "rel_path": rel_path,
                "size": size,
                "is_large": size > size_threshold_bytes,
                "git_status": "unknown",
            }
        )
    if not repo or repo.working_dir != str(regression_dir):
        return file_info

    # Read the current status of the repository just once.
    untracked_files = set(repo.untracked_files)
    staged_files = {item.a_path for item in repo.index.diff("HEAD")}
    unstaged_files = {item.a_path for item in repo.index.diff(None)}

    # Update the git status of each file.
    for item in file_info:
        rel_path = item["rel_path"]

        # Check if untracked
        if rel_path in untracked_files:
            item["git_status"] = "new"
            continue

        # Check staged changes (index vs HEAD)
        if rel_path in staged_files:
            item["git_status"] = "staged"
            continue

        # Check unstaged changes (working tree vs index)
        if rel_path in unstaged_files:
            item["git_status"] = "modified"
            continue

        # Check if file is tracked
        try:
            repo.git.ls_files("--error-unmatch", rel_path)
            item["git_status"] = "tracked"
        except Exception:
            item["git_status"] = "untracked"

    return file_info
