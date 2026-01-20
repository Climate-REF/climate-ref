"""Git utilities for CLI commands."""

from pathlib import Path
from typing import Any

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


def get_git_status(file_path: Path, repo: Repo) -> str:
    """
    Get git status for a file using GitPython.

    Parameters
    ----------
    file_path
        Absolute path to the file
    repo
        GitPython Repo object

    Returns
    -------
    :
        Status string: "new", "staged", "modified", "tracked", "untracked", or "unknown"
    """
    try:
        rel_path = str(file_path.relative_to(repo.working_dir))

        # Check if untracked
        if rel_path in repo.untracked_files:
            return "new"

        # Check staged changes (index vs HEAD)
        staged_files = {item.a_path for item in repo.index.diff("HEAD")}
        if rel_path in staged_files:
            return "staged"

        # Check unstaged changes (working tree vs index)
        unstaged_files = {item.a_path for item in repo.index.diff(None)}
        if rel_path in unstaged_files:
            return "modified"

        # Check if file is tracked
        try:
            repo.git.ls_files("--error-unmatch", rel_path)
            return "tracked"
        except Exception:
            return "untracked"
    except Exception:
        return "unknown"


def collect_regression_file_info(
    regression_dir: Path,
    repo: Repo | None,
    size_threshold_bytes: int,
) -> list[dict[str, Any]]:
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

    file_info: list[dict[str, Any]] = []
    for file_path in files:
        size = file_path.stat().st_size
        rel_path = str(file_path.relative_to(regression_dir))
        git_status = get_git_status(file_path, repo) if repo else "unknown"

        file_info.append(
            {
                "rel_path": rel_path,
                "size": size,
                "is_large": size > size_threshold_bytes,
                "git_status": git_status,
            }
        )

    return file_info
