from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_mock
from git import InvalidGitRepositoryError, Repo

from climate_ref.cli._git_utils import collect_regression_file_info, get_repo_for_path


class TestGetRepoForPath:
    """Tests for get_repo_for_path function."""

    def test_returns_none_for_non_repo_path(self, tmp_path):
        """Test returns None when path is not in a git repository."""
        # tmp_path is typically not a git repo
        # We need to mock to ensure consistent behavior
        with patch("climate_ref.cli._git_utils.Repo") as mock_repo_cls:
            mock_repo_cls.side_effect = InvalidGitRepositoryError("not a repo")
            result = get_repo_for_path(tmp_path)
            assert result is None

    def test_returns_repo_for_valid_git_path(self, tmp_path):
        """Test returns Repo when path is in a git repository."""
        mock_repo = MagicMock()
        with patch("climate_ref.cli._git_utils.Repo", return_value=mock_repo) as mock_repo_cls:
            result = get_repo_for_path(tmp_path)
            assert result is mock_repo
            mock_repo_cls.assert_called_once_with(tmp_path, search_parent_directories=True)


class TestCollectRegressionFileInfo:
    """Tests for collect_regression_file_info function."""

    @pytest.fixture
    def regression_dir(self, tmp_path: Path) -> Path:
        path = tmp_path / "regression"
        path.mkdir()
        return path

    @pytest.fixture
    def mock_repo(
        self,
        mocker: pytest_mock.MockerFixture,
        regression_dir: Path,
    ) -> pytest_mock.MockType:
        repo = mocker.create_autospec(Repo, instance=True)
        repo.working_dir = str(regression_dir)
        repo.untracked_files = []
        repo.index.diff.return_value = []
        return repo

    def test_returns_new_for_untracked_file(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test returns 'new' for untracked files."""
        mock_repo.untracked_files = ["test_file.txt"]

        file_path = regression_dir / "test_file.txt"
        file_path.touch()

        result = collect_regression_file_info(regression_dir, mock_repo, 1024)
        assert result == [
            {
                "git_status": "new",
                "is_large": False,
                "rel_path": file_path.name,
                "size": 0,
            }
        ]

    def test_returns_staged_for_staged_changes(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test returns 'staged' for files in the index diff with HEAD."""
        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "staged_file.txt"
        mock_repo.index.diff.return_value = [mock_diff_item]

        file_path = regression_dir / "staged_file.txt"
        file_path.touch()

        result = collect_regression_file_info(regression_dir, mock_repo, 1024)
        assert result == [
            {
                "git_status": "staged",
                "is_large": False,
                "rel_path": file_path.name,
                "size": 0,
            }
        ]

    def test_returns_modified_for_unstaged_changes(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test returns 'modified' for unstaged changes."""

        # No staged changes
        def diff_side_effect(arg):
            if arg == "HEAD":
                return []  # No staged changes
            else:
                # Unstaged changes (working tree vs index)
                mock_diff_item = MagicMock()
                mock_diff_item.a_path = "modified_file.txt"
                return [mock_diff_item]

        mock_repo.index.diff.side_effect = diff_side_effect

        file_path = regression_dir / "modified_file.txt"
        file_path.touch()

        result = collect_regression_file_info(regression_dir, mock_repo, 1024)
        assert result == [
            {
                "git_status": "modified",
                "is_large": False,
                "rel_path": file_path.name,
                "size": 0,
            }
        ]

    def test_returns_tracked_for_clean_tracked_file(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test returns 'tracked' for clean tracked files."""
        file_path = regression_dir / "tracked_file.txt"
        file_path.touch()

        result = collect_regression_file_info(regression_dir, mock_repo, 1024)
        assert result == [
            {
                "git_status": "tracked",
                "is_large": False,
                "rel_path": file_path.name,
                "size": 0,
            }
        ]

    def test_returns_untracked_when_ls_files_fails(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test returns 'untracked' when ls-files raises exception."""
        mock_repo.git.ls_files.side_effect = Exception("File not tracked")

        file_path = regression_dir / "untracked_file.txt"
        file_path.touch()

        result = collect_regression_file_info(regression_dir, mock_repo, 1024)
        assert result == [
            {
                "git_status": "untracked",
                "is_large": False,
                "rel_path": file_path.name,
                "size": 0,
            }
        ]

    def test_returns_unknown_when_relative_path_fails(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test returns 'unknown' when file path cannot be made relative."""
        mock_repo.working_dir = "/completely/different/path"

        file_path = regression_dir / "some_file.txt"
        file_path.touch()

        result = collect_regression_file_info(regression_dir, mock_repo, 1024)
        assert result == [
            {
                "git_status": "unknown",
                "is_large": False,
                "rel_path": file_path.name,
                "size": 0,
            }
        ]

    def test_handles_nested_directory_path(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test correctly handles files in nested directories."""
        mock_repo.untracked_files = ["subdir/nested/file.txt"]

        file_path = regression_dir / "subdir" / "nested" / "file.txt"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = collect_regression_file_info(regression_dir, mock_repo, 1024)
        assert result == [
            {
                "git_status": "new",
                "is_large": False,
                "rel_path": file_path.relative_to(regression_dir).as_posix(),
                "size": 0,
            }
        ]

    def test_returns_empty_list_for_empty_directory(self, regression_dir: Path) -> None:
        """Test returns empty list when directory has no files."""
        result = collect_regression_file_info(regression_dir, None, 1024)
        assert result == []

    def test_collects_file_info_without_repo(self, regression_dir: Path) -> None:
        """Test collects file info when no repo is provided."""
        # Create test files
        (regression_dir / "small.txt").write_text("small")
        (regression_dir / "large.txt").write_text("x" * 2000)

        result = collect_regression_file_info(regression_dir, None, 1000)

        assert len(result) == 2
        # Results are sorted by path
        small_file = next(f for f in result if f["rel_path"] == "small.txt")
        large_file = next(f for f in result if f["rel_path"] == "large.txt")

        assert small_file["size"] == 5
        assert small_file["is_large"] is False
        assert small_file["git_status"] == "unknown"

        assert large_file["size"] == 2000
        assert large_file["is_large"] is True
        assert large_file["git_status"] == "unknown"

    def test_collects_file_info_with_repo(
        self,
        regression_dir: Path,
        mock_repo: pytest_mock.MockType,
    ) -> None:
        """Test collects file info with git status when repo provided."""
        (regression_dir / "file.txt").write_text("content")
        mock_repo.untracked_files = ["file.txt"]

        result = collect_regression_file_info(regression_dir, mock_repo, 1000)

        assert len(result) == 1
        assert result[0]["rel_path"] == "file.txt"
        assert result[0]["git_status"] == "new"

    def test_handles_nested_files(self, regression_dir: Path) -> None:
        """Test handles files in nested directories."""
        subdir = regression_dir / "subdir"
        subdir.mkdir(parents=True)

        (subdir / "nested.txt").write_text("nested content")

        result = collect_regression_file_info(regression_dir, None, 1000)

        assert len(result) == 1
        assert result[0]["rel_path"] == "subdir/nested.txt"

    def test_excludes_directories(self, regression_dir: Path) -> None:
        """Test excludes directories from results."""
        subdir = regression_dir / "subdir"
        subdir.mkdir(parents=True)

        (regression_dir / "file.txt").write_text("content")

        result = collect_regression_file_info(regression_dir, None, 1000)

        assert len(result) == 1
        assert result[0]["rel_path"] == "file.txt"

    def test_size_threshold_boundary(self, regression_dir: Path) -> None:
        """Test size threshold at exact boundary."""
        (regression_dir / "exact.txt").write_text("x" * 100)

        # At threshold - should not be large
        result = collect_regression_file_info(regression_dir, None, 100)
        assert result[0]["is_large"] is False

        # Below threshold - should be large
        result = collect_regression_file_info(regression_dir, None, 99)
        assert result[0]["is_large"] is True
