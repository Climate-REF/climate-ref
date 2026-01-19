from unittest.mock import MagicMock, patch

from git import InvalidGitRepositoryError

from climate_ref.cli._git_utils import collect_regression_file_info, get_git_status, get_repo_for_path


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


class TestGetGitStatus:
    """Tests for get_git_status function."""

    def test_returns_new_for_untracked_file(self, tmp_path):
        """Test returns 'new' for untracked files."""
        mock_repo = MagicMock()
        mock_repo.working_dir = str(tmp_path)
        mock_repo.untracked_files = ["test_file.txt"]

        file_path = tmp_path / "test_file.txt"

        result = get_git_status(file_path, mock_repo)
        assert result == "new"

    def test_returns_staged_for_staged_changes(self, tmp_path):
        """Test returns 'staged' for files in the index diff with HEAD."""
        mock_repo = MagicMock()
        mock_repo.working_dir = str(tmp_path)
        mock_repo.untracked_files = []

        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "staged_file.txt"
        mock_repo.index.diff.return_value = [mock_diff_item]

        file_path = tmp_path / "staged_file.txt"

        result = get_git_status(file_path, mock_repo)
        assert result == "staged"

    def test_returns_modified_for_unstaged_changes(self, tmp_path):
        """Test returns 'modified' for unstaged changes."""
        mock_repo = MagicMock()
        mock_repo.working_dir = str(tmp_path)
        mock_repo.untracked_files = []

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

        file_path = tmp_path / "modified_file.txt"

        result = get_git_status(file_path, mock_repo)
        assert result == "modified"

    def test_returns_tracked_for_clean_tracked_file(self, tmp_path):
        """Test returns 'tracked' for clean tracked files."""
        mock_repo = MagicMock()
        mock_repo.working_dir = str(tmp_path)
        mock_repo.untracked_files = []
        mock_repo.index.diff.return_value = []  # No changes

        file_path = tmp_path / "tracked_file.txt"

        result = get_git_status(file_path, mock_repo)
        assert result == "tracked"

    def test_returns_untracked_when_ls_files_fails(self, tmp_path):
        """Test returns 'untracked' when ls-files raises exception."""
        mock_repo = MagicMock()
        mock_repo.working_dir = str(tmp_path)
        mock_repo.untracked_files = []
        mock_repo.index.diff.return_value = []
        mock_repo.git.ls_files.side_effect = Exception("File not tracked")

        file_path = tmp_path / "untracked_file.txt"

        result = get_git_status(file_path, mock_repo)
        assert result == "untracked"

    def test_returns_unknown_when_relative_path_fails(self, tmp_path):
        """Test returns 'unknown' when file path cannot be made relative."""
        mock_repo = MagicMock()
        mock_repo.working_dir = "/completely/different/path"

        file_path = tmp_path / "some_file.txt"

        result = get_git_status(file_path, mock_repo)
        assert result == "unknown"

    def test_handles_nested_directory_path(self, tmp_path):
        """Test correctly handles files in nested directories."""
        mock_repo = MagicMock()
        mock_repo.working_dir = str(tmp_path)
        mock_repo.untracked_files = ["subdir/nested/file.txt"]

        file_path = tmp_path / "subdir" / "nested" / "file.txt"

        result = get_git_status(file_path, mock_repo)
        assert result == "new"


class TestCollectRegressionFileInfo:
    """Tests for collect_regression_file_info function."""

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        """Test returns empty list when directory has no files."""
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()

        result = collect_regression_file_info(regression_dir, None, 1024)
        assert result == []

    def test_collects_file_info_without_repo(self, tmp_path):
        """Test collects file info when no repo is provided."""
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()

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

    def test_collects_file_info_with_repo(self, tmp_path):
        """Test collects file info with git status when repo provided."""
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()

        (regression_dir / "file.txt").write_text("content")

        mock_repo = MagicMock()
        mock_repo.working_dir = str(tmp_path)
        mock_repo.untracked_files = ["regression/file.txt"]

        result = collect_regression_file_info(regression_dir, mock_repo, 1000)

        assert len(result) == 1
        assert result[0]["rel_path"] == "file.txt"
        assert result[0]["git_status"] == "new"

    def test_handles_nested_files(self, tmp_path):
        """Test handles files in nested directories."""
        regression_dir = tmp_path / "regression"
        subdir = regression_dir / "subdir"
        subdir.mkdir(parents=True)

        (subdir / "nested.txt").write_text("nested content")

        result = collect_regression_file_info(regression_dir, None, 1000)

        assert len(result) == 1
        assert result[0]["rel_path"] == "subdir/nested.txt"

    def test_excludes_directories(self, tmp_path):
        """Test excludes directories from results."""
        regression_dir = tmp_path / "regression"
        subdir = regression_dir / "subdir"
        subdir.mkdir(parents=True)

        (regression_dir / "file.txt").write_text("content")

        result = collect_regression_file_info(regression_dir, None, 1000)

        assert len(result) == 1
        assert result[0]["rel_path"] == "file.txt"

    def test_size_threshold_boundary(self, tmp_path):
        """Test size threshold at exact boundary."""
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()

        (regression_dir / "exact.txt").write_text("x" * 100)

        # At threshold - should not be large
        result = collect_regression_file_info(regression_dir, None, 100)
        assert result[0]["is_large"] is False

        # Below threshold - should be large
        result = collect_regression_file_info(regression_dir, None, 99)
        assert result[0]["is_large"] is True
