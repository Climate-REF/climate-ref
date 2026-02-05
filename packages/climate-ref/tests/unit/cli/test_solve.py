def test_solve_help(invoke_cli):
    result = invoke_cli(["solve", "--help"])

    assert "Solve for executions that require recalculation" in result.stdout


class TestSolve:
    def test_solve(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve"])

        assert mock_solve.call_count == 1
        _args, kwargs = mock_solve.call_args

        assert kwargs["timeout"] == 60
        assert not kwargs["dry_run"]
        assert kwargs["execute"]
        assert kwargs["filters"].diagnostic is None
        assert kwargs["filters"].provider is None

    def test_solve_with_timeout(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve", "--timeout", "10"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["timeout"] == 10

    def test_solve_with_dryrun(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve", "--dry-run"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["dry_run"]

    def test_solve_with_filters(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(
            [
                "solve",
                "--diagnostic",
                "global-mean-timeseries",
                "--provider",
                "esmvaltool",
                "--provider",
                "ilamb",
            ]
        )

        _args, kwargs = mock_solve.call_args
        assert kwargs["filters"].diagnostic == ["global-mean-timeseries"]
        assert kwargs["filters"].provider == ["esmvaltool", "ilamb"]
