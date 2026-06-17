def test_solve_help(invoke_cli):
    result = invoke_cli(["solve", "--help"])

    assert "Solve for executions that require recalculation" in result.stdout


class TestSolve:
    def test_solve(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve"])

        assert mock_solve.call_count == 1
        _args, kwargs = mock_solve.call_args

        assert kwargs["timeout"] == 6 * 60 * 60
        assert kwargs["wait"] is True
        assert not kwargs["dry_run"]
        assert kwargs["execute"]
        assert kwargs["filters"].diagnostic is None
        assert kwargs["filters"].provider is None

    def test_solve_with_timeout(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve", "--timeout", "10"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["timeout"] == 10

    def test_solve_with_zero_timeout_still_waits(self, sample_data_dir, db, invoke_cli, mocker):
        # Regression: --timeout 0 must mean "wait with no time limit", not "queue and exit".
        # It is no longer collapsed onto the --no-wait sentinel.
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve", "--timeout", "0"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["timeout"] == 0
        assert kwargs["wait"] is True

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

    def test_solve_with_dataset_filter(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(
            [
                "solve",
                "--dataset-filter",
                "source_id=ACCESS-CM2",
                "--dataset-filter",
                "variable_id=tas",
            ]
        )

        _args, kwargs = mock_solve.call_args
        assert kwargs["filters"].dataset == {
            "source_id": ["ACCESS-CM2"],
            "variable_id": ["tas"],
        }

    def test_solve_with_dataset_filter_multiple_values_same_key(
        self, sample_data_dir, db, invoke_cli, mocker
    ):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(
            [
                "solve",
                "--dataset-filter",
                "source_id=ACCESS-CM2",
                "--dataset-filter",
                "source_id=CESM2",
            ]
        )

        _args, kwargs = mock_solve.call_args
        assert kwargs["filters"].dataset == {
            "source_id": ["ACCESS-CM2", "CESM2"],
        }

    def test_solve_with_invalid_dataset_filter(self, sample_data_dir, db, invoke_cli, mocker):
        mocker.patch("climate_ref.solver.solve_required_executions")
        result = invoke_cli(
            ["solve", "--dataset-filter", "missing_equals"],
            expected_exit_code=2,
        )

        assert "Invalid filter format" in result.stderr
        assert "Expected format: 'key=value'" in result.stderr

    def test_solve_with_limit(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve", "--limit", "5"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["limit"] == 5

    def test_solve_without_limit(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["limit"] is None

    def test_solve_with_rerun_failed(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve", "--rerun-failed"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["rerun_failed"] is True

    def test_solve_with_no_wait(self, sample_data_dir, db, invoke_cli, mocker):
        mock_solve = mocker.patch("climate_ref.solver.solve_required_executions")
        invoke_cli(["solve", "--no-wait", "--timeout", "30"])

        # --no-wait is now passed through explicitly rather than being collapsed
        # onto timeout=0, so --timeout keeps its literal value.
        _args, kwargs = mock_solve.call_args
        assert kwargs["wait"] is False
        assert kwargs["timeout"] == 30
