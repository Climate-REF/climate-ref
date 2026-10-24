import pytest

from climate_ref.solver import SolveOutcome


def test_solve_help(invoke_cli):
    result = invoke_cli(["solve", "--help"])

    assert "Solve for executions that require recalculation" in result.stdout


@pytest.fixture
def mock_solve(mocker):
    return mocker.patch("climate_ref.solver.solve_required_executions", return_value=SolveOutcome())


class TestSolve:
    def test_solve_exits_non_zero_for_unhealthy_providers(self, sample_data_dir, db, invoke_cli, mock_solve):
        mock_solve.return_value = SolveOutcome(unhealthy_providers=["example"])

        invoke_cli(["solve"], expected_exit_code=1)

        assert mock_solve.call_args.kwargs["check_health"] is True

    def test_solve_no_check_health(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve", "--no-check-health"])

        assert mock_solve.call_args.kwargs["check_health"] is False

    def test_solve(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve"])

        assert mock_solve.call_count == 1
        _args, kwargs = mock_solve.call_args

        assert kwargs["timeout"] == 6 * 60 * 60
        assert kwargs["wait"] is True
        assert not kwargs["dry_run"]
        assert kwargs["execute"]
        assert kwargs["filters"].diagnostic is None
        assert kwargs["filters"].provider is None

    @pytest.mark.parametrize(("args", "expected"), [([], 1), (["--dry-run"], 0)])
    def test_solve_refreshes_statistics(
        self, sample_data_dir, db, invoke_cli, mock_solve, mocker, args, expected
    ):
        optimize = mocker.patch("climate_ref.database.Database.optimize")

        invoke_cli(["solve", *args])

        assert optimize.call_count == expected

    def test_solve_with_timeout(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve", "--timeout", "10"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["timeout"] == 10

    def test_solve_with_zero_timeout_still_waits(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve", "--timeout", "0"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["timeout"] == 0
        assert kwargs["wait"] is True

    def test_solve_with_dryrun(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve", "--dry-run"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["dry_run"]

    def test_solve_with_filters(self, sample_data_dir, db, invoke_cli, mock_solve, config):
        from climate_ref.config import DiagnosticProviderConfig

        # The provider filter is validated against configured providers,
        # so the filtered providers must be configured for this parsing test.
        config.diagnostic_providers = [
            DiagnosticProviderConfig(provider="climate_ref_esmvaltool"),
            DiagnosticProviderConfig(provider="climate_ref_ilamb"),
        ]
        config.save()

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

    def test_solve_with_exclude_diagnostic(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(
            [
                "solve",
                "--exclude-diagnostic",
                "ilamb/thetao-woa2023-surface",
                "--exclude-diagnostic",
                "annual-cycle",
            ]
        )

        _args, kwargs = mock_solve.call_args
        assert kwargs["filters"].exclude_diagnostic == ["ilamb/thetao-woa2023-surface", "annual-cycle"]

    def test_solve_with_mixed_case_provider_filter(self, sample_data_dir, db, invoke_cli, mock_solve, config):
        from climate_ref.config import DiagnosticProviderConfig

        # Provider matching is a case-insensitive substring match against the slug,
        # so a mixed-case partial filter must validate and be passed through verbatim.
        config.diagnostic_providers = [
            DiagnosticProviderConfig(provider="climate_ref_esmvaltool"),
        ]
        config.save()

        invoke_cli(["solve", "--provider", "ESMval"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["filters"].provider == ["ESMval"]

    def test_solve_provider_filter_no_match_fails(self, sample_data_dir, db, invoke_cli, mock_solve):
        # A --provider filter that matches no configured provider must fail loudly
        # rather than exiting 0 having done nothing.
        result = invoke_cli(["solve", "--provider", "does-not-exist"], expected_exit_code=1)

        assert "No configured providers match" in result.stderr
        mock_solve.assert_not_called()

    def test_solve_no_providers_configured_fails(self, sample_data_dir, db, invoke_cli, mock_solve, config):
        config.diagnostic_providers = []
        config.save()

        result = invoke_cli(["solve"], expected_exit_code=1)

        assert "No diagnostic providers are configured" in result.stderr
        mock_solve.assert_not_called()

    def test_solve_with_dataset_filter(self, sample_data_dir, db, invoke_cli, mock_solve):
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
        self, sample_data_dir, db, invoke_cli, mock_solve
    ):
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
        result = invoke_cli(
            ["solve", "--dataset-filter", "missing_equals"],
            expected_exit_code=2,
        )

        assert "Invalid filter format" in result.stderr
        assert "Expected format: 'key=value'" in result.stderr

    def test_solve_with_limit(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve", "--limit", "5"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["limit"] == 5

    def test_solve_without_limit(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["limit"] is None

    def test_solve_with_rerun_failed(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve", "--rerun-failed"])

        _args, kwargs = mock_solve.call_args
        assert kwargs["rerun_failed"] is True

    def test_solve_with_no_wait(self, sample_data_dir, db, invoke_cli, mock_solve):
        invoke_cli(["solve", "--no-wait", "--timeout", "30"])

        # --no-wait is now passed through explicitly rather than being collapsed
        # onto timeout=0, so --timeout keeps its literal value.
        _args, kwargs = mock_solve.call_args
        assert kwargs["wait"] is False
        assert kwargs["timeout"] == 30


@pytest.mark.parametrize(
    ("outcome", "expected_exit_code"),
    [
        (SolveOutcome(), 0),
        (SolveOutcome(failed_executions=2), 1),
    ],
)
def test_solve_exit_code(sample_data_dir, db, invoke_cli, mock_solve, outcome, expected_exit_code):
    mock_solve.return_value = outcome

    invoke_cli(["solve"], expected_exit_code=expected_exit_code)
