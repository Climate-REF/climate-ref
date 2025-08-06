import platform
from collections.abc import Iterable

import pandas as pd
import pytest

from climate_ref.config import DiagnosticProviderConfig
from climate_ref.database import Database
from climate_ref.models import ExecutionGroup


def create_execution_dataframe(execution_groups: Iterable[ExecutionGroup]) -> pd.DataFrame:
    data = []

    for group in execution_groups:
        metadata = {
            "diagnostic": group.diagnostic.slug,
            "provider": group.diagnostic.provider.slug,
            "execution_id": group.id,
            "execution_key": group.key,
        }

        if group.executions:
            result = group.executions[-1]
            metadata["result_id"] = result.id
            metadata["successful"] = result.successful

        data.append(metadata)

        print(metadata)

    return pd.DataFrame(data)


@pytest.fixture
def config_cmip7_aft(config):
    """
    Overwrite the default test config to use the diagnostic providers for CMIP7 Assessment Fast Track
    """
    # Force the default diagnostic providers
    config.diagnostic_providers = [
        DiagnosticProviderConfig(provider=provider)
        for provider in ["climate_ref_esmvaltool", "climate_ref_ilamb", "climate_ref_pmp"]
    ]
    # Use the local executor to parallise the executions
    config.executor.executor = "climate_ref.executor.LocalExecutor"

    # Write the config to disk so it is used by the CLI
    # This overwrites the default config
    config.save()

    return config


@pytest.mark.slow
def test_solve_cmip7_aft(
    sample_data_dir,
    config_cmip7_aft,
    invoke_cli,
    monkeypatch,
):
    # Arm-based MacOS users will need to set the environment variable `MAMBA_PLATFORM=osx-64`
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        monkeypatch.setenv("MAMBA_PLATFORM", "osx-64")

    # The conda environments should already be created in the default location
    # See github CI integration test

    assert len(config_cmip7_aft.diagnostic_providers) == 3

    db = Database.from_config(config_cmip7_aft)

    invoke_cli(
        [
            "datasets",
            "fetch-data",
            "--registry",
            "pmp-climatology",
            "--output-directory",
            str(sample_data_dir / "pmp-climatology"),
        ]
    )

    # Ingest the sample data
    invoke_cli(["datasets", "ingest", "--source-type", "cmip6", str(sample_data_dir / "CMIP6")])
    invoke_cli(["datasets", "ingest", "--source-type", "obs4mips", str(sample_data_dir / "obs4REF")])
    invoke_cli(
        ["datasets", "ingest", "--source-type", "pmp-climatology", str(sample_data_dir / "pmp-climatology")]
    )

    # Solve
    # This will also create conda environments for the diagnostic providers
    # We always log the std out and stderr from the command as it is useful for debugging
    invoke_cli(["--verbose", "solve", "--one-per-diagnostic", "--timeout", f"{60 * 60}"], always_log=True)

    execution_groups = db.session.query(ExecutionGroup).all()
    df = create_execution_dataframe(execution_groups)

    print(df)

    # Check that all 3 diagnostic providers have been used
    assert set(df["provider"].unique()) == {"esmvaltool", "ilamb", "pmp"}

    # Check that some of the diagnostics have been marked successful
    assert df["successful"].any()
