"""
Integration test fixtures.

"""

import pytest

# Number of unique source_ids to retain per catalog entry.
# Enough to exercise all four providers while keeping solve time short.
_MAX_SOURCE_IDS = 5


@pytest.fixture(scope="session")
def esgf_data_catalog(esgf_solve_catalog, test_data_dir):
    """
    Trimmed ESGF catalog for integration tests.

    Keeps only the first ``_MAX_SOURCE_IDS`` unique source_ids per catalog
    DataFrame, reducing row count by ~95% while preserving representative
    coverage of experiments, variables, and members needed to produce
    executions for each provider.
    """
    if esgf_solve_catalog is None:
        expected_path = test_data_dir / "esgf-catalog"
        pytest.fail(
            f"ESGF parquet catalog not found in {expected_path}. "
            "Run scripts/generate_esgf_catalog.py to generate it."
        )

    result = {}
    for source_type, df in esgf_solve_catalog.items():
        if "source_id" in df.columns and df["source_id"].nunique() > _MAX_SOURCE_IDS:
            keep = df["source_id"].unique()[:_MAX_SOURCE_IDS]
            result[source_type] = df[df["source_id"].isin(keep)].reset_index(drop=True)
        else:
            result[source_type] = df

    return result
