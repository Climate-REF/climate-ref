"""Test that the recipes are updated correctly."""

from pathlib import Path

import pandas as pd
import pytest
from climate_ref_esmvaltool import provider
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from pytest_regressions.file_regression import FileRegressionFixture

from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType


@pytest.mark.parametrize(
    "diagnostic", [pytest.param(diagnostic, id=diagnostic.slug) for diagnostic in provider.diagnostics()]
)
def test_write_recipe(
    tmp_path: Path,
    file_regression: FileRegressionFixture,
    esgf_data_catalog_trimmed: dict[SourceDatasetType, pd.DataFrame],
    diagnostic: ESMValToolDiagnostic,
) -> None:
    """Test that the recipes are updated correctly."""

    # Select the first ensemble member as it has the most data
    cmip6_catalog = esgf_data_catalog_trimmed[SourceDatasetType.CMIP6]
    cmip6_catalog = cmip6_catalog[cmip6_catalog["variant_label"] == "r1i1p1f1"]
    esgf_data_catalog_trimmed[SourceDatasetType.CMIP6] = cmip6_catalog

    def get_source_types(execution):
        return tuple(sorted(k.value for k in execution.datasets.keys()))

    seen = set()
    for execution in solve_executions(
        data_catalog=esgf_data_catalog_trimmed,
        diagnostic=diagnostic,
        provider=diagnostic.provider,
    ):
        if (source_types := get_source_types(execution)) not in seen:
            seen.add(source_types)
            definition = execution.build_execution_definition(output_root=tmp_path)
            definition.output_directory.mkdir(parents=True, exist_ok=True)
            tmp_recipe = diagnostic.write_recipe(definition=definition)
            reference_recipe = (
                Path(__file__).parent
                / "recipes"
                / f"recipe-{diagnostic.slug}-{'-'.join(source_types)}.yml".replace("-", "_")
            )
            encoding = "utf-8"
            file_regression.check(
                tmp_recipe.read_text(encoding),
                encoding=encoding,
                fullpath=reference_recipe,
            )
