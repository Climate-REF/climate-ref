import pytest
from climate_ref_pmp import provider as pmp_provider
from climate_ref_pmp.diagnostics.enso import ENSO
from climate_ref_pmp.pmp_driver import _get_resource


def test_invalid_diagnostic():
    with pytest.raises(ValueError, match="Unknown metrics collection: invalid_metrics"):
        ENSO(metrics_collection="invalid_metrics")


def test_build_cmds(solved_definition_factory):
    diagnostic = ENSO("ENSO_tel")
    diagnostic.provider = pmp_provider

    definition = solved_definition_factory(diagnostic)
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    cmd = diagnostic.build_cmd(definition)

    assert cmd == [
        "python",
        _get_resource("climate_ref_pmp.drivers", "enso_driver.py", use_resources=True),
        "--metrics_collection",
        "ENSO_tel",
        "--experiment_id",
        "historical",
        "--input_json_path",
        str(definition.output_directory / "input_ENSO_tel_ACCESS-ESM1-5_historical_r1i1p1f1.json"),
        "--output_directory",
        str(definition.output_directory),
    ]
