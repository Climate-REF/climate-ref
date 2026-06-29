"""Tests for the documentation linter (``scripts/lint_docs.py``)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
_LINT_DOCS_PATH = REPO_ROOT / "scripts" / "lint_docs.py"


def _load_lint_docs():
    spec = importlib.util.spec_from_file_location("lint_docs", _LINT_DOCS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection can resolve the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lint_docs = _load_lint_docs()


@pytest.fixture(scope="module")
def cli_options():
    options = lint_docs._build_cli_options()
    if options is None:  # pragma: no cover - the CLI is always importable in tests
        pytest.skip("ref CLI not importable")
    return options


def test_malformed_flag_detected_without_cli():
    # The triple-dash check does not require the CLI to be importable.
    errors = lint_docs._check_ref_command("ref datasets fetch-data ---output-directory data", options=None)
    assert any("malformed flag" in e for e in errors)


def test_split_flag_detected(cli_options):
    # '--output directory' splits the real '--output-directory' flag in two.
    errors = lint_docs._check_ref_command(
        "ref datasets fetch-data --registry pmp-climatology --output directory data",
        cli_options,
    )
    assert any("unknown flag '--output'" in e for e in errors)


def test_valid_command_passes(cli_options):
    errors = lint_docs._check_ref_command(
        "ref datasets fetch-data --registry sample-data --output-directory data",
        cli_options,
    )
    assert errors == []


def test_global_flag_before_subcommand_passes(cli_options):
    # A global option ahead of the subcommand must not cause false positives.
    errors = lint_docs._check_ref_command(
        "ref --verbose test-cases run --provider my-provider --diagnostic my-diagnostic",
        cli_options,
    )
    assert errors == []


def test_broken_blob_link_detected(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "See [the script](https://github.com/Climate-REF/climate-ref/blob/main/scripts/does-not-exist.py).\n"
    )
    problems = lint_docs._lint_file(md, options=None)
    assert any("broken internal link" in p.message for p in problems)


def test_valid_blob_link_passes(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "See [the example]"
        "(https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-example).\n"
    )
    problems = lint_docs._lint_file(md, options=None)
    assert problems == []


def test_repository_docs_pass():
    # The shipped docs and READMEs must lint cleanly.
    assert lint_docs.main() == 0
