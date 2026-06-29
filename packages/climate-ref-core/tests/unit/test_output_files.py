"""Unit tests for :mod:`climate_ref_core.output_files`."""

import pytest

from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.output_files import (
    PLACEHOLDER_OUTPUT_DIR,
    PLACEHOLDER_SOFTWARE_ROOT_DIR,
    PLACEHOLDER_TEST_DATA_DIR,
    PlaceholderMap,
    copy_execution_outputs,
    copy_output_file,
)
from climate_ref_core.pycmec.output import CMECOutput


def test_sanitise_and_expand_round_trip(tmp_path):
    output_dir = tmp_path / "scratch" / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()

    placeholders = PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir)
    original = f"path={output_dir}\ndata={test_data_dir}\n"
    (directory / "diagnostic.json").write_text(original)

    placeholders.sanitise(directory)
    sanitised = (directory / "diagnostic.json").read_text()
    assert str(output_dir) not in sanitised
    assert str(test_data_dir) not in sanitised
    assert PLACEHOLDER_OUTPUT_DIR in sanitised
    assert PLACEHOLDER_TEST_DATA_DIR in sanitised

    placeholders.hydrate(directory)
    assert (directory / "diagnostic.json").read_text() == original


def test_sanitise_software_root_round_trip(tmp_path):
    output_dir = tmp_path / "scratch" / "output"
    test_data_dir = tmp_path / "test-data"
    software_root_dir = tmp_path / "software"
    directory = tmp_path / "regression"
    directory.mkdir()

    placeholders = PlaceholderMap.for_baseline(
        test_data_dir=test_data_dir, software_root_dir=software_root_dir
    ).with_output(output_dir)
    # A provenance command line referencing the software root and the output dir.
    original = f"cmd={software_root_dir}/conda/bin/driver.py --out {output_dir}\n"
    (directory / "diagnostic.json").write_text(original)

    placeholders.sanitise(directory)
    sanitised = (directory / "diagnostic.json").read_text()
    assert str(software_root_dir) not in sanitised
    assert PLACEHOLDER_SOFTWARE_ROOT_DIR in sanitised
    assert PLACEHOLDER_OUTPUT_DIR in sanitised

    placeholders.hydrate(directory)
    assert (directory / "diagnostic.json").read_text() == original


def test_software_root_substitution_is_opt_in(tmp_path):
    # Without software_root_dir, no <SOFTWARE_ROOT_DIR> substitution occurs (back-compatible default).
    software_root_dir = tmp_path / "software"
    directory = tmp_path / "regression"
    directory.mkdir()
    original = f"cmd={software_root_dir}/conda/bin/driver.py\n"
    (directory / "diagnostic.json").write_text(original)

    PlaceholderMap.for_baseline(test_data_dir=tmp_path / "td").with_output(tmp_path / "out").sanitise(
        directory
    )

    assert (directory / "diagnostic.json").read_text() == original


def test_sanitise_only_touches_text_globs(tmp_path):
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()

    binary = f"{output_dir}".encode()
    (directory / "data.nc").write_bytes(binary)
    (directory / "meta.json").write_text(str(output_dir))

    PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir).sanitise(directory)

    # Binary file is untouched; JSON is rewritten.
    assert (directory / "data.nc").read_bytes() == binary
    assert (directory / "meta.json").read_text() == PLACEHOLDER_OUTPUT_DIR


def test_sanitise_longest_match_first(tmp_path):
    # test_data_dir is a prefix of output_dir; the longer path must win.
    test_data_dir = tmp_path / "base"
    output_dir = tmp_path / "base" / "scratch"
    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / "a.json").write_text(str(output_dir))

    PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir).sanitise(directory)

    assert (directory / "a.json").read_text() == PLACEHOLDER_OUTPUT_DIR


@pytest.mark.parametrize("suffix", ("json", "txt", "yaml", "yml", "html", "xml"))
def test_sanitise_covers_all_text_globs(suffix, tmp_path):
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / f"artefact.{suffix}").write_text(str(output_dir))

    PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir).sanitise(directory)

    assert (directory / f"artefact.{suffix}").read_text() == PLACEHOLDER_OUTPUT_DIR


def test_sanitise_respects_custom_globs(tmp_path):
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / "config.cfg").write_text(str(output_dir))
    (directory / "skip.json").write_text(str(output_dir))

    PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir).sanitise(
        directory, globs=("*.cfg",)
    )

    # Only the custom glob is rewritten; the default JSON glob is left untouched.
    assert (directory / "config.cfg").read_text() == PLACEHOLDER_OUTPUT_DIR
    assert (directory / "skip.json").read_text() == str(output_dir)


def test_sanitise_leaves_unmatched_content_untouched(tmp_path):
    # A text file that contains neither path must be left byte-for-byte identical
    # (exercises the "no rewrite" branch that skips the write).
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()
    original = "nothing to replace here\n"
    target = directory / "untouched.json"
    target.write_text(original)
    before = target.stat().st_mtime_ns

    PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir).sanitise(directory)

    assert target.read_text() == original
    assert target.stat().st_mtime_ns == before


def test_placeholder_map_as_replacements_maps_real_to_token(tmp_path):
    output_dir = tmp_path / "out"
    test_data_dir = tmp_path / "td"
    software_root_dir = tmp_path / "sw"

    placeholders = PlaceholderMap.for_baseline(
        test_data_dir=test_data_dir, software_root_dir=software_root_dir
    ).with_output(output_dir)

    assert placeholders.as_replacements() == {
        str(output_dir): PLACEHOLDER_OUTPUT_DIR,
        str(test_data_dir): PLACEHOLDER_TEST_DATA_DIR,
        str(software_root_dir): PLACEHOLDER_SOFTWARE_ROOT_DIR,
    }


def test_placeholder_map_for_baseline_omits_software_root_when_absent(tmp_path):
    # The optional software root is declared once on the map; an absent root is an explicit omission.
    tokens = {token for token, _ in PlaceholderMap.for_baseline(test_data_dir=tmp_path / "td").pairs}

    assert PLACEHOLDER_TEST_DATA_DIR in tokens
    assert PLACEHOLDER_SOFTWARE_ROOT_DIR not in tokens
    assert PLACEHOLDER_OUTPUT_DIR not in tokens  # only added by with_output


def test_placeholder_map_with_output_returns_a_new_map(tmp_path):
    # Frozen value object: with_output does not mutate the base map.
    base = PlaceholderMap.for_baseline(test_data_dir=tmp_path / "td")
    bound = base.with_output(tmp_path / "out")

    assert PLACEHOLDER_OUTPUT_DIR not in {token for token, _ in base.pairs}
    assert PLACEHOLDER_OUTPUT_DIR in {token for token, _ in bound.pairs}


def test_placeholder_map_with_output_rebinding_replaces(tmp_path):
    # A second with_output replaces the first: one <OUTPUT_DIR> entry, hydrating to the latest dir.
    first = tmp_path / "first"
    second = tmp_path / "second"
    base = PlaceholderMap.for_baseline(test_data_dir=tmp_path / "td")
    rebound = base.with_output(first).with_output(second)

    output_dirs = [path for token, path in rebound.pairs if token == PLACEHOLDER_OUTPUT_DIR]
    assert output_dirs == [second]

    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / "a.json").write_text(PLACEHOLDER_OUTPUT_DIR)
    rebound.hydrate(directory)
    assert (directory / "a.json").read_text() == str(second)


def test_placeholder_map_is_output_bound(tmp_path):
    base = PlaceholderMap.for_baseline(test_data_dir=tmp_path / "td")
    assert not base.is_output_bound
    assert base.with_output(tmp_path / "out").is_output_bound


@pytest.mark.parametrize("filename", ("bundle.zip", "nested/bundle.zip"))
def test_copy_output_file_returns_relpath(filename, tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    src = scratch / fragment / filename
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()

    rel = copy_output_file(scratch, results, fragment, filename)

    assert (results / fragment / filename).exists()
    assert rel.as_posix() == filename


def test_copy_output_file_missing_raises(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    with pytest.raises(FileNotFoundError, match=r"Could not find missing\.json"):
        copy_output_file(scratch, results, "frag", "missing.json")


def test_copy_output_file_accepts_absolute_filename(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    src = scratch / fragment / "bundle.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()

    # An absolute path under scratch/fragment is normalised back to a relative path.
    rel = copy_output_file(scratch, results, fragment, src)

    assert rel.as_posix() == "bundle.json"
    assert (results / fragment / "bundle.json").exists()


def test_copy_output_file_rejects_identical_directories(tmp_path):
    same = (tmp_path / "shared").resolve()
    with pytest.raises(ValueError, match="must differ"):
        copy_output_file(same, same, "frag", "bundle.json")


class _FakeResult:
    def __init__(self, metric_bundle_filename, output_bundle_filename=None, series_filename=None):
        self.metric_bundle_filename = metric_bundle_filename
        self.output_bundle_filename = output_bundle_filename
        self.series_filename = series_filename


def _seed(scratch, fragment, *names):
    for name in names:
        path = scratch / fragment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_copy_execution_outputs_curated_set(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, EXECUTION_LOG_FILENAME, "bundle.json", "series.json")

    result = _FakeResult(metric_bundle_filename="bundle.json", series_filename="series.json")
    copied = copy_execution_outputs(scratch, results, fragment, result, include_log=True)

    names = {p.as_posix() for p in copied}
    assert names == {EXECUTION_LOG_FILENAME, "bundle.json", "series.json"}
    for name in names:
        assert (results / fragment / name).exists()


def test_copy_execution_outputs_excludes_log_by_default(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json")

    result = _FakeResult(metric_bundle_filename="bundle.json")
    copied = copy_execution_outputs(scratch, results, fragment, result)

    assert [p.as_posix() for p in copied] == ["bundle.json"]
    assert not (results / fragment / EXECUTION_LOG_FILENAME).exists()


def test_copy_execution_outputs_requires_metric_bundle(tmp_path):
    result = _FakeResult(metric_bundle_filename=None)
    with pytest.raises(ValueError, match="without a metric bundle"):
        copy_execution_outputs(tmp_path / "scratch", tmp_path / "results", "frag", result)


def _write_output_bundle(scratch, fragment, bundle_filename, referenced):
    """Seed a CMEC output bundle plus the plots/data/html files it references."""
    cmec_output = CMECOutput(**CMECOutput.create_template())
    for attr, short_name, filename in referenced:
        cmec_output.update(
            attr,
            short_name=short_name,
            dict_content={"long_name": short_name, "filename": filename, "description": ""},
        )
        _seed(scratch, fragment, filename)
    cmec_output.dump_to_json(scratch / fragment / bundle_filename)


def test_copy_execution_outputs_includes_output_bundle_and_referenced_files(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json")
    _write_output_bundle(
        scratch,
        fragment,
        "output.json",
        referenced=[
            ("plots", "fig1", "fig_1.png"),
            ("plots", "fig2", "nested/fig_2.png"),
            ("html", "index", "index.html"),
        ],
    )

    result = _FakeResult(metric_bundle_filename="bundle.json", output_bundle_filename="output.json")
    copied = copy_execution_outputs(scratch, results, fragment, result)

    names = {p.as_posix() for p in copied}
    assert names == {"bundle.json", "output.json", "fig_1.png", "nested/fig_2.png", "index.html"}
    for name in names:
        assert (results / fragment / name).exists()


def test_copy_execution_outputs_full_curated_set(tmp_path):
    # Exercises every branch together: log + metric + output bundle (+ referenced) + series.
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, EXECUTION_LOG_FILENAME, "bundle.json", "series.json")
    _write_output_bundle(scratch, fragment, "output.json", referenced=[("data", "d1", "data_1.nc")])

    result = _FakeResult(
        metric_bundle_filename="bundle.json",
        output_bundle_filename="output.json",
        series_filename="series.json",
    )
    copied = copy_execution_outputs(scratch, results, fragment, result, include_log=True)

    names = {p.as_posix() for p in copied}
    assert names == {
        EXECUTION_LOG_FILENAME,
        "bundle.json",
        "output.json",
        "data_1.nc",
        "series.json",
    }
    for name in names:
        assert (results / fragment / name).exists()


def test_copy_execution_outputs_empty_output_bundle(tmp_path):
    # An output bundle that references no files still copies the bundle itself.
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json")
    _write_output_bundle(scratch, fragment, "output.json", referenced=[])

    result = _FakeResult(metric_bundle_filename="bundle.json", output_bundle_filename="output.json")
    copied = copy_execution_outputs(scratch, results, fragment, result)

    assert {p.as_posix() for p in copied} == {"bundle.json", "output.json"}


def test_copy_execution_outputs_extra_globs_persists_unreferenced_files(tmp_path):
    # Raw artefacts the bundle does not reference are persisted when declared via extra_globs.
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(
        scratch,
        fragment,
        "bundle.json",
        "run/a/provenance.yml",
        "run/b/provenance.yml",
        "driver_cmip6_cmec.json",
        "leave_me_alone.log",
    )

    result = _FakeResult(metric_bundle_filename="bundle.json")
    copied = copy_execution_outputs(
        scratch,
        results,
        fragment,
        result,
        extra_globs=("run/*/provenance.yml", "*_cmec.json"),
    )

    names = {p.as_posix() for p in copied}
    assert names == {
        "bundle.json",
        "run/a/provenance.yml",
        "run/b/provenance.yml",
        "driver_cmip6_cmec.json",
    }
    for name in names:
        assert (results / fragment / name).exists()
    # A file matched by no glob is not persisted.
    assert not (results / fragment / "leave_me_alone.log").exists()


def test_copy_execution_outputs_extra_globs_supports_recursive_patterns(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json", "a/b/c/deep.yml")

    result = _FakeResult(metric_bundle_filename="bundle.json")
    copied = copy_execution_outputs(scratch, results, fragment, result, extra_globs=("**/*.yml",))

    assert {p.as_posix() for p in copied} == {"bundle.json", "a/b/c/deep.yml"}


def test_copy_execution_outputs_extra_globs_deduped_against_curated(tmp_path):
    # A glob that also matches an already-curated file must not duplicate its manifest key.
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json", "series.json", "extra.yml")

    result = _FakeResult(metric_bundle_filename="bundle.json", series_filename="series.json")
    copied = copy_execution_outputs(
        scratch,
        results,
        fragment,
        result,
        extra_globs=("*.json", "*.yml"),  # *.json also matches the curated bundle/series files
    )

    names = [p.as_posix() for p in copied]
    assert sorted(names) == ["bundle.json", "extra.yml", "series.json"]
    assert len(names) == len(set(names)), "curated files must not be duplicated by an extra glob"


def test_copy_execution_outputs_extra_globs_skips_directories(tmp_path):
    # A glob that matches a directory is ignored; only files under it are copied.
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json", "run/keep.yml")

    result = _FakeResult(metric_bundle_filename="bundle.json")
    copied = copy_execution_outputs(scratch, results, fragment, result, extra_globs=("run", "run/*"))

    assert {p.as_posix() for p in copied} == {"bundle.json", "run/keep.yml"}
