import json
import math
import pathlib

import attrs
import cftime
import pandas as pd
import pytest

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import InvalidProviderException
from climate_ref_core.serialisation import from_wire, to_wire


def roundtrip(value):
    """Encode, push through json, and decode again."""
    return from_wire(json.loads(json.dumps(to_wire(value))))


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        "a string",
        3,
        4.5,
        [1, "two", None],
        {"a": 1, "b": [2, 3]},
        pathlib.Path("/scratch/output"),
        pathlib.Path("relative/path.json"),
    ],
)
def test_scalars_roundtrip(value):
    assert roundtrip(value) == value


@pytest.mark.parametrize("calendar", ["standard", "noleap", "360_day", "proleptic_gregorian"])
def test_cftime_roundtrip(calendar):
    value = cftime.datetime(1850, 2, 30 if calendar == "360_day" else 28, 6, 30, 15, calendar=calendar)

    result = roundtrip(value)

    assert result == value
    assert result.calendar == calendar


def test_cftime_keeps_microseconds():
    value = cftime.datetime(2000, 1, 1, 0, 0, 0, 123456, calendar="standard")

    assert roundtrip(value) == value


def test_unknown_type_is_rejected():
    with pytest.raises(TypeError, match="Cannot encode object as JSON"):
        to_wire(object())


def test_unknown_tag_is_rejected():
    with pytest.raises(ValueError, match="Unknown tagged value 'nope'"):
        from_wire({"__ref_type__": "nope"})


def test_frame_preserves_dtypes_and_index():
    index = pd.Index([10, 20], name="dataset_id")
    frame = pd.DataFrame(
        {
            "name": pd.Series(["a", "b"], dtype="str", index=index),
            "count": pd.Series([1, 2], dtype="int64", index=index),
            "flag": pd.Series([True, False], dtype="bool", index=index),
            "ratio": pd.Series([1.5, 2.5], dtype="float64", index=index),
        },
    )

    result = roundtrip(frame)

    pd.testing.assert_frame_equal(result, frame)


def test_frame_with_cftime_and_missing_values():
    frame = pd.DataFrame(
        {
            "start_time": pd.Series([cftime.datetime(1850, 1, 1, calendar="noleap"), None], dtype="object"),
            "calendar": pd.Series(["noleap", None], dtype="str"),
        }
    )

    result = roundtrip(frame)

    pd.testing.assert_frame_equal(result, frame)


def test_empty_frame_roundtrips():
    frame = pd.DataFrame({"path": pd.Series([], dtype="str")})

    pd.testing.assert_frame_equal(roundtrip(frame), frame)


@pytest.mark.parametrize("source_type", [SourceDatasetType.CMIP6, SourceDatasetType.obs4MIPs])
def test_real_catalog_roundtrips(data_catalog, source_type):
    """The catalogs carry cftime dates, a str dtype holding NaN, and int64 columns."""
    frame = data_catalog[source_type]

    pd.testing.assert_frame_equal(roundtrip(frame), frame)


def test_dataset_collection_roundtrips(cmip6_data_catalog):
    collection = DatasetCollection(
        cmip6_data_catalog,
        "instance_id",
        (("source_id", "ACCESS-ESM1-5"), ("variable_id", "tas")),
    )

    result = roundtrip(collection)

    pd.testing.assert_frame_equal(result.datasets, collection.datasets)
    assert result.slug_column == collection.slug_column
    assert result.selector == collection.selector
    assert result.stable_hash == collection.stable_hash


def test_definition_roundtrips(metric_definition):
    result = roundtrip(metric_definition)

    assert result.diagnostic_full_slug == metric_definition.diagnostic_full_slug
    assert result.execution_slug() == metric_definition.execution_slug()
    assert result.key == metric_definition.key
    assert result.output_directory == metric_definition.output_directory
    assert result.output_fragment() == metric_definition.output_fragment()
    assert result.datasets.hash == metric_definition.datasets.hash


def test_definition_roundtrip_preserves_every_field(cmip6_data_catalog):
    """
    `_structure_definition` lists the fields by hand,
    so a new field on `ExecutionDefinition` must fail here rather than fall off the wire.
    """
    definition = ExecutionDefinition(
        diagnostic=None,
        diagnostic_full_slug="example/global-mean-timeseries",
        key="key",
        datasets=ExecutionDatasetCollection(
            {SourceDatasetType.CMIP6: DatasetCollection(cmip6_data_catalog, "instance_id")}
        ),
        root_directory=pathlib.Path("/scratch"),
        output_directory=pathlib.Path("/scratch/fragment"),
    )

    # Every field except the omitted diagnostic must appear in the payload,
    # so a new field with a default cannot silently stay off the wire.
    payload = to_wire(definition)
    field_names = {field.name.lstrip("_") for field in attrs.fields(ExecutionDefinition)} - {"diagnostic"}
    assert field_names <= set(payload)

    decoded = roundtrip(definition)

    for field in attrs.fields(ExecutionDefinition):
        original = getattr(definition, field.name)
        restored = getattr(decoded, field.name)
        if isinstance(original, ExecutionDatasetCollection):
            assert restored.hash == original.hash, field.name
        else:
            assert restored == original, field.name


def test_definition_does_not_carry_the_diagnostic(metric_definition):
    payload = json.dumps(to_wire(metric_definition))

    assert metric_definition.diagnostic_full_slug in payload
    assert type(metric_definition.diagnostic).__name__ not in payload


def test_definition_does_not_carry_the_environment(metric_definition, monkeypatch):
    """
    The provider used to ship a snapshot of `os.environ` in every message.

    That put `REF_DATABASE_URL` and `SECRET_KEY` in the broker,
    readable by anything that could reach it, for as long as results were kept.
    See Climate-REF/climate-ref#847.
    """
    monkeypatch.setenv("REF_DATABASE_URL", "postgresql://ref:hunter2@db/ref")
    monkeypatch.setenv("SECRET_KEY", "a-secret-that-must-not-travel")

    payload = json.dumps(to_wire(metric_definition))

    assert "hunter2" not in payload
    assert "a-secret-that-must-not-travel" not in payload


def test_definitions_differing_only_in_diagnostic_are_not_equal(cmip6_data_catalog):
    """
    The diagnostic is a live object, so it takes no part in equality.

    Its slug has to stand in for it,
    otherwise two definitions running different diagnostics over the same datasets compare equal.
    """
    datasets = ExecutionDatasetCollection(
        {SourceDatasetType.CMIP6: DatasetCollection(cmip6_data_catalog, "instance_id")}
    )
    common = {
        "key": "key",
        "datasets": datasets,
        "root_directory": pathlib.Path("/scratch"),
        "output_directory": pathlib.Path("/scratch/fragment"),
    }

    one = ExecutionDefinition(diagnostic=None, diagnostic_full_slug="example/one", **common)
    other = ExecutionDefinition(diagnostic=None, diagnostic_full_slug="example/two", **common)

    assert one != other
    assert one == ExecutionDefinition(diagnostic=None, diagnostic_full_slug="example/one", **common)


def test_definition_derives_the_slug_it_compares_on(metric_definition):
    """
    A definition built from a diagnostic derives the slug at construction.

    Both sides of the wire therefore hold it as a field,
    so equality compares the diagnostic even though the diagnostic itself does not travel.
    """
    assert metric_definition._diagnostic_full_slug == metric_definition.diagnostic.full_slug()
    assert roundtrip(metric_definition)._diagnostic_full_slug == metric_definition._diagnostic_full_slug


def test_definition_resolves_its_diagnostic_lazily(cmip6_data_catalog):
    from climate_ref_example import provider  # noqa: PLC0415

    diagnostic = provider.get("global-mean-timeseries")
    definition = ExecutionDefinition(
        diagnostic=diagnostic,
        key="key",
        datasets=ExecutionDatasetCollection(
            {SourceDatasetType.CMIP6: DatasetCollection(cmip6_data_catalog, "instance_id")}
        ),
        root_directory=pathlib.Path("/scratch"),
        output_directory=pathlib.Path("/scratch/fragment"),
    )

    result = roundtrip(definition)

    # Nothing is resolved until the diagnostic is actually asked for
    assert result._diagnostic is None
    assert result.execution_slug() == definition.execution_slug()

    # Resolution returns the provider's own singleton, not a copy off the wire
    assert result.diagnostic is diagnostic
    assert result.diagnostic.provider is provider


def test_unregistered_provider_reports_what_is_available(cmip6_data_catalog):
    definition = ExecutionDefinition(
        diagnostic=None,
        diagnostic_full_slug="not-a-provider/some-diagnostic",
        key="key",
        datasets=ExecutionDatasetCollection(
            {SourceDatasetType.CMIP6: DatasetCollection(cmip6_data_catalog, "instance_id")}
        ),
        root_directory=pathlib.Path("/scratch"),
        output_directory=pathlib.Path("/scratch/fragment"),
    )

    # The slug alone is enough to identify the execution, so this still works
    assert definition.execution_slug() == "not-a-provider/some-diagnostic/key"

    with pytest.raises(InvalidProviderException, match="No provider with slug 'not-a-provider'"):
        definition.diagnostic


def test_definition_requires_a_diagnostic_or_a_slug(cmip6_data_catalog):
    # Construction stays permissive, matching a diagnostic that has not been registered
    # with a provider yet. The slug is only needed when something asks for it.
    definition = ExecutionDefinition(
        diagnostic=None,
        key="key",
        datasets=ExecutionDatasetCollection(
            {SourceDatasetType.CMIP6: DatasetCollection(cmip6_data_catalog, "instance_id")}
        ),
        root_directory=pathlib.Path("/scratch"),
        output_directory=pathlib.Path("/scratch/fragment"),
    )

    with pytest.raises(ValueError, match="Either diagnostic or diagnostic_full_slug must be given"):
        definition.diagnostic_full_slug


def test_unregistered_diagnostic_can_still_build_a_definition(cmip6_data_catalog):
    """A diagnostic only needs a provider once its slug is asked for."""
    from climate_ref_example.example import GlobalMeanTimeseries  # noqa: PLC0415

    definition = ExecutionDefinition(
        diagnostic=GlobalMeanTimeseries(),
        key="key",
        datasets=ExecutionDatasetCollection(
            {SourceDatasetType.CMIP6: DatasetCollection(cmip6_data_catalog, "instance_id")}
        ),
        root_directory=pathlib.Path("/scratch"),
        output_directory=pathlib.Path("/scratch/fragment"),
    )

    with pytest.raises(ValueError, match="Please register"):
        definition.execution_slug()


def test_result_roundtrips(metric_definition):
    result = ExecutionResult(
        definition=metric_definition,
        output_bundle_filename=pathlib.Path("output.json"),
        metric_bundle_filename=pathlib.Path("diagnostic.json"),
        series_filename=pathlib.Path("series.json"),
        successful=True,
    )

    decoded = roundtrip(result)

    assert decoded.successful is True
    assert decoded.retryable is False
    assert decoded.output_bundle_filename == result.output_bundle_filename
    assert decoded.metric_bundle_filename == result.metric_bundle_filename
    assert decoded.series_filename == result.series_filename
    assert decoded.definition.execution_slug() == metric_definition.execution_slug()


def test_failed_result_roundtrips(metric_definition):
    result = ExecutionResult.build_from_failure(metric_definition, retryable=True)

    decoded = roundtrip(result)

    assert decoded.successful is False
    assert decoded.retryable is True
    assert decoded.output_bundle_filename is None
    assert decoded.metric_bundle_filename is None


def test_execution_dataset_collection_roundtrips_every_source_type(cmip6_data_catalog):
    collection = ExecutionDatasetCollection(
        {SourceDatasetType.CMIP6: DatasetCollection(cmip6_data_catalog, "instance_id")}
    )
    definition = ExecutionDefinition(
        diagnostic=None,
        diagnostic_full_slug="example/global-mean-timeseries",
        key="key",
        datasets=collection,
        root_directory=pathlib.Path("/scratch"),
        output_directory=pathlib.Path("/scratch/fragment"),
    )

    decoded = roundtrip(definition)

    assert list(decoded.datasets.keys()) == [SourceDatasetType.CMIP6]
    assert decoded.datasets.hash == collection.hash


def dumps_strict(value):
    """Encode the way the Celery codec does, refusing non-JSON literals."""
    return json.dumps(to_wire(value), allow_nan=False)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinity_roundtrips(value):
    """An infinity is a real metric value, so it survives rather than becoming null."""
    assert roundtrip(value) == value
    assert json.loads(dumps_strict(value))["__ref_type__"] == "float"


def test_nan_roundtrips():
    """
    NaN is tagged like the infinities rather than nulled.

    Inside a frame the column dtype would restore it either way,
    but a NaN on its own has no dtype to be restored from,
    so nulling it would quietly turn it into None.
    """
    assert math.isnan(roundtrip(float("nan")))
    assert json.loads(dumps_strict(float("nan")))["__ref_type__"] == "float"


def test_none_is_still_null():
    """Only the missing markers become null. NaN is a float value, not one of them."""
    assert to_wire(None) is None
    assert to_wire(pd.NA) is None
    assert to_wire(pd.NaT) is None


def _reject_constant(name):
    raise AssertionError(f"body contains the bare literal {name}, which is not JSON")


def assert_strict_json(body):
    """Parse refusing NaN/Infinity/-Infinity, the literals Postgres rejects."""
    return json.loads(body, parse_constant=_reject_constant)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        [float("nan"), float("inf")],
        {"a": float("-inf")},
        pd.DataFrame({"value": pd.Series([1.0, float("nan"), float("inf")], dtype="float64")}),
    ],
)
def test_non_finite_floats_never_produce_invalid_json(value):
    """Regression for the bug class in Climate-REF/climate-ref#839."""
    assert_strict_json(dumps_strict(value))


def test_frame_with_infinity_roundtrips():
    frame = pd.DataFrame({"value": pd.Series([1.0, float("inf"), float("-inf")], dtype="float64")})

    pd.testing.assert_frame_equal(roundtrip(frame), frame)


def test_frame_with_nan_roundtrips():
    frame = pd.DataFrame({"value": pd.Series([1.0, float("nan")], dtype="float64")})

    pd.testing.assert_frame_equal(roundtrip(frame), frame)


def test_frame_with_timestamps_is_rejected_naming_the_column():
    frame = pd.DataFrame({"start_time": pd.to_datetime(["2000-01-01"])})

    with pytest.raises(TypeError, match="column 'start_time'"):
        to_wire(frame)
