"""
CMEC diagnostic bundle class

Following the CMEC diagnostic bundle standards at
https://github.com/Earth-System-Diagnostics-Standards/EMDS

To validate that a dictionary is compatible with the CMEC
diagnostic bundle standards, please use:
 - class instantiation: cmec = CMECMetric(**result_dict)
 - class model_validate method: cmec = CMECMetric.model_validate(result_dict)
Both ways will create the CMECMetric instance (cmec)
"""

import json
import pathlib
import warnings
from collections import Counter
from collections.abc import Generator
from copy import deepcopy
from enum import Enum
from typing import Any, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    RootModel,
    ValidationInfo,
    field_validator,
    model_validator,
    validate_call,
)
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue
from pydantic_core import CoreSchema
from typing_extensions import Self

from climate_ref_core.env import env
from climate_ref_core.metric_values import ScalarMetricValue

ALLOW_EXTRA_KEYS = env.bool("ALLOW_EXTRA_KEYS", default=True)


class MetricCV(Enum):
    """
    CMEC diagnostic bundle controlled vocabulary
    """

    DIMENSIONS = "DIMENSIONS"
    JSON_STRUCTURE = "json_structure"
    RESULTS = "RESULTS"
    PROVENANCE = "PROVENANCE"
    DISCLAIMER = "DISCLAIMER"
    NOTES = "NOTES"
    ATTRIBUTES = "attributes"


class MetricDimensions(RootModel[Any]):
    """
    CMEC diagnostic bundle DIMENSIONS object

    This describes the order of the dimensions and their possible values.
    The order of the dimensions matter as that determines how the executions are nested.
    """

    root: dict[str, Any] = Field(
        default={
            MetricCV.JSON_STRUCTURE.value: [],
        }
    )

    @model_validator(mode="after")
    def _validate_dimensions(self) -> Self:
        """Validate a MetricDimensions object"""
        # assert the items in json_structure are same as the keys of dimensions

        if MetricCV.JSON_STRUCTURE.value not in self.root.keys():
            raise ValueError(f"{MetricCV.JSON_STRUCTURE.value} is required keyword")

        if not (
            Counter(self.root[MetricCV.JSON_STRUCTURE.value])
            == Counter([k for k in self.root.keys() if k != MetricCV.JSON_STRUCTURE.value])
        ):
            raise ValueError("json_structure items are not in the keys of the DIMENSIONS")

        return self

    @validate_call
    def add_dimension(self, dim_name: str, dim_content: dict[str, Any]) -> None:
        """
        Add or update one dimension to MetricDimensions object

        Parameters
        ----------
        dim_name
            Name of new dimension to be added
        dim_content
            Dictionary contains contents associated with dim_name

        Returns
        -------
        :
            CMEC MetricDimensions object with dim_name added
        """
        if dim_name in self.root[MetricCV.JSON_STRUCTURE.value]:
            self.root[dim_name].update(dim_content)

        else:
            self.root[MetricCV.JSON_STRUCTURE.value].append(dim_name)
            self.root[dim_name] = dim_content

    @classmethod
    def merge_dimension(cls, metric_dim1: Any, metric_dim2: Any) -> Self:
        """
        Merge two MetricDimensions objects

        Parameters
        ----------
        metric_dim1
            First CMEC MetricDimensions object to be merged
        metric_dim2
            Second CMEC MetricDimensions object to be merged

        Returns
        -------
        :
            Return a merged CMEC MetricDimensions object
        """
        mdim1 = cls.model_validate(metric_dim1)
        mdim2 = cls.model_validate(metric_dim2)

        if not (mdim1.root[MetricCV.JSON_STRUCTURE.value] == mdim2.root[MetricCV.JSON_STRUCTURE.value]):
            raise ValueError("JSON_STRUCTURES are not same")

        merged_dim = {MetricCV.JSON_STRUCTURE.value: mdim1.root[MetricCV.JSON_STRUCTURE.value]}

        for dim in mdim1.root[MetricCV.JSON_STRUCTURE.value]:
            merged_dim[dim] = mdim1.root[dim]

        for dim in mdim2.root[MetricCV.JSON_STRUCTURE.value]:
            for key in mdim2.root[dim].keys():
                if key not in merged_dim[dim].keys():
                    merged_dim[dim][key] = mdim2.root[dim][key]
        return cls(merged_dim)

    def __getitem__(self, item: str) -> Any:
        return self.root[item]


class MetricResults(RootModel[Any]):
    """
    CMEC diagnostic bundle RESULTS object
    """

    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    root: dict[str, dict[Any, Any]]

    @classmethod
    def _check_nested_dict_keys(cls, nested: dict[Any, Any], metdims: dict[Any, Any], level: int = 0) -> None:  # noqa: PLR0912
        dim_name = metdims[MetricCV.JSON_STRUCTURE.value][level]

        dict_keys = set(nested.keys())
        if MetricCV.ATTRIBUTES.value in dict_keys:
            dict_keys.remove(MetricCV.ATTRIBUTES.value)

        if level < len(metdims[MetricCV.JSON_STRUCTURE.value]) - 1:
            if not (Counter(list(metdims[dim_name].keys())) == Counter(dict_keys)):
                raise ValueError(
                    f"Dimension key mismatch in '{dim_name}' and level {level}\n"
                    f"Actual keys: {sorted(dict_keys)}\n"
                    f"Expected keys: {sorted(metdims[dim_name].keys())}\n"
                    "Full actual structure:\n" + json.dumps(list(dict_keys), indent=2) + "\n\n"
                    "Full expected structure:\n" + json.dumps(metdims[dim_name], indent=2)
                )

            for key, value in nested.items():
                if key == MetricCV.ATTRIBUTES.value:
                    continue

                elif isinstance(value, dict):
                    cls._check_nested_dict_keys(value, metdims, level + 1)

                else:
                    raise ValueError(
                        f"{dim_name} is not the last/deepest dimension, \n"
                        f"a dictionary is expected for the key {key}"
                    )
        else:
            expected_keys = set(metdims[dim_name].keys())
            if not (dict_keys.issubset(expected_keys)):
                msg = f"Unknown dimension values: {dict_keys - expected_keys} for {dim_name}"
                if not ALLOW_EXTRA_KEYS:  # pragma: no cover
                    raise ValueError(f"{msg}\nExpected keys: {expected_keys}")
                else:
                    warnings.warn(msg)
                    for key in dict_keys - expected_keys:
                        nested.pop(key)

            tmp = dict(nested)
            if MetricCV.ATTRIBUTES.value in tmp:
                tmp.pop(MetricCV.ATTRIBUTES.value)
            StrNumDict(tmp)

    @field_validator("root", mode="after")
    @classmethod
    def _validate_results(cls, rlt: Any, info: ValidationInfo) -> Any:
        """Validate a MetricResults object"""
        if not isinstance(info.context, MetricDimensions):
            s = "\nTo validate MetricResults object, MetricDimensions is needed,\n"
            s += "please use model_validate(Results, context=MetricDimensions) to instantiate\n"
            raise ValueError(s)
        else:
            # executions = rlt.root
            results = rlt
            metdims = info.context.root
            if len(metdims[MetricCV.JSON_STRUCTURE.value]) == 0:
                if rlt != {}:
                    raise ValueError("Expected an empty dictionary for the metric bundle")
            else:
                cls._check_nested_dict_keys(results, metdims, level=0)

        return rlt


class StrNumDict(RootModel[Any]):
    """A class contains string key and numeric value"""

    model_config = ConfigDict(strict=True)
    root: dict[str, float | int | None]


def remove_dimensions(raw_metric_bundle: dict[str, Any], dimensions: str | list[str]) -> dict[str, Any]:
    """
    Remove the dimensions from the raw metric bundle

    Currently only the first dimension is supported to be removed.
    Multiple dimensions can be removed at once, but only if they are in order from the first
    dimension.

    Parameters
    ----------
    raw_metric_bundle
        The raw metric bundle to be modified
    dimensions
        The name of the dimensions to be removed

    Returns
    -------
        The new, modified metric bundle with the dimension removed
    """
    if isinstance(dimensions, str):
        dimensions = [dimensions]

    metric_bundle = deepcopy(raw_metric_bundle)

    for dim in dimensions:
        # bundle_dims is modified inplace below
        bundle_dims = metric_bundle[MetricCV.DIMENSIONS.value]

        level_id = bundle_dims[MetricCV.JSON_STRUCTURE.value].index(dim)
        if level_id != 0:
            raise NotImplementedError("Only the first dimension can be removed")

        values = list(bundle_dims[dim].keys())
        if len(values) != 1:
            raise ValueError(f"Can only remove dimensions with a single value. Found: {values}")
        value = values[0]

        new_result = metric_bundle[MetricCV.RESULTS.value][value]

        # Update the dimensions and results to remove the dimension
        bundle_dims.pop(dim)
        bundle_dims[MetricCV.JSON_STRUCTURE.value].pop(level_id)
        metric_bundle[MetricCV.RESULTS.value] = new_result

    return metric_bundle


class CMECMetric(BaseModel):
    """
    CMEC diagnostic bundle object

    Contains the diagnostics calculated during a diagnostic execution, in a standardised format.
    """

    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)

    DIMENSIONS: MetricDimensions
    """
    Describes the dimensionality of the diagnostics produced.

    This includes the order of dimensions in `RESULTS`
    """
    RESULTS: dict[str, Any]
    """
    The diagnostic values.

    Results is a nested dictionary of values.
    The order of the nested dictionaries corresponds to the order of the dimensions.
    """
    PROVENANCE: dict[str, Any] | None = None
    """
    Provenance information

    Not currently used in the REF.
    The provenance information from the output bundle is used instead
    """
    DISCLAIMER: dict[str, Any] | None = None
    """
    Disclaimer information

    Not currently used in the REF.
    """
    NOTES: dict[str, Any] | None = None
    """
    Additional notes.

    Not currently used in the REF.
    """

    @model_validator(mode="after")
    def _validate_metrics(self) -> Self:
        """Validate a CMECMetric object"""
        # validate executions data
        results = self.RESULTS
        MetricResults.model_validate(results, context=self.DIMENSIONS)
        return self

    @validate_call
    def dump_to_json(self, json_file: str | pathlib.Path = "./cmec.json") -> None:
        """
        Save the CMECMetric object to a file in JSON format

        Parameters
        ----------
        json_file
            JSON file path in the CMEC format to be saved

        Returns
        -------
        :
            None
        """
        pathlib.Path(json_file).write_text(
            json.dumps(
                self.model_dump(mode="json"),
                indent=2,
                allow_nan=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    @validate_call
    def load_from_json(cls, json_file: FilePath) -> Self:
        """
        Create CMECMetric object from a compatible json file

        Parameters
        ----------
        json_file
            JSON file path to be read

        Returns
        -------
        :
            CMEC Diagnostic object if the file is CMEC-compatible
        """
        json_str = pathlib.Path(json_file).read_text(encoding="utf-8")
        metric_obj = cls.model_validate_json(json_str)

        return metric_obj

    @classmethod
    def _merge(cls, dict_a: dict[Any, Any], dict_b: dict[Any, Any]) -> dict[Any, Any]:
        """Merge the values from dict_b into dict_a inplace"""
        for key, value_b in dict_b.items():
            if key in dict_a:
                if isinstance(dict_a[key], dict) and isinstance(value_b, dict):
                    cls._merge(dict_a[key], value_b)
                else:
                    dict_a[key] = value_b
            else:
                dict_a[key] = value_b
        return dict_a

    @classmethod
    def _fill(cls, mdict: dict[Any, Any], mdims: dict[Any, Any], level: int = 0) -> None:
        dim_name = mdims[MetricCV.JSON_STRUCTURE.value][level]
        for key in mdims[dim_name].keys():
            if key not in mdict:
                if level < len(mdims[MetricCV.JSON_STRUCTURE.value]) - 1:
                    mdict[key] = {}

        for key, value in mdict.items():
            if (
                isinstance(value, dict)
                and level < len(mdims[MetricCV.JSON_STRUCTURE.value]) - 1
                and key != MetricCV.ATTRIBUTES.value
            ):
                cls._fill(value, mdims, level + 1)

    @classmethod
    @validate_call
    def merge(cls, metric_obj1: Any, metric_obj2: Any) -> Self:
        """
        Merge two CMECMetric objects with the same json_structure

        Parameters
        ----------
        metric_obj1
            First CMECMetric object to be merged
        metric_obj2
            Second CMECMetric object to be merged

        Returns
        -------
        :
            Merged CMEC Diagnostic object
        """
        mobj1 = cls.model_validate(metric_obj1)
        mobj2 = cls.model_validate(metric_obj2)

        merged_obj_dims = MetricDimensions.merge_dimension(mobj1.DIMENSIONS, mobj2.DIMENSIONS)

        result1 = mobj1.RESULTS
        result2 = mobj2.RESULTS
        merged_obj_rlts = cls._merge(dict(result1), result2)

        cls._fill(merged_obj_rlts, merged_obj_dims.root)

        MetricResults.model_validate(merged_obj_rlts, context=merged_obj_dims)

        return cls(DIMENSIONS=merged_obj_dims, RESULTS=merged_obj_rlts)

    def remove_dimensions(self, dimensions: str | list[str]) -> "CMECMetric":
        """
        Remove the dimensions from the metric bundle

         Currently only the first dimension is supported to be removed.
        Multiple dimensions can be removed at once, but only if they are in order from the first
        dimension..

        Parameters
        ----------
        dimensions
            The name of the dimension to be removed

        Returns
        -------
        :
            A new CMECMetric object with the dimensions removed
        """
        return CMECMetric(**remove_dimensions(self.model_dump(), dimensions))

    def prepend_dimensions(self, values: dict[str, str]) -> "CMECMetric":
        """
        Prepend the existing metric values with additional dimensions

        Parameters
        ----------
        values
            Additional metric dimensions and their values to be added to the metric bundle

        Returns
        -------
        :
            A new CMECMetric object with the additional dimensions prepended to the existing metric bundle
        """
        results: dict[str, Any] = {}
        current = results

        existing_dimensions = self.DIMENSIONS.root[MetricCV.JSON_STRUCTURE.value]
        for dim in existing_dimensions:
            if dim in values:
                raise ValueError(f"Dimension {dim!r} is already defined in the metric bundle")

        dimensions = self.DIMENSIONS.model_copy(deep=True)
        dimensions.root[MetricCV.JSON_STRUCTURE.value] = [
            *list(values.keys()),
            *existing_dimensions,
        ]

        # Nest each new dimension inside the previous one
        for key, value in values.items():
            if not isinstance(value, str):
                raise TypeError(f"Dimension value {value!r} is not a string")

            current[value] = {}
            current = current[value]
            dimensions.root[key] = {value: {}}
        # Add the existing dimensions as the innermost dimensions
        current.update(self.RESULTS)

        MetricResults.model_validate(results, context=dimensions)

        result = self.model_copy()
        result.DIMENSIONS = dimensions
        result.RESULTS = results
        return result

    @staticmethod
    def create_template() -> dict[str, Any]:
        """
        Return an empty dictionary in CMEC diagnostic bundle format
        """
        default_dimensions = MetricDimensions()

        return {
            MetricCV.DIMENSIONS.value: default_dimensions.root,
            MetricCV.RESULTS.value: {},
            MetricCV.PROVENANCE.value: None,
            MetricCV.DISCLAIMER.value: None,
            MetricCV.NOTES.value: None,
        }

    def iter_results(self) -> Generator[ScalarMetricValue]:
        """
        Iterate over the executions in the diagnostic bundle

        This will yield a dictionary for each result, with the dimensions and the value

        Returns
        -------
            A generator of diagnostic values

        """
        dimensions = cast(list[str], self.DIMENSIONS[MetricCV.JSON_STRUCTURE.value])

        if len(dimensions) == 0:
            # There is no data to iterate over
            return

        yield from _walk_results(dimensions, self.RESULTS, {})


def _walk_results(
    dimensions: list[str], results: dict[str, Any], metadata: dict[str, str]
) -> Generator[ScalarMetricValue]:
    assert len(dimensions), "Not enough dimensions"
    dimension = dimensions[0]
    for key, value in results.items():
        if key == MetricCV.ATTRIBUTES.value:
            continue
        metadata[dimension] = key
        if isinstance(value, float | int):
            yield ScalarMetricValue(
                dimensions=metadata, value=value, attributes=results.get(MetricCV.ATTRIBUTES.value)
            )
        elif value is None:
            # Replace any None values with NaN
            # This translates null values in JSON to Python NaN's
            # Missing values are different from NaN values
            yield ScalarMetricValue(
                dimensions=metadata, value=np.nan, attributes=results.get(MetricCV.ATTRIBUTES.value)
            )
        else:
            yield from _walk_results(dimensions[1:], value, {**metadata})


class CMECGenerateJsonSchema(GenerateJsonSchema):
    """
    Customized CMEC JSON schema generation
    """

    def generate(self: Self, schema: CoreSchema, mode: JsonSchemaMode = "validation") -> JsonSchemaValue:
        """Generate customized json schema"""
        json_schema = super().generate(schema, mode=mode)
        json_schema["title"] = "CMEC"
        json_schema["$schema"] = self.schema_dialect
        return json_schema
