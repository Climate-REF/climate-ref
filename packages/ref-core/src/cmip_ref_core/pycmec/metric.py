"""
CMEC metric bundle class

Following the CMEC metric bundle standards at
https://github.com/Earth-System-Diagnostics-Standards/EMDS

To validate that a dictionary is compatible with the CMEC
metric bundle standards, please use:
 - class instantiation: cmec = CMECMetric(**result_dict)
 - class model_validate method: cmec = CMECMetric.model_validate(result_dict)
Both ways will create the CMECMetric instance (cmec)
"""

import pathlib
from collections import Counter
from enum import Enum
from typing import Any

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


class MetricCV(Enum):
    """
    CMEC metric bundle controlled vocabulary
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
    CMEC metric bundle DIMENSIONS object

    This describes the order of the dimensions and their possible values.
    The order of the dimensions matter as that determines how the results are nested.
    """

    root: dict[str, Any] = Field(
        default={
            MetricCV.JSON_STRUCTURE.value: ["model", "metric"],
            "model": {},
            "metric": {},
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


class MetricResults(RootModel[Any]):
    """
    CMEC metric bundle RESULTS object
    """

    model_config = ConfigDict(strict=True)
    root: dict[str, dict[Any, Any]]

    @classmethod
    def _check_nested_dict_keys(cls, nested: dict[Any, Any], metdims: dict[Any, Any], level: int = 0) -> None:
        dim_name = metdims[MetricCV.JSON_STRUCTURE.value][level]

        dict_key = list(nested.keys())
        if MetricCV.ATTRIBUTES.value in dict_key:
            dict_key.remove(MetricCV.ATTRIBUTES.value)

        if not (Counter(list(metdims[dim_name].keys())) == Counter(dict_key)):
            raise ValueError("Error in dicts of Results")

        for key, value in nested.items():
            if not (key == MetricCV.ATTRIBUTES.value):
                if isinstance(value, dict) and level < len(metdims[MetricCV.JSON_STRUCTURE.value]) - 1:
                    cls._check_nested_dict_keys(value, metdims, level + 1)
                elif isinstance(value, dict):
                    tmp = dict(value)
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
            # results = rlt.root
            results = rlt
            metdims = info.context.root
            cls._check_nested_dict_keys(results, metdims, level=0)

        return rlt


class StrNumDict(RootModel[Any]):
    """A class contains string key and numeric value"""

    model_config = ConfigDict(strict=True)
    root: dict[str, float | str]


class CMECMetric(BaseModel):
    """
    CMEC metric bundle object

    Contains the metrics calculated during a metric execution, in a standardised format.
    """

    model_config = ConfigDict(strict=True, extra="allow")

    DIMENSIONS: MetricDimensions
    """
    Describes the dimensionality of the metrics produced.

    This includes the order of dimensions in `RESULTS`
    """
    RESULTS: dict[str, Any]
    """
    The metric values.

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
        # validate results data
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
        pathlib.Path(json_file).write_text(self.model_dump_json(indent=2))

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
            CMEC Metric object if the file is CMEC-compatible
        """
        json_str = pathlib.Path(json_file).read_text()
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
            Merged CMEC Metric object
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

    @staticmethod
    def create_template() -> dict[str, Any]:
        """
        Return an empty dictionary in CMEC metric bundle format
        """
        default_dimensions = MetricDimensions()

        return {
            MetricCV.DIMENSIONS.value: default_dimensions.root,
            MetricCV.RESULTS.value: {},
            MetricCV.PROVENANCE.value: None,
            MetricCV.DISCLAIMER.value: None,
            MetricCV.NOTES.value: None,
        }


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
