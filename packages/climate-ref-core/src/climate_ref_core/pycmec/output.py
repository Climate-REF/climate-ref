"""
CMEC output bundle class

An output bundle describes the figures and data generated
during a diagnostic execution.

Following the CMEC output bundle standards at
https://github.com/Earth-System-Diagnostics-Standards/EMDS

To validate that a dictionary is compatible with the CMEC
output bundle standards, please use:
 - class instantiation: cmec = CMECOutput(**result_dict)
 - class model_validate method: cmec = CMECOutput.model_validate(result_dict)
Both ways will create the CMECOutput instance (cmec)
"""

import pathlib
from enum import Enum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    FilePath,
    validate_call,
)


class OutputCV(Enum):
    """CMEC output bundle controlled vocabulary"""

    INDEX = "index"
    PROVENANCE = "provenance"
    DATA = "data"
    PLOTS = "plots"
    HTML = "html"
    METRICS = "diagnostics"
    FILENAME = "filename"
    LONG_NAME = "long_name"
    DESCRIPTION = "description"
    ENVIRONMENT = "environment"
    MODELDATA = "modeldata"
    OBSDATA = "obsdata"
    LOG = "log"

    # REF-specific additions
    # These are not part of the CMEC standard
    DIMENSIONS = "dimensions"


class OutputProvenance(BaseModel):
    """CMEC output bundle provenance object"""

    environment: dict[str, str | None]
    """
    Key/value pairs listing all relevant diagnostic and
    framework environment variables.
    """

    modeldata: str | list[str] | dict[str, Any]
    """
    Path to the model data used in this analysis.
    """

    obsdata: str | list[str] | dict[str, Any]
    """
    Key/value pairs containing short names and versions of
    all observational datasets used.
    """

    log: str
    """
    Filename of a free format log file written during execution.
    """

    model_config = ConfigDict(strict=True, extra="allow")


class OutputDict(BaseModel):
    """
    Description of an output
    """

    filename: str  # Filename of plot produced (relative to output directory path)
    long_name: str  # Human readable name describing the plot
    description: str  # Description of what is depicted in the plot

    dimensions: dict[str, str] | None = None
    """
    Dimensions associated with this output

    These dimensions can be used to filter the output for given sources/experiments/variables etc.
    They should match the controlled vocabulary dimensions defined in the system.
    """

    model_config = ConfigDict(strict=True, extra="allow")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        return setattr(self, key, value)


class CMECOutput(BaseModel):
    """
    CMEC output bundle object

    Describes the assets generated during a diagnostic execution.
    """

    index: str | None = None
    """
    Short name of the plot/html/diagnostic that should be opened
    when the user chooses to “open” the output bundle.
    """

    provenance: OutputProvenance
    """
    Command line and version information used to execute the DEM.
    This includes environment variables and the observational
    datasets used (including dataset versions).
    """

    data: dict[str, OutputDict] | None = None
    """
    (optional) Dictionary of data files produced with the output.
    """

    plots: dict[str, OutputDict] | None = None
    """
    (optional) Dictionary of plots produced with the output.
    """

    html: dict[str, OutputDict] | None = None
    """
    (optional) Dictionary of html files produced with the output.
    """

    metrics: dict[str, OutputDict] | None = None
    """
    (optional) Dictionary of diagnostic files produced with the output.
    """

    model_config = ConfigDict(strict=True, extra="allow")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        return setattr(self, key, value)

    @validate_call
    def dump_to_json(self, json_path: str | pathlib.Path = "./cmec_output.json") -> None:
        """Save the CMECOutput object to a JSON file in the CMEC format"""
        pathlib.Path(json_path).write_text(self.model_dump_json(indent=2))

    @classmethod
    @validate_call
    def load_from_json(cls, json_file: FilePath) -> Self:
        """
        Create a CMECOuput object from a CMEC standard JSON file
        """
        json_str = pathlib.Path(json_file).read_text()
        output_obj = cls.model_validate_json(json_str)

        return output_obj

    # from PMP
    @validate_call
    def update(
        self,
        output_kw: Literal["data", "plots", "html", "diagnostics"],
        *,
        short_name: str,
        dict_content: dict[str, Any],
    ) -> None:
        """
        Update the content of output_kw using short_name and dict_content pair

        Parameters
        ----------
        output_kw
            CMEC output bundle keywords, one of [data, plots, html, diagnostics]
        short_name
            Key name of the dictionary nested in the output_kw dictionary
        dict_content
            Value of the dictionary with the key of short_name

        Returns
        -------
        :
            CMECOutput object with content updated
        """
        cmec_output = OutputDict(**dict_content)

        if self[output_kw] is None:
            self[output_kw] = {}
        self[output_kw].update({short_name: cmec_output})

    @staticmethod
    def create_template() -> dict[str, Any]:
        """
        Return a empty dictionary in CMEC output bundle format
        """
        return {
            OutputCV.INDEX.value: "index.html",
            OutputCV.PROVENANCE.value: {
                OutputCV.ENVIRONMENT.value: {},
                OutputCV.MODELDATA.value: [],
                OutputCV.OBSDATA.value: {},
                OutputCV.LOG.value: "cmec_output.log",
            },
            OutputCV.DATA.value: {},
            OutputCV.HTML.value: {},
            OutputCV.METRICS.value: {},
            OutputCV.PLOTS.value: {},
        }
