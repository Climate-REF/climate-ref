from collections.abc import Iterable

from loguru import logger

from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    CommandLineDiagnostic,
    DataRequirement,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_pmp.pmp_driver import build_pmp_command, process_json_result


class ExtratropicalModesOfVariability(CommandLineDiagnostic):
    """
    Calculate the extratropical modes of variability for a given area
    """

    ts_modes = ("PDO", "NPGO", "AMO")
    psl_modes = ("NAO", "NAM", "PNA", "NPO", "SAM")

    facets = (
        "source_id",
        "member_id",
        "experiment_id",
        "reference_source_id",
        "mode",
        "season",
        "method",
        "statistic",
    )

    def __init__(self, mode_id: str):
        self.mode_id = mode_id.upper()
        self.name = f"Extratropical modes of variability: {mode_id}"
        self.slug = f"extratropical-modes-of-variability-{mode_id.lower()}"

        def _get_data_requirements(
            obs_source: str,
            obs_variable: str,
            model_variable: str,
            extra_experiments: str | tuple[str, ...] | list[str] = (),
        ) -> tuple[DataRequirement, DataRequirement]:
            filters = [
                FacetFilter(
                    facets={
                        "frequency": "mon",
                        "experiment_id": ("historical", "hist-GHG", "piControl", *extra_experiments),
                        "variable_id": model_variable,
                    }
                )
            ]

            return (
                DataRequirement(
                    source_type=SourceDatasetType.obs4MIPs,
                    filters=(
                        FacetFilter(facets={"source_id": (obs_source,), "variable_id": (obs_variable,)}),
                    ),
                    group_by=("source_id", "variable_id"),
                ),
                DataRequirement(
                    source_type=SourceDatasetType.CMIP6,
                    filters=tuple(filters),
                    # TODO: remove unneeded variant_label
                    group_by=("source_id", "experiment_id", "variant_label", "member_id"),
                ),
            )

        if self.mode_id in self.ts_modes:
            self.parameter_file = "pmp_param_MoV-ts.py"
            self.data_requirements = _get_data_requirements("HadISST-1-1", "ts", "ts")
        elif self.mode_id in self.psl_modes:
            self.parameter_file = "pmp_param_MoV-psl.py"
            self.data_requirements = _get_data_requirements("20CR", "psl", "psl", extra_experiments=("amip",))
        else:
            raise ValueError(
                f"Unknown mode_id '{self.mode_id}'. Must be one of {self.ts_modes + self.psl_modes}"
            )

    def build_cmd(self, definition: ExecutionDefinition) -> Iterable[str]:
        """
        Build the command to run the diagnostic

        Parameters
        ----------
        definition
            Definition of the diagnostic execution

        Returns
        -------
            Command arguments to execute in the PMP environment
        """
        input_datasets = definition.datasets[SourceDatasetType.CMIP6]
        source_id = input_datasets["source_id"].unique()[0]
        experiment_id = input_datasets["experiment_id"].unique()[0]
        member_id = input_datasets["member_id"].unique()[0]

        logger.debug(f"input_datasets: {input_datasets}")
        logger.debug(f"source_id: {source_id}")
        logger.debug(f"experiment_id: {experiment_id}")
        logger.debug(f"member_id: {member_id}")

        reference_dataset = definition.datasets[SourceDatasetType.obs4MIPs]
        reference_dataset_name = reference_dataset["source_id"].unique()[0]
        reference_dataset_path = reference_dataset.datasets.iloc[0]["path"]

        logger.debug(f"reference_dataset: {reference_dataset}")
        logger.debug(f"reference_dataset_name: {reference_dataset_name}")
        logger.debug(f"reference_dataset_path: {reference_dataset_path}")

        model_files = input_datasets.path.to_list()

        if len(model_files) != 1:
            # Have some logic to replace the dates in the filename with a wildcard
            raise NotImplementedError("Only one model file is supported at this time.")

        if isinstance(model_files, list):
            modpath = " ".join([str(p) for p in model_files])
        else:
            modpath = model_files

        if isinstance(reference_dataset_path, list):
            reference_data_path = " ".join([str(p) for p in reference_dataset_path])
        else:
            reference_data_path = reference_dataset_path

        # Build the command to run the PMP driver script
        params: dict[str, str | int | None] = {
            "variability_mode": self.mode_id,
            "modpath": modpath,
            "modpath_lf": "none",
            "exp": experiment_id,
            "realization": member_id,
            "modnames": source_id,
            "reference_data_name": reference_dataset_name,
            "reference_data_path": reference_data_path,
            "results_dir": str(definition.output_directory),
            "cmec": None,
            "no_provenance": None,
        }

        # Add conditional parameters
        if self.mode_id in ["SAM"]:  # pragma: no cover
            params["osyear"] = 1950
            params["oeyear"] = 2005

        # Pass the parameters using **kwargs
        return build_pmp_command(
            driver_file="variability_mode/variability_modes_driver.py",
            parameter_file=self.parameter_file,
            **params,
        )

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Build a diagnostic result from the output of the PMP driver

        Parameters
        ----------
        definition
            Definition of the diagnostic execution

        Returns
        -------
            Result of the diagnostic execution
        """
        results_files = list(definition.output_directory.glob("*_cmec.json"))
        if len(results_files) != 1:  # pragma: no cover
            logger.warning(f"A single cmec output file not found: {results_files}")
            return ExecutionResult.build_from_failure(definition)

        # Find the other outputs
        png_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.png")]
        data_files = [definition.as_relative_path(f) for f in definition.output_directory.glob("*.nc")]

        cmec_output_bundle, cmec_metric_bundle = process_json_result(results_files[0], png_files, data_files)

        # Add additional metadata to the metrics
        input_selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()
        reference_selectors = definition.datasets[SourceDatasetType.obs4MIPs].selector_dict()
        cmec_metric_bundle = cmec_metric_bundle.remove_dimensions(
            [
                "model",
                "realization",
                "reference",
            ],
        ).prepend_dimensions(
            {
                "source_id": input_selectors["source_id"],
                "member_id": input_selectors["member_id"],
                "experiment_id": input_selectors["experiment_id"],
                "reference_source_id": reference_selectors["source_id"],
            }
        )

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output_bundle,
            cmec_metric_bundle=cmec_metric_bundle,
        )
