from collections.abc import Iterable

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import (
    CommandLineMetric,
    DataRequirement,
    MetricExecutionDefinition,
    MetricExecutionResult,
)
from cmip_ref_metrics_pmp.pmp_driver import build_pmp_command, process_json_result


class ExtratropicalModesOfVariability(CommandLineMetric):
    """
    Calculate the annual cycle for a dataset
    """

    def __init__(self, mode_id: str):
        self.mode_id = mode_id.upper()
        self.name = f"PMP Extratropical modes of variability {mode_id}"
        self.slug = f"pmp-extratropical-modes-of-variability-{mode_id.lower()}"

        def get_data_requirements(
            obs_source: str,
            obs_variable: str,
            cmip_variable: str,
            extra_experiments: str | tuple[str, ...] | list[str] = (),
            remove_experiments: str | tuple[str, ...] | list[str] = (),
        ) -> tuple[DataRequirement, DataRequirement]:
            filters = [
                FacetFilter(
                    facets={
                        "frequency": "mon",
                        "experiment_id": ("historical", "hist-GHG", "piControl", *extra_experiments),
                        "variable_id": cmip_variable,
                    }
                )
            ]

            if remove_experiments:
                filters.append(FacetFilter(facets={"experiment_id": remove_experiments}, keep=False))

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
                    group_by=("source_id", "experiment_id", "variant_label", "member_id"),
                ),
            )

        if self.mode_id in ["PDO", "NPGO", "AMO"]:
            self.parameter_file = "pmp_param_MoV-ts.py"
            self.data_requirements = get_data_requirements(
                "HadISST-1-1", "ts", "ts", remove_experiments=("amip",)
            )
        elif self.mode_id in ["NAO", "NAM", "PNA", "NPO", "SAM"]:
            self.parameter_file = "pmp_param_MoV-psl.py"
            self.data_requirements = get_data_requirements("20CR", "psl", "psl", extra_experiments=("amip",))
        else:
            raise ValueError(
                f"Unknown mode_id '{self.mode_id}'. Must be one of PDO, NPGO, AMO, NAO, NAM, PNA, NPO, SAM"
            )

    def build_cmd(self, definition: MetricExecutionDefinition) -> Iterable[str]:
        """
        Build the command to run the metric

        Parameters
        ----------
        definition
            Definition of the metric execution

        Returns
        -------
            Command arguments to execute in the PMP environment
        """
        input_datasets = definition.metric_dataset[SourceDatasetType.CMIP6]
        source_id = input_datasets["source_id"].unique()[0]
        experiment_id = input_datasets["experiment_id"].unique()[0]
        member_id = input_datasets["member_id"].unique()[0]

        print("input_datasets:", input_datasets)
        print("source_id:", source_id)
        print("experiment_id:", experiment_id)
        print("member_id:", member_id)

        reference_dataset = definition.metric_dataset[SourceDatasetType.obs4MIPs]
        reference_dataset_name = reference_dataset["source_id"].unique()[0]
        # reference_dataset_path = reference_dataset.datasets[0]["path"]
        reference_dataset_path = reference_dataset.datasets.iloc[0]["path"]

        print("reference_dataset:", reference_dataset)
        print("reference_dataset_name:", reference_dataset_name)
        print("reference_dataset_path:", reference_dataset_path)

        params = {
            "driver_file": "variability_mode/variability_modes_driver.py",
            "parameter_file": self.parameter_file,
            "model_files": input_datasets.path.to_list(),
            "reference_name": reference_dataset_name,
            "reference_paths": reference_dataset_path,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "member_id": member_id,
            "output_directory_path": str(definition.output_directory),
            "variability_mode": self.mode_id,
        }

        # Add conditional parameters
        if self.mode_id in ["SAM"]:  # pragma: no cover
            params["osyear"] = 1950
            params["oeyear"] = 2005

        development_mode = False

        if development_mode:  # pragma: no cover
            # Get current time in 'yyyymmdd-hhmm' format
            from datetime import datetime

            current_time = datetime.now().strftime("%Y%m%d")
            output_directory_path = f"/Users/lee1043/Documents/Research/REF/output/{current_time}/{self.slug}"
            params.update(
                {
                    "msyear": 2000,
                    "meyear": 2005,
                    "osyear": 2000,
                    "oeyear": 2005,
                    "output_directory_path": output_directory_path,
                }
            )

        # Pass the parameters using **kwargs
        return build_pmp_command(**params)

    def build_metric_result(self, definition: MetricExecutionDefinition) -> MetricExecutionResult:
        """
        Build a metric result from the output of the PMP driver

        Parameters
        ----------
        definition
            Definition of the metric execution

        Returns
        -------
            Result of the metric execution
        """
        print("build_metric_result start")
        results_files = list(definition.output_directory.glob("*_cmec.json"))
        if len(results_files) != 1:  # pragma: no cover
            return MetricExecutionResult.build_from_failure(definition)

        # Find the other outputs
        png_files = list(definition.output_directory.glob("*.png"))
        data_files = list(definition.output_directory.glob("*.nc"))

        cmec_output, cmec_metric = process_json_result(results_files[0], png_files, data_files)

        return MetricExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output,
            cmec_metric_bundle=cmec_metric,
        )
