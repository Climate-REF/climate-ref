from collections.abc import Iterable

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import CommandLineMetric, DataRequirement, MetricExecutionDefinition, MetricResult
from cmip_ref_metrics_pmp.pmp_driver import build_pmp_command, process_json_result


class ExtratropicalModesOfVariability(CommandLineMetric):
    """
    Calculate the annual cycle for a dataset
    """

    def __init__(self, mode_id: str):
        self.mode_id = mode_id.upper()
        self.name = f"PMP Extratropical modes of variability {mode_id}"
        self.slug = f"pmp-extratropical-modes-of-variability-{mode_id.lower()}"

        if self.mode_id in ["PDO", "NPGO", "AMO"]:
            self.data_requirements = (
                DataRequirement(
                    source_type=SourceDatasetType.obs4MIPs,
                    filters=(
                        FacetFilter(
                            facets={
                                "source_id": ("HadISST-1-1",),
                                "variable_id": ("ts",),
                            }
                        ),
                    ),
                    group_by=("source_id", "variable_id"),
                ),
                DataRequirement(
                    source_type=SourceDatasetType.CMIP6,
                    filters=(
                        FacetFilter(
                            facets={
                                "frequency": "mon",
                                "experiment_id": ("historical", "hist-GHG", "piControl"),
                                "variable_id": "ts",
                            }
                        ),
                        # Ignore some experiments because they are not relevant
                        FacetFilter(facets={"experiment_id": ("amip",)}, keep=False),
                    ),
                    # Add cell areas to the groups
                    # constraints=(AddCellAreas(),),
                    # Run the metric on each unique combination of model, variable, experiment, and variant
                    group_by=("source_id", "experiment_id", "variant_label", "member_id"),
                ),
            )
        elif self.mode_id in ["NAO", "NAM", "PNA", "NPO", "SAM"]:
            self.data_requirements = (
                DataRequirement(
                    source_type=SourceDatasetType.obs4MIPs,
                    filters=(
                        FacetFilter(
                            facets={
                                "source_id": ("20CR",),
                                "variable_id": ("psl",),
                            }
                        ),
                    ),
                    group_by=("source_id", "variable_id"),
                ),
                DataRequirement(
                    source_type=SourceDatasetType.CMIP6,
                    filters=(
                        FacetFilter(
                            facets={
                                "frequency": "mon",
                                "experiment_id": ("historical", "hist-GHG", "piControl", "amip"),
                                "variable_id": "ts",
                            }
                        ),
                    ),
                    # Add cell areas to the groups
                    # constraints=(AddCellAreas(),),
                    # Run the metric on each unique combination of model, variable, experiment, and variant
                    group_by=("source_id", "experiment_id", "variant_label", "member_id"),
                ),
            )
        else:
            raise ValueError(
                f"Unknown mode_id {mode_id}. Must be one of " "PDO, NPGO, AMO, NAO, NAM, PNA, NPO, SAM"
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
        reference_dataset_path = reference_dataset.datasets.iloc[0]["path"]

        print("reference_dataset:", reference_dataset)
        print("reference_dataset_name:", reference_dataset_name)
        print("reference_dataset_path:", reference_dataset_path)

        if self.mode_id in ["PDO", "NPGO", "AMO"]:
            parameter_file = "pmp_param_MoV-PDO.py"
        elif self.mode_id in ["NAO", "NAM", "PNA", "NPO", "SAM"]:
            parameter_file = "pmp_param_MoV-NAO.py"
        else:
            raise ValueError(
                f"Unknown mode_id {self.mode_id}. Must be one of " "PDO, NPGO, AMO, NAO, NAM, PNA, NPO, SAM"
            )

        return build_pmp_command(
            driver_file="variability_mode/variability_modes_driver.py",
            parameter_file=parameter_file,
            model_files=input_datasets.path.to_list(),
            reference_name=reference_dataset_name,
            reference_paths=reference_dataset_path,
            source_id=source_id,
            experiment_id=experiment_id,
            member_id=member_id,
            output_directory_path=str(definition.output_directory),
        )

    def build_metric_result(self, definition: MetricExecutionDefinition) -> MetricResult:
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
        results_files = list(definition.output_directory.glob("*_cmec.json"))
        if len(results_files) != 1:  # pragma: no cover
            return MetricResult.build_from_failure(definition)

        # Find the other outputs
        png_files = list(definition.output_directory.glob("*.png"))
        data_files = list(definition.output_directory.glob("*.nc"))

        cmec_output, cmec_metric = process_json_result(results_files[0], png_files, data_files)

        return MetricResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_output,
            cmec_metric_bundle=cmec_metric,
        )
