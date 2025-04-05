from collections.abc import Iterable

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import CommandLineMetric, DataRequirement, MetricExecutionDefinition, MetricResult
from cmip_ref_metrics_pmp.pmp_driver import build_pmp_command, process_json_result


class AnnualCycle(CommandLineMetric):
    """
    Calculate the annual cycle for a dataset
    """

    def __init__(self) -> None:
        self.name = "PMP Annual Cycle"
        self.slug = "pmp-annual-cycle"
        self.data_requirements = (
            DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(FacetFilter(facets={"source_id": ("GPCP-2-3"), "variable_id": ("pr")}),),
                group_by=("source_id", "variable_id"),
            ),
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "frequency": "mon",
                            "experiment_id": ("amip", "historical", "hist-GHG", "piControl"),
                            "variable_id": ("pr"),
                        }
                    ),
                ),
                group_by=("source_id", "experiment_id", "variant_label", "member_id", "variable_id"),
            ),
        )

        self.parameter_file_1 = "pmp_param_annualcycle_1-clims.py"
        self.parameter_file_2 = "pmp_param_annualcycle_1-metrics.py"

    def build_cmd(self, params: dict) -> Iterable[str]:
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
        # Pass the parameters using **kwargs
        return build_pmp_command(**params)

    def build_cmds(self, definition: MetricExecutionDefinition) -> list[Iterable[str]]:
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
        variable_id = input_datasets["variable_id"].unique()[0]
        model_files = input_datasets.path.to_list()[0]  # Limits to one file, needs to be fixed in the future

        print("build_cmd start")

        print("source_id:", source_id)
        print("experiment_id:", experiment_id)
        print("member_id:", member_id)
        print("variable_id:", variable_id)

        reference_dataset = definition.metric_dataset[SourceDatasetType.obs4MIPs]
        reference_dataset_name = reference_dataset["source_id"].unique()[0]
        reference_dataset_path = reference_dataset.datasets.iloc[0]["path"]

        print("reference_dataset_name:", reference_dataset_name)
        print("reference_dataset_path:", reference_dataset_path)

        parameter_file_1_clims = self.parameter_file_1
        # parameter_file_2_metrics = self.parameter_file_2

        development_mode = True

        if development_mode:  # pragma: no cover
            # Get current time in 'yyyymmdd-hhmm' format
            from datetime import datetime

            current_time = datetime.now().strftime("%Y%m%d")
            output_directory_path = f"/Users/lee1043/Documents/Research/REF/output/{current_time}/{self.slug}"
        else:
            output_directory_path = str(definition.output_directory)

        cmds = []

        # Build the command for climatologies
        for data in ["reference", "model"]:
            if data == "reference":
                data_name = reference_dataset_name
                data_path = reference_dataset_path
            else:
                data_name = f"{source_id}-{experiment_id}-{member_id}"
                data_path = model_files

            params = {
                "driver_file": "mean_climate/pcmdi_compute_climatologies.py",
                "parameter_file": parameter_file_1_clims,
                "vars": variable_id,
                "infile": data_path,
                "outfile": f"{output_directory_path}/{variable_id}_{data_name}_clims.nc",
            }

            if development_mode:  # pragma: no cover
                params.update(
                    {
                        "start": "2000-01",
                        "end": "2005-12",
                        "outfile": f"{output_directory_path}/{variable_id}_{data_name}_clims.nc",
                    }
                )

            cmds.append(self.build_cmd(params))

        # Build the command for metrics

        print("jwlee123_test ac cmds:", cmds)

        return cmds

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
        print("build_metric_result start")
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

    def run(self, definition: MetricExecutionDefinition) -> MetricResult:
        """
        Run the metric on the given configuration.

        Parameters
        ----------
        definition : MetricExecutionDefinition
            The configuration to run the metric on.

        Returns
        -------
        :
            The result of running the metric.
        """
        print("PMP annual cycle run start")
        cmds = self.build_cmds(definition)

        runs = [self.provider.run(cmd) for cmd in cmds]
        print("jwlee test, runs:", runs)

        return self.build_metric_result(definition)
