from collections.abc import Iterable

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import (
    CommandLineMetric,
    DataRequirement,
    MetricExecutionDefinition,
    MetricExecutionResult,
)
from cmip_ref_metrics_pmp.pmp_driver import build_pmp_command, process_json_result
from cmip_ref_metrics_pmp.registry import fetch_reference_data


class ExtratropicalModesOfVariability_PDO(CommandLineMetric):
    """
    Calculate the annual cycle for a dataset
    """

    name = "PMP Extratropical modes of variability PDO"
    slug = "pmp-extratropical-modes-of-variability-pdo"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "frequency": "mon",
                        "experiment_id": ("historical", "hist-GHG"),
                        "variable_id": "ts",
                    }
                ),
                # Ignore some experiments because they are not relevant
                # JL: This won't ever be triggered because the experiment_id filter more specific
                # Should this be ignoring all AerChemMIP experiments?
                FacetFilter(facets={"experiment_id": ("amip",)}, keep=False),
            ),
            # Add cell areas to the groups
            # constraints=(AddCellAreas(),),
            # Run the metric on each unique combination of model, variable, experiment, and variant
            group_by=("source_id", "variable_id", "experiment_id", "variant_label", "member_id"),
        ),
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

        reference_dataset_name = "HadISST-1-1"

        source_id = input_datasets["source_id"].unique()[0]
        member_id = input_datasets["member_id"].unique()[0]

        return build_pmp_command(
            driver_file="variability_mode/variability_modes_driver.py",
            parameter_file="pmp_param_MoV-PDO.py",
            model_files=input_datasets.path.to_list(),
            reference_name=reference_dataset_name,
            reference_paths=[fetch_reference_data(reference_dataset_name)],
            source_id=source_id,
            member_id=member_id,
            output_directory_path=str(definition.output_directory),
        )

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
