import os
import subprocess
from pathlib import Path

import requests

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import DataRequirement, Metric, MetricExecutionDefinition, MetricResult


class ILAMBStandardTAS(Metric):
    """
    Apply the standard ILAMB analysis methodology to a dataset (in this case specialized to Fluxnet's tas).
    """

    name = "ILAMB Standard TAS"
    slug = "ilamb-standard-tas"

    data_requirements = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(FacetFilter(facets={"variable_id": "tas", "experiment_id": ["historical", "land-hist"]}),),
        group_by=("source_id", "variant_label", "grid_label"),
    )

    def run(self, definition: MetricExecutionDefinition) -> MetricResult:
        """
        Run a metric from ILAMB. Will factor out in functions once it is more clear how things will work.

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the metric

        Returns
        -------
        The result of running the metric.
        """
        # ILAMB has ways of downloading/updating all reference datasets.
        # However, since this is just a test and until we sort out how to
        # specify non-model data requirements, I will just download a small file
        # locally.
        ref_file = definition.to_output_path("Fluxnet2015_tas.nc")
        if not ref_file.is_file():
            response = requests.get(
                "https://www.ilamb.org/ILAMB-Data/DATA/tas/FLUXNET2015/tas.nc", stream=True, timeout=10
            )
            response.raise_for_status()
            with open(ref_file, "wb") as out:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        out.write(chunk)

        # Write out a piece of the ILAMB configure file setup to make the
        # comparison with Fluxnet's tas. We could end up reading these out of a
        # larger file or iterate over them setting up these metrics
        # programmatically.
        cfg_file = definition.to_output_path("fluxnet_tas.cfg")
        with open(cfg_file, "w") as out:
            out.write("""
[h1: ILAMB-REF Output]
[h2: Surface Air Temperature]
variable = "tas"
[FLUXNET2015]
source   = "Fluxnet2015_tas.nc"
""")

        # Write a model setup file. It seems counter-intuitive to me that even
        # though I defined data_requirements above, I stil lhave to manually
        # apply them to the datasets found in the definition. TODO: Add pointers
        # to cell measures, think about how to assign colors for plots.
        mod_file = definition.to_output_path("models_fluxnet_tas.yaml")
        df = self.data_requirements.apply_filters(
            definition.metric_dataset[self.data_requirements.source_type].datasets
        )
        mod_config = ""
        for grp, dfg in df.groupby(list(self.data_requirements.group_by)):
            mod_name = "_".join(grp)
            paths = "\n".join(set([f"  - {Path(row.path).parent}" for _, row in dfg.iterrows()]))
            mod_config += f"""
{mod_name}:
  modelname: {mod_name}
  path: null
  paths:
{paths}
"""
        with open(mod_file, "w") as out:
            out.write(mod_config)

        # Run the ilamb study. The reference data path is relative to an
        # environment variable which we will set here. This also dumps the
        # screen output into a log file to help debug what happened.
        env = os.environ.copy()
        env["ILAMB_ROOT"] = str(definition.output_fragment)
        out = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "ilamb-run",
                "--config",
                str(cfg_file),
                "--model_setup",
                str(mod_file),
                "--build_dir",
                str(definition.output_directory),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,  # skip so I can write log and then raise manually
            env=env,
        )
        with open(definition.to_output_path("run.log"), mode="w") as log:
            log.write(out.stdout)
        out.check_returncode()

        return MetricResult.build_from_output_bundle(definition, {})
