import re
import shutil
import subprocess
import textwrap
from typing import Any

from parsl.launchers import SimpleLauncher
from parsl.providers import PBSProProvider


class SmartPBSProvider(PBSProProvider):
    """
    A PBSProProvider subclass that adapts to systems where `-l select` is not supported.

    Falls back to individual resource requests (ncpus, mem, jobfs, storage) if needed.
    """

    def __init__(  # noqa: PLR0913
        self,
        account: str | None = None,
        queue: str | None = None,
        scheduler_options: str = "",
        worker_init: str = "",
        nodes_per_block: int | None = 1,
        cpus_per_node: int | None = 1,
        ncpus: int | None = None,
        mem: str = "4GB",
        jobfs: str = "10GB",
        storage: str = "",
        init_blocks: int = 1,
        min_blocks: int = 0,
        max_blocks: int = 1,
        parallelism: int = 1,
        launcher: SimpleLauncher = SimpleLauncher(),
        walltime: str = "00:20:00",
        cmd_timeout: int = 120,
    ) -> None:
        self.ncpus = ncpus
        self.mem = mem
        self.jobfs = jobfs
        self.storage = storage
        self._select_supported = self._detect_select_support()

        # Prepare fallback resource dictionary
        self._fallback_resources = {"mem": mem, "jobfs": jobfs, "storage": storage}

        # Parse and strip select if present in scheduler_options
        if not self._select_supported and "-l select=" in scheduler_options:
            scheduler_options = self._parse_select_from_scheduler_options(scheduler_options)

        # Determine fallback ncpus
        if "ncpus" not in self._fallback_resources:
            self._fallback_resources["ncpus"] = str(ncpus if ncpus is not None else (cpus_per_node or 1))

        # Map ncpus to cpus_per_node if needed (select mode only)
        if self._select_supported:
            if not ncpus and cpus_per_node:
                cpus_per_node = ncpus
            elif ncpus and cpus_per_node and int(ncpus) != int(cpus_per_node):
                print(f"Warning: ncpus={ncpus} and cpus_per_node={cpus_per_node} differ.")
                print(f"Using cpus_per_node={cpus_per_node}.")
        else:
            cpus_per_node = int(self._fallback_resources["ncpus"])

        super().__init__(
            account=account,
            queue=queue,
            scheduler_options=scheduler_options,
            select_options="",  # Not used; we handle resources ourselves
            worker_init=worker_init,
            nodes_per_block=nodes_per_block,
            cpus_per_node=cpus_per_node,
            init_blocks=init_blocks,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            parallelism=parallelism,
            launcher=launcher,
            walltime=walltime,
            cmd_timeout=cmd_timeout,
        )  # type: ignore

        if not self._select_supported:
            self.template_string = self._fallback_template()

    def _detect_select_support(self) -> bool:
        """Detect whether `-l select` is supported by the underlying PBS system."""
        qsub_path = shutil.which("qsub")
        if qsub_path is None:
            raise RuntimeError("qsub command not found. Ensure PBS is installed and in PATH.")

        try:
            result = subprocess.run(  # noqa: S603
                [qsub_path, "-l", "wd,select=1:ncpus=1", "--version"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            stderr = result.stderr.decode().lower()
            return "unknown" not in stderr and result.returncode == 0
        except Exception:
            return False

    def _parse_select_from_scheduler_options(self, scheduler_options: str) -> str:
        """
        Parse `-l select=...` from scheduler_options and update fallback resources.

        Removes the select line from scheduler_options.
        """
        select_pattern = r"-l\s+select=([^\s]+)"
        match = re.search(select_pattern, scheduler_options)
        if match:
            select_string = match.group(1)
            scheduler_options = re.sub(select_pattern, "", scheduler_options).strip()

            parts = select_string.split(":")[1:]  # skip the initial `select=1`
            for part in parts:
                if "=" in part:
                    key, val = part.split("=")
                    self._fallback_resources[key.strip()] = val.strip()
        return scheduler_options

    def _fallback_template(self) -> str:
        """Submit script template used if `select` is not supported."""
        return textwrap.dedent("""\
            #!/bin/bash
            #PBS -N ${jobname}
            #PBS -l ncpus=${ncpus}
            #PBS -l mem=${mem}
            #PBS -l jobfs=${jobfs}
            #PBS -l walltime=${walltime}
            #PBS -l storage=${storage}
            #PBS -o ${job_stdout_path}
            #PBS -e ${job_stderr_path}
            ${scheduler_options}

            ${worker_init}

            export JOBNAME="${jobname}"
            ${user_script}

        """)

    def _write_submit_script(
        self, template: str, script_filename: str, job_name: str, configs: dict[str, Any]
    ) -> str:
        """Inject fallback values into the submit script if `select` is not supported."""
        if not self._select_supported:
            configs.setdefault("ncpus", self._fallback_resources.get("ncpus", "1"))
            configs.setdefault("mem", self._fallback_resources.get("mem", "4GB"))
            configs.setdefault("jobfs", self._fallback_resources.get("jobfs", "10GB"))
            configs.setdefault("storage", self._fallback_resources.get("storage", "gdata1"))
        return super()._write_submit_script(template, script_filename, job_name, configs)  # type: ignore
