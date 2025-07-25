import tempfile
from pathlib import Path

import pytest

from climate_ref.executor.pbs_scheduler import SmartPBSProvider


@pytest.mark.parametrize("force_select", [True, False])
def test_smart_pbs_provider_script_generation(force_select):
    """Test SmartPBSProvider script generation in both select and fallback modes."""

    class TestablePBSProvider(SmartPBSProvider):
        def _detect_select_support(self):
            return force_select  # Force override to control logic path

    provider = TestablePBSProvider(
        account="test",
        queue="normal",
        scheduler_options="-l select=1:ncpus=4:mem=8GB:jobfs=20GB",
        worker_init="module load myenv",
        ncpus=4,
        mem="8GB",
        jobfs="20GB",
        storage="gdata1+gdata2",
        walltime="00:30:00",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "job.sh"
        job_name = "pytest_job"
        stdout = Path(tmpdir) / "out.txt"
        stderr = Path(tmpdir) / "err.txt"

        configs = {
            "jobname": job_name,
            "job_stdout_path": str(stdout),
            "job_stderr_path": str(stderr),
            "scheduler_options": provider.scheduler_options,
            "worker_init": provider.worker_init,
            "user_script": "echo 'Hello from PBS'",
            "walltime": provider.walltime,
        }

        # Use internal method for template + script writing
        provider._write_submit_script(provider.template_string, str(script_path), job_name, configs)

        script_content = script_path.read_text()

        # Check general content
        assert f"#PBS -N {job_name}" in script_content
        assert "echo 'Hello from PBS'" in script_content

        if force_select:
            assert "#PBS -l select=1:ncpus=4:mem=8GB:jobfs=20GB" in script_content
            assert "#PBS -l ncpus=" not in script_content
        else:
            assert "#PBS -l ncpus=4" in script_content
            assert "#PBS -l mem=8GB" in script_content
            assert "#PBS -l jobfs=20GB" in script_content
