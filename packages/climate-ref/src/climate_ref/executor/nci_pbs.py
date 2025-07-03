from parsl.providers import PBSProProvider
from parsl.launchers import SimpleLauncher


template_string = '''#!/bin/bash

#PBS -N ${jobname}
#PBS -l ncpus=${ncpus}
#PBS -l mem=${mem}
#PBS -l jobfs=${jobfs}
#PBS -l storage=${storage}
#PBS -l walltime=$walltime
#PBS -o ${job_stdout_path}
#PBS -e ${job_stderr_path}
${scheduler_options}

${worker_init}

export JOBNAME="${jobname}"

${user_script}

'''


class NCIGadiPBSProProvider(PBSProProvider):
    """PBSProProvider with a custom template string for NCI."""

    def __init__(self, 
                 account=None,
                 queue=None,
                 ncpus=1,
                 mem='4GB',
                 jobfs='10GB',
                 storage='',
                 scheduler_options='',
                 worker_init='',
                 launcher=SimpleLauncher(),
                 walltime='00:20:00',
                 cmd_timeout=120):
        super().__init__(queue=queue,
                         account=account,
                         scheduler_options=scheduler_options,
                         worker_init=worker_init,
                         launcher=launcher,
                         walltime=walltime,
                         cmd_timeout=cmd_timeout)

        self.ncpus = ncpus
        self.mem = mem
        self.jobfs = jobfs
        self.storage = storage
        self.template_string = template_string

    def _write_submit_script(self, template, script_filename, job_name, configs):
        """Write the submit script using the custom template."""
        configs['ncpus'] = self.ncpus
        configs['mem'] = self.mem
        configs['jobfs'] = self.jobfs
        configs['storage'] = self.storage
        return super()._write_submit_script(template, script_filename, job_name, configs)
        
