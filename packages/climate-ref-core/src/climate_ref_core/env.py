"""Environment variable management"""

import os

import platformdirs
from environs import Env


def _set_defaults() -> None:
    os.environ.setdefault("REF_CONFIGURATION", str(platformdirs.user_config_path("climate_ref")))


def get_env() -> Env:
    """
    Get the current environment

    Returns
    -------
    :
        The current environment including any environment variables loaded from the .env file
        and any defaults set by this application.
    """
    # Set the default values for the environment variables
    _set_defaults()

    env = Env(expand_vars=True)

    # Load the environment variables from the .env file
    # This will override any defaults set above
    env.read_env(verbose=True)

    return env


def get_available_cpu_count() -> int:
    """
    Detect the number of CPU cores available considering cgroup limitations.

    Returns
    -------
    :
        The number of allocated CPUs or total cpu count if not running in a cgroup-limited environment.
    """
    try:
        # Check for CPU quota
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())

        if quota > 0 and period > 0:
            return quota // period

        # If no quota, check for cpuset
        with open("/sys/fs/cgroup/cpuset/cpuset.cpus") as f:
            cpuset = f.read().strip()
            # Parse the cpuset string (e.g., "0-3", "0,2")
            count = 0
            for part in cpuset.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    count += end - start + 1
                else:
                    count += 1
            return count

    except FileNotFoundError:
        # Not running in a cgroup-limited environment or cgroup files not found
        return os.cpu_count() or 1


env = get_env()
