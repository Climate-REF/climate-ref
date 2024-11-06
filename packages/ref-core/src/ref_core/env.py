from environs import Env


def load_environment() -> Env:
    """
    Load the environment variables from the `.env` file.
    """
    env = Env()
    env.read_env()

    return env


env = load_environment()
