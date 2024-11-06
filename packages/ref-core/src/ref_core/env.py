from environs import Env


def load_environment() -> Env:
    """
    Load the environment variables from the `.env` file.
    """
    new_env = Env()
    new_env.read_env(verbose=True)

    return new_env


env = load_environment()
