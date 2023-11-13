from importlib import import_module

__all__ = ["data", "models"]


class ConfigNotFound(Exception):
    """
    An exception to raise when an configuration file for a dataset or model is not found.
    """

    def __init__(self, message="This configuration file does not exist!"):
        self.message = message

    def __str__(self):
        return self.message


def import_config(config_name):
    try:
        CONFIG = import_module(
            "chess_transformers.configs.models.{}".format(config_name)
        )
    except ModuleNotFoundError:
        try:
            CONFIG = import_module(
                "chess_transformers.configs.data.{}".format(config_name)
            )
        except ModuleNotFoundError:
            raise ConfigNotFound

    return CONFIG
