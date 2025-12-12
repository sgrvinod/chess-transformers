"""Utility helpers for loading chess-transformers configuration files.

This module provides functions for dynamically loading configuration files
from the chess-transformers package by name.

Key Features:
    - ConfigNotFoundError: Exception raised when a config cannot be found.
    - import_config: Load a config by name from models.configs or data.configs.

Notes:
    The search is case-insensitive, so both "vole" and "Vole" will match
    a configuration file named "vole.py".
"""

from importlib import import_module

from chess_transformers.data.configs.base import DataConfig
from chess_transformers.utilities.loggers import setup_logger


logger = setup_logger(__file__)


class ConfigNotFoundError(Exception):
    """Raised when a requested configuration module cannot be located.

    This error is triggered after exhausting the expected search paths for
    configuration files and not finding a module that both exists and exposes
    a top-level ``config`` object.
    """

    def __init__(self, message: str = "This configuration file does not exist!"):
        """Initialize the exception.

        Args:
            message: Custom error message to display. Defaults to a generic
                "This configuration file does not exist!".
        """
        super().__init__(message)


def import_config(config_name: str) -> DataConfig:
    """Load a data or model configuration.

    The helper attempts to import ``config_name`` from the two canonical
    configuration namespaces:

    ``chess_transformers.models.configs`` and ``chess_transformers.data.configs``.

    The search is case-insensitive: e.g., "LE25ct", "le25ct", and "Le25CT"
    will all match a file named "LE25ct.py".

    Each configuration module must expose a top-level variable named
    ``config`` that is an instance of :class:`chess_transformers.data.configs.base.DataConfig`.

    Args:
        config_name: File stem (without the ``.py`` suffix) of the configuration
            module to import. Matching is case-insensitive.

    Returns:
        The parsed configuration object.

    Raises:
        ConfigNotFoundError: If no matching configuration module is found, or
            the module lacks a ``config`` attribute.
    """
    from pathlib import Path

    # Base packages to search
    config_packages = [
        "chess_transformers.models.configs",
        "chess_transformers.data.configs",
    ]

    config_name_lower = config_name.lower()

    for package_name in config_packages:
        # Get the directory path for this package
        try:
            package_module = import_module(package_name)
        except ImportError:
            continue

        package_path = Path(package_module.__file__).parent

        # Find a matching config file (case-insensitive)
        for py_file in package_path.glob("*.py"):
            if py_file.stem.startswith("_"):
                continue  # Skip __init__.py, __pycache__, etc.
            if py_file.stem.lower() == config_name_lower:
                # Found a match - import using the actual filename
                module_path = f"{package_name}.{py_file.stem}"
                try:
                    module = import_module(module_path)
                except ImportError as e:
                    logger.critical(f"Failed to import {module_path}: {e}")
                    raise ConfigNotFoundError(f"Failed to import {module_path}: {e}")

                config = getattr(module, "config", None)
                if config is None:
                    logger.critical(f"No `config` attribute in module {module_path}")
                    raise ConfigNotFoundError(
                        f"No `config` attribute in module {module_path}"
                    )

                logger.info(f"Imported configuration from {module_path}")
                return config

    logger.critical(f"There is no configuration file for {config_name}")
    raise ConfigNotFoundError(f"There is no configuration file for {config_name}")
