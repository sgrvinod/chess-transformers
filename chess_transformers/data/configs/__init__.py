"""Dataset configuration modules for chess-transformers.

This package contains configuration classes and dataset-specific instances
used by the data processing pipeline.

Key Features:
    - base: Base DataConfig class for defining dataset parameters.
    - LE25c: Configuration for Lichess Elite 2025 checkmate dataset.
    - LE25ct: Configuration for Lichess Elite 2025 checkmate + time control dataset.

Notes:
    Use the `import_config` utility function from `chess_transformers.utilities.configs`
    to load configurations by name rather than importing directly.
"""

from chess_transformers.data.configs.base import DataConfig

__all__ = ["DataConfig"]
