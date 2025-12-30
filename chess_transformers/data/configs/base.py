"""Base configuration classes for chess data processing.

This module defines the base Pydantic model used to configure dataset
processing parameters. All dataset-specific configurations should inherit
from or instantiate the classes defined here.

Key Features:
    - DataConfig: Configuration for LMDB dataset processing.
    - Type-safe configuration with Pydantic validation.
    - Automatic LMDB filename generation from config name.

Notes:
    DataConfig names are validated to be unique (case-insensitive) to prevent
    accidental configuration conflicts.

Example:
    Creating a custom dataset configuration::

        from chess_transformers.data.configs.base import DataConfig

        config = DataConfig(
            name="MyDataset",
            expected_rows=1_000_000,
            val_split_fraction=0.95,
        )
"""

from typing import Literal, Optional, Self
from pydantic import BaseModel, Field, model_validator


# Registry to track all config names and prevent duplicates
_CONFIG_REGISTRY = set()

# Type alias for legal mask mode options
LegalMaskMode = Literal["disjoint", "conditional"]


class DataConfig(BaseModel):
    """Configuration for processing chess game data into LMDB format.

    This Pydantic model defines the parameters needed to process raw chess
    game data (moves and FENs) into an LMDB database suitable for training.

    Attributes:
        name: Unique identifier for this dataset configuration. Used to
            construct the data folder path and output LMDB filename.
        expected_rows: Estimated number of datapoints (half-moves) in the
            dataset. Used to pre-allocate LMDB map size. A good estimate
            is number_of_games * 50 (average half-moves per game by winner).
        val_split_fraction: Fraction of data to use for training, with the
            remainder used for validation. For example, 0.98 means 98%
            training and 2% validation. Defaults to 0.98.
        legal_mask_mode: Mode for storing legal move masks. Options are:
            - None: Do not store legal move masks (default).
            - "disjoint": Store all legal from-squares and all legal
              to-squares as independent 64-bit masks.
            - "conditional": Store all legal from-squares, plus the legal
              to-squares for the ground-truth from-square only.

    Example:
        >>> config = DataConfig(
        ...     name="LE25ct",
        ...     expected_rows=396_881 * 50,
        ...     val_split_fraction=0.95,
        ... )
        >>> config.lmdb_filename
        'LE25ct.lmdb'
    """

    name: str = Field(..., description="Name and identifier for this configuration")
    expected_rows: int = Field(
        ..., description="Expected number of rows for calculating LMDB map size"
    )
    val_split_fraction: Optional[float] = Field(
        0.98, description="Fraction of data to use for training (rest for validation)"
    )
    legal_mask_mode: Optional[LegalMaskMode] = Field(
        None,
        description=(
            "Mode for storing legal move masks: None (no masks), "
            "'disjoint' (all legal from/to squares), or "
            "'conditional' (legal to-squares for target from-square only)"
        ),
    )

    @property
    def lmdb_filename(self) -> str:
        """Generate the LMDB output filename from the configuration name.

        Returns:
            The LMDB filename. If legal_mask_mode is set, returns
            "{name}_{mode}.lmdb", otherwise returns "{name}.lmdb".
        """
        if self.legal_mask_mode:
            return f"{self.name}_{self.legal_mask_mode}.lmdb"
        return f"{self.name}.lmdb"

    @model_validator(mode="after")
    def _check_unique_name(self) -> Self:
        """Validate that the config name is unique (case-insensitive)."""
        name_lower = self.name.lower()
        if name_lower in _CONFIG_REGISTRY:
            raise ValueError(
                f"A config with name '{self.name}' already exists. "
                "Config names must be unique (case-insensitive)."
            )
        _CONFIG_REGISTRY.add(name_lower)
        return self
