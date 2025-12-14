"""Base configuration classes for chess transformer models.

This module defines the base Pydantic models used to configure model
architecture and data loading parameters. All model-specific configurations
should inherit from or instantiate the classes defined here.

Key Features:
    - ModelConfig: Configuration for model architecture parameters.
    - DataLoaderConfig: Configuration for PyTorch DataLoader instantiation.
    - Type-safe configuration with Pydantic validation.
    - Unique name validation to prevent configuration conflicts.

Notes:
    Model config names are validated to be unique (case-insensitive) to prevent
    accidental configuration conflicts.

Example:
    Creating a model configuration::

        from chess_transformers.models.configs.base import ModelConfig, DataLoaderConfig

        config = ModelConfig(
            name="CT-EFT-20",
            vocab_sizes={
                "turn": 2,
                "white_kingside_castling_rights": 2,
                ...
            },
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_layers=6,
            dropout=0.1,
            dataloader=DataLoaderConfig(batch_size=256, num_workers=4),
        )
"""

import pathlib

from typing import Any, Callable, Dict, Optional, Self, Type

import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field, model_validator
from torch.utils.data import Dataset


# Registry to track all config names and prevent duplicates
_CONFIG_REGISTRY = set()


class DataLoaderConfig(BaseModel):
    """Configuration for PyTorch DataLoader instantiation.

    This Pydantic model defines the parameters needed to create a
    torch.utils.data.DataLoader for training or evaluation.

    Attributes:
        batch_size: Number of samples per batch. Larger batches typically
            provide more stable gradients but require more memory.
        num_workers: Number of subprocesses for data loading. Set to 0 for
            single-process loading (useful for debugging). Higher values
            can speed up data loading on multi-core systems.
        shuffle: Whether to shuffle the data at each epoch. Typically True
            for training, False for validation/testing.
        pin_memory: If True, the data loader will copy tensors into pinned
            memory before returning them. This can speed up host-to-GPU
            transfers. Should be True when training on GPU.
        drop_last: If True, drop the last incomplete batch if the dataset
            size is not divisible by batch_size. Useful for training to
            avoid variable batch sizes.
        prefetch_factor: Number of batches to prefetch per worker. Higher
            values can hide data loading latency but use more memory.
            Only used when num_workers > 0.
        persistent_workers: If True, worker processes are kept alive between
            epochs. Avoids worker startup overhead but uses more memory.
            Only used when num_workers > 0.

    Example:
        >>> config = DataLoaderConfig(
        ...     batch_size=256,
        ...     num_workers=4,
        ...     shuffle=True,
        ... )
        >>> dataloader = DataLoader(dataset, **config.to_dataloader_kwargs())
    """

    model_config = {"arbitrary_types_allowed": True}

    dataset: Type[Dataset] = Field(
        ..., description="Dataset class to use for loading data"
    )
    batch_size: int = Field(512, description="Number of samples per batch", gt=0)
    num_workers: int = Field(
        8, description="Number of subprocesses for data loading", ge=0
    )
    shuffle: bool = Field(True, description="Whether to shuffle data each epoch")
    pin_memory: bool = Field(
        False, description="Copy tensors to pinned memory for faster GPU transfer"
    )
    drop_last: bool = Field(False, description="Drop the last incomplete batch")
    prefetch_factor: Optional[int] = Field(
        2, description="Number of batches to prefetch per worker", ge=1
    )
    persistent_workers: bool = Field(
        True, description="Keep worker processes alive between epochs"
    )
    lmdb_filepath: Optional[str] = Field(
        None, description="Path to the LMDB database file for training data"
    )

    def to_dataloader_kwargs(self) -> dict:
        """Convert config to kwargs for torch.utils.data.DataLoader.

        Returns:
            A dictionary of keyword arguments suitable for passing to
            the DataLoader constructor.

        Note:
            prefetch_factor and persistent_workers are only included
            when num_workers > 0, as they have no effect otherwise.
        """
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        # These options only apply when using multiple workers
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return kwargs


class TrainConfig(BaseModel):
    """Configuration for training hyperparameters.

    This Pydantic model defines all training-related parameters including
    optimizer settings, learning rate scheduling, gradient handling,
    and checkpoint saving behavior.

    Attributes:
        epochs: Total number of training epochs.
        gradient_accumulation_steps: Number of forward passes to accumulate
            gradients before performing an optimizer step. Effective batch
            size is batch_size * gradient_accumulation_steps.
        optimizer: Optimizer class to use for training (e.g., torch.optim.AdamW).
        optimizer_args: Dictionary of arguments to pass to the optimizer.
        lr_schedule: Factory function that returns an lr_lambda callable.
            The factory is called with total_steps plus any args from
            lr_schedule_args. Example: get_warmup_cosine_schedule.
        lr_schedule_args: Arguments to pass to the lr_schedule factory,
            excluding total_steps which is computed at runtime.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        use_amp: Whether to use automatic mixed precision (AMP) for faster
            training on CUDA devices with reduced memory usage.
        loss_fn: Loss function class for training.
        loss_fn_args: Dictionary of arguments to pass to the loss function.
        epochs_per_log: Fraction of an epoch between logging intervals.
        log_dir: Directory path for saving training logs.
        checkpoint_dir: Directory path for saving model checkpoints.

    Example:
        >>> from chess_transformers.train.schedules import get_warmup_cosine_schedule
        >>> config = TrainConfig(
        ...     epochs=10,
        ...     optimizer=torch.optim.AdamW,
        ...     optimizer_args={"lr": 3e-4, "weight_decay": 0.01},
        ...     lr_schedule=get_warmup_cosine_schedule,
        ...     lr_schedule_args={"warmup_steps": 4000},
        ...     loss_fn=DualLabelSmoothedCE,
        ...     loss_fn_args={"eps": 0.1},
        ... )
    """

    model_config = {"arbitrary_types_allowed": True}

    epochs: int = Field(10, description="Number of training epochs", gt=0)
    gradient_accumulation_steps: int = Field(
        1, description="Gradient accumulation steps", gt=0
    )
    optimizer: Type[optim.Optimizer] = Field(
        ..., description="Optimizer class (e.g., torch.optim.AdamW)"
    )
    optimizer_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the optimizer"
    )
    lr_schedule: Callable = Field(
        ...,
        description="Factory function returning lr_lambda, called with total_steps + lr_schedule_args",
    )
    lr_schedule_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for lr_schedule factory (excluding total_steps)",
    )
    max_grad_norm: float = Field(
        1.0, description="Maximum gradient norm for clipping", gt=0
    )
    use_amp: bool = Field(True, description="Use automatic mixed precision")
    loss_fn: Type[nn.Module] = Field(
        ..., description="Loss function class (e.g., DualLabelSmoothedCE)"
    )
    loss_fn_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the loss function"
    )
    log_dir: Optional[pathlib.Path] = Field(
        None, description="Directory path for saving training logs"
    )
    checkpoint_dir: Optional[pathlib.Path] = Field(
        None, description="Directory path for saving model checkpoints"
    )

    @model_validator(mode="after")
    def _check_lr_schedule_signature(self) -> Self:
        """Validate that lr_schedule accepts a total_steps parameter."""
        import inspect

        sig = inspect.signature(self.lr_schedule)
        if "total_steps" not in sig.parameters:
            raise ValueError(
                f"lr_schedule function '{self.lr_schedule.__name__}' must accept "
                "a 'total_steps' parameter. See chess_transformers.train.schedules "
                "for valid schedule factory functions."
            )
        return self


class ModelConfig(BaseModel):
    """Configuration for chess transformer model architecture.

    This Pydantic model defines the parameters needed to instantiate
    a chess transformer model, including optional dataloader configuration.

    Attributes:
        model_type: The model class to instantiate (e.g., ChessTransformerEncoderFT).
        name: Unique identifier for this model configuration.
        vocab_sizes: Dictionary mapping token type names to vocabulary sizes.
            Required keys: "turn", "white_kingside_castling_rights",
            "white_queenside_castling_rights", "black_kingside_castling_rights",
            "black_queenside_castling_rights", "board_position".
        d_model: Model dimension (size of embedding vectors).
        n_heads: Number of attention heads in each transformer layer.
        d_ff: Feed-forward network hidden dimension.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout probability applied throughout the model.
        share_rel_pos_embeddings: If True, share relative position embeddings
            across all attention layers to reduce parameters.
        dataloader_config: Optional DataLoader configuration for training/evaluation.

    Example:
        >>> config = ModelConfig(
        ...     name="CT-EFT-20",
        ...     vocab_sizes={"turn": 2, "board_position": 13, ...},
        ...     d_model=512,
        ...     n_heads=8,
        ...     d_ff=2048,
        ...     n_layers=6,
        ...     dropout=0.1,
        ...     dataloader_config=DataLoaderConfig(batch_size=256),
        ... )
    """

    model_config = {"arbitrary_types_allowed": True}

    model_type: Type[nn.Module] = Field(..., description="Model class to instantiate")
    name: str = Field(..., description="Name and identifier for this configuration")
    version: str = Field("1.0", description="Version string for this configuration")
    vocab_sizes: Dict[str, int] = Field(
        ..., description="Vocabulary sizes for each token type"
    )
    d_model: int = Field(..., description="Model dimension (embedding size)")
    n_heads: int = Field(..., description="Number of attention heads")
    d_ff: int = Field(..., description="Feed-forward hidden dimension")
    n_layers: int = Field(..., description="Number of transformer layers")
    dropout: float = Field(0.0, description="Dropout probability")
    share_rel_pos_embeddings: bool = Field(
        False, description="Share relative position embeddings across layers"
    )
    disable_compilation: bool = Field(
        False, description="Disable torch.compile() for debugging"
    )
    compilation_mode: str = Field(
        "default",
        description="torch.compile() mode: default, reduce-overhead, max-autotune",
    )
    dynamic_compilation: bool = Field(
        False, description="Enable dynamic shapes for torch.compile()"
    )
    fullgraph_compilation: bool = Field(
        False, description="Enable fullgraph capture for torch.compile()"
    )
    dataloader_config: Optional[DataLoaderConfig] = Field(
        None, description="DataLoader configuration for training/evaluation"
    )
    train_config: Optional[TrainConfig] = Field(
        None, description="Training configuration for this model"
    )

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
