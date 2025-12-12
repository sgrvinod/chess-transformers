"""Learning rate schedule utilities.

This module provides factory functions for creating learning rate lambda
functions used with PyTorch's LambdaLR scheduler.

Key Features:
    - get_warmup_cosine_schedule: Linear warmup followed by cosine decay.
    - get_warmup_linear_schedule: Linear warmup followed by linear decay.
    - get_warmup_constant_schedule: Linear warmup followed by constant LR.

Notes:
    All schedule factory functions MUST accept a ``total_steps`` parameter,
    even if unused by the schedule logic. This ensures a consistent interface
    for the training script, which always passes ``total_steps`` when calling
    the factory. The returned functions are closures that capture the
    schedule parameters.

Example:
    >>> from chess_transformers.train.schedules import get_warmup_cosine_schedule
    >>> lr_lambda = get_warmup_cosine_schedule(warmup_steps=4000, total_steps=100000)
    >>> config = ModelConfig(
    ...     train_config=TrainConfig(
    ...         lr_scheduler=torch.optim.lr_scheduler.LambdaLR,
    ...         lr_scheduler_args={"lr_lambda": lr_lambda},
    ...     ),
    ... )
"""

import math

from typing import Callable


def get_warmup_cosine_schedule(
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    """Create a warmup + cosine decay learning rate schedule.

    Args:
        warmup_steps: Number of steps for linear warmup from 0 to base LR.
        total_steps: Total number of training steps.

    Returns:
        A callable that takes a step number and returns the LR multiplier.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def get_warmup_linear_schedule(
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    """Create a warmup + linear decay learning rate schedule.

    Args:
        warmup_steps: Number of steps for linear warmup from 0 to base LR.
        total_steps: Total number of training steps.

    Returns:
        A callable that takes a step number and returns the LR multiplier.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return lr_lambda


def get_warmup_constant_schedule(
    warmup_steps: int,
    total_steps: int,  # noqa: ARG001
) -> Callable[[int], float]:
    """Create a warmup + constant learning rate schedule.

    Args:
        warmup_steps: Number of steps for linear warmup from 0 to base LR.
        total_steps: Total number of training steps (unused, for consistent interface).

    Returns:
        A callable that takes a step number and returns the LR multiplier.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return lr_lambda
