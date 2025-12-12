"""Chess Transformer models package.

This package provides neural network models for chess move prediction,
including transformer-based architectures with various configurations.

Key Features:
    - ChessTransformerEncoderFT: Encoder-only model for From/To square prediction
    - Custom loss criteria with label smoothing
    - Modular components for building transformer architectures
"""

__all__ = ["criteria", "models", "modules"]


def __getattr__(name: str) -> type:
    """Lazy import to avoid circular import issues when running modules directly.

    This function is called when an attribute is accessed on the package,
    allowing for on-demand loading of submodules or specific classes.

    Args:
        name: Name of the attribute to retrieve.

    Returns:
        The requested module or class.

    Raises:
        AttributeError: If the attribute is not found in this module.
    """
    if name == "ChessTransformerEncoderFT":
        from .models import ChessTransformerEncoderFT

        return ChessTransformerEncoderFT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
