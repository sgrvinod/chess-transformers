"""
Chess Transformer model for move prediction.

This module contains the ChessTransformerEncoderFT model, which predicts
the "From" square and "To" square separately for the next move.

Key Features:
    - ChessTransformerEncoderFT: Encoder-only transformer for From/To prediction.
    - Separate prediction heads for "From" and "To" squares.
    - Supports configurable model architecture via config objects.

Notes:
    The From/To decomposition was found to perform better than predicting
    the full UCI move string directly.
"""

import math
import torch
import torch.nn as nn

from typing import Dict, Tuple

from chess_transformers.models.modules import BoardEncoder
from chess_transformers.utilities.loggers import setup_logger
from chess_transformers.models.configs.base import ModelConfig

# Logger
logger = setup_logger(__file__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChessTransformerEncoderFT(nn.Module):
    """
    Chess Transformer (Encoder only) for predicting From/To squares.

    Predicts the next move by separately predicting:
    - From square: which of the 64 squares to move from
    - To square: which of the 64 squares to move to

    This decomposition was found to perform better than predicting
    the full UCI move string directly.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the model.

        Args:
            config: Configuration object with the following attributes:
                vocab_sizes (dict): Vocabulary sizes for embeddings
                d_model (int): Model dimension
                n_heads (int): Number of attention heads
                d_ff (int): Feed-forward hidden dimension
                n_layers (int): Number of transformer layers
                dropout (float): Dropout probability
                share_rel_pos_embeddings (bool, optional): Share relative position
                    embeddings across layers. Default: False.
        """
        super().__init__()

        self.code = "EFT"  # Encoder, From-To prediction

        # Store config values
        self.vocab_sizes = config.vocab_sizes
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.share_rel_pos_embeddings = getattr(
            config, "share_rel_pos_embeddings", False
        )

        # Board encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            dropout=self.dropout,
            share_rel_pos_embeddings=self.share_rel_pos_embeddings,
        )

        # Prediction heads for From and To squares
        # Each takes the 64 board position embeddings and outputs a score per square
        self.from_head = nn.Linear(self.d_model, 1, bias=False)
        self.to_head = nn.Linear(self.d_model, 1, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Xavier uniform initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=math.pow(self.d_model, -0.5)
                )

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - turns: Current turn, (batch, 1)
                - white_kingside_castling_rights: (batch, 1)
                - white_queenside_castling_rights: (batch, 1)
                - black_kingside_castling_rights: (batch, 1)
                - black_queenside_castling_rights: (batch, 1)
                - board_positions: Piece at each square, (batch, 64)

        Returns:
            from_logits: Logits for From square prediction, (batch, 1, 64)
            to_logits: Logits for To square prediction, (batch, 1, 64)
        """
        # Encode board state
        encoded = self.board_encoder(
            turns=batch["turns"],
            white_kingside_castling_rights=batch["white_kingside_castling_rights"],
            white_queenside_castling_rights=batch["white_queenside_castling_rights"],
            black_kingside_castling_rights=batch["black_kingside_castling_rights"],
            black_queenside_castling_rights=batch["black_queenside_castling_rights"],
            board_positions=batch["board_positions"],
        )  # (batch, 69, d_model)

        # Extract board square embeddings (positions 5-68)
        board_embeddings = encoded[:, 5:, :]  # (batch, 64, d_model)

        # Predict From and To squares
        # Apply linear head to each square's embedding, then transpose for (batch, 1, 64) shape
        from_logits = (
            self.from_head(board_embeddings).squeeze(-1).unsqueeze(1)
        )  # (batch, 1, 64)
        to_logits = (
            self.to_head(board_embeddings).squeeze(-1).unsqueeze(1)
        )  # (batch, 1, 64)

        return from_logits, to_logits

    def predict(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted From and To squares.

        Args:
            batch: Input batch dictionary.

        Returns:
            from_squares: Predicted From square indices, (batch,)
            to_squares: Predicted To square indices, (batch,)
        """
        from_logits, to_logits = self.forward(batch)
        from_squares = from_logits.squeeze(1).argmax(dim=-1)  # (batch,)
        to_squares = to_logits.squeeze(1).argmax(dim=-1)  # (batch,)
        return from_squares, to_squares


if __name__ == "__main__":
    # Test model instantiation and forward pass.
    import argparse
    from chess_transformers.utilities.configs import import_config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()

    config = import_config(args.config_name)
    model = config.model_type(config).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {n_params:,} learnable parameters.")
