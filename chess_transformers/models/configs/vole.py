"""Configuration for the Vole model.

This module defines the model architecture configuration for Vole,
a chess transformer encoder with From-To square prediction.

Key Features:
    - Encoder-only transformer with From/To prediction heads.
    - 512-dimensional embeddings with 8 attention heads.
    - 6 transformer layers (~28M parameters).
    - Pre-configured DataLoader settings for training.

Notes:
    Import this config using the `import_config` utility function rather
    than importing directly, to ensure proper initialization.

Example:
    >>> from chess_transformers.utilities.configs import import_config
    >>> config = import_config("vole")
    >>> model = ChessTransformerEncoderFT(config)
"""

import pathlib

import torch.optim

from chess_transformers.data.levels import BOOL, PIECES, TURN, UCI_MOVES
from chess_transformers.models.configs.base import (
    DataLoaderConfig,
    ModelConfig,
    TrainConfig,
)
from chess_transformers.models.criteria import DualLabelSmoothedCE
from chess_transformers.models.models import ChessTransformerEncoderFT
from chess_transformers.train.datasets import ChessDatasetFT
from chess_transformers.train.schedules import get_warmup_cosine_schedule

# Model name and version (defined here to avoid self-reference in config)
MODEL_NAME = "Vole"
MODEL_VERSION = "1.0"

# Paths for logs and checkpoints
_project_root = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
_log_dir = _project_root / "logs"
_checkpoint_dir = _project_root / "checkpoints"


config = ModelConfig(
    model_type=ChessTransformerEncoderFT,
    name=MODEL_NAME,
    version=MODEL_VERSION,
    vocab_sizes={
        "moves": len(UCI_MOVES),
        "turn": len(TURN),
        "white_kingside_castling_rights": len(BOOL),
        "white_queenside_castling_rights": len(BOOL),
        "black_kingside_castling_rights": len(BOOL),
        "black_queenside_castling_rights": len(BOOL),
        "board_position": len(PIECES),
    },
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6,
    dropout=0.1,
    share_rel_pos_embeddings=False,
    disable_compilation=False,
    compilation_mode="default",
    dynamic_compilation=False,
    fullgraph_compilation=True,
    dataloader_config=DataLoaderConfig(
        dataset=ChessDatasetFT,
        batch_size=4096,
        num_workers=16,
        prefetch_factor=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        lmdb_filepath="${CT_DATA_FOLDER}/LE25ct/LE25ct.lmdb",
        source_name="LE25ct",
        n_rows=21_102_878,
        val_split_fraction=0.95,
        legal_mask_mode=None,
    ),
    train_config=TrainConfig(
        epochs=15,
        gradient_accumulation_steps=1,
        optimizer=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3, "weight_decay": 0.01},
        lr_schedule=get_warmup_cosine_schedule,
        lr_schedule_args={"warmup_steps": 4000},
        max_grad_norm=1.0,
        use_amp=True,
        loss_fn=DualLabelSmoothedCE,
        loss_fn_args={"eps": 0.1},
        log_dir=_log_dir,
        checkpoint_dir=_checkpoint_dir,
    ),
)
