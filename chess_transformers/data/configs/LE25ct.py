"""Configuration for the LE25ct dataset.

LE25ct (Lichess Elite 2025 Checkmate + Time control) contains games from
the Lichess Elite Database from September 2013 through August 2025, filtered
to include only games that ended in checkmate and used a time control of at
least 5 minutes.

Key Features:
    - ~19.8 million half-moves from winning players.
    - Games between 2400+ vs 2200+ Elo and 2500+ vs 2300+ Elo players.
    - Stored in 1 chunk for simpler processing.

Notes:
    See LE25c for the same source without the time control filter (larger
    dataset). Use the shell script at chess_transformers/data/LE25ct.sh to
    generate source .moves and .fens files.
"""

from chess_transformers.data.configs.base import DataConfig

config = DataConfig(
    name="LE25ct",
    expected_rows=396_881 * 50,  # 396,881 games × ~50 half-moves per game
    val_split_fraction=0.95,
    legal_mask_mode=None,  # Options: None, "disjoint", "conditional"
)
