"""Configuration for the LE25c dataset.

LE25c (Lichess Elite 2025 Checkmate) contains games from the Lichess Elite
Database from September 2013 through August 2025, filtered to include only
games that ended in checkmate.

Key Features:
    - ~179 million half-moves from winning players.
    - Games between 2400+ vs 2200+ Elo and 2500+ vs 2300+ Elo players.
    - Stored in 9 chunks for parallel processing.

Notes:
    See LE25ct for the same source with an additional 5+ minute time control
    filter. Use the shell script at chess_transformers/data/LE25c.sh to
    generate source .moves and .fens files.
"""

from chess_transformers.data.configs.base import DataConfig

config = DataConfig(
    name="LE25c",
    expected_rows=3_581_552 * 50,  # 3,581,552 games × ~50 half-moves per game
    val_split_fraction=0.98,
)
